"""API flow with a fake analysis step: upload -> review -> overrides -> export -> download."""
import json
import shutil
import subprocess
import time
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from voicefilter import config, media, pipeline
from voicefilter.jobs import JobStore, sanitize_filename
from voicefilter.server import create_app, lan_ipv4


def fake_analyze(path, preset, work_dir, progress=None):
    work = Path(work_dir)
    if progress:
        progress("extract", 0.1, "extracting audio")
    media.extract_audio(path, work / "audio.wav")
    x, sr = sf.read(str(work / "audio.wav"), dtype="float32", always_2d=True)
    sf.write(str(work / "vocals.wav"), x, sr, subtype="FLOAT")
    sf.write(str(work / "background.wav"), np.zeros_like(x), sr, subtype="FLOAT")
    if progress:
        progress("separate", 0.5, "separating")
    dur = len(x) / sr
    an = {"duration": dur, "low_conf_threshold": 0.6,
          "segments": [
              {"id": 1, "start": 0.2, "end": 0.9, "speaker": "S1", "kind": "speech", "label": "female",
               "p_female": 0.9, "confidence": 0.8, "low_confidence": False, "keep": True, "override": None},
              {"id": 2, "start": 1.0, "end": 1.8, "speaker": "S2", "kind": "nonverbal", "label": "male",
               "p_female": 0.1, "confidence": 0.8, "low_confidence": False, "keep": False, "override": None}],
          "speakers": [
              {"id": "S1", "p_female": 0.9, "label": "female", "total_dur": 0.7, "n_segments": 1, "override": None},
              {"id": "S2", "p_female": 0.1, "label": "male", "total_dur": 0.8, "n_segments": 1, "override": None}]}
    (work / "replacements.json").write_text("[]")
    (work / "peaks.json").write_text(json.dumps(media.compute_peaks(work / "vocals.wav")))
    media.encode_preview(work / "audio.wav", work / "original.m4a")
    media.encode_preview(work / "vocals.wav", work / "vocals.m4a")
    pipeline.render_preview(work, an)
    (work / "analysis.json").write_text(json.dumps(an))
    return an


FAKE = types.SimpleNamespace(analyze_file=fake_analyze, render_from_analysis=pipeline.render_from_analysis,
                             render_preview=pipeline.render_preview, export_video=pipeline.export_video)


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    d = tmp_path_factory.mktemp("api")
    p = d / "My clip (1).mp4"
    subprocess.run([config.FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=2",
                    "-f", "lavfi", "-i", "sine=f=440:d=2:sample_rate=48000", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-c:a", "aac", "-shortest", str(p)], check=True)
    return p


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setitem(config.SETTINGS, "preview_debounce_s", 0.05)
    store = JobStore(root=tmp_path / "jobs", pipeline=FAKE)
    app = create_app(store=store)
    with TestClient(app) as c:
        yield c, store


def wait_state(c, jid, states, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["state"] in states:
            return j
        time.sleep(0.1)
    raise AssertionError(f"timeout, job is {j}")


def test_info(client):
    c, _ = client
    r = c.get("/api/info").json()
    assert set(r) >= {"version", "device", "gpu_name", "lan_urls", "presets"}
    assert [p["id"] for p in r["presets"]] == ["fast", "balanced", "best"]
    assert all(set(p) == {"id", "name", "description"} for p in r["presets"])


def test_full_flow(client, video):
    c, store = client
    with open(video, "rb") as f:
        r = c.post("/api/jobs", files={"file": (video.name, f, "video/mp4")},
                   data={"preset": "fast", "keep_background": "true", "auto_export": "false"})
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    j = wait_state(c, jid, ("review", "error"))
    assert j["state"] == "review", j
    assert set(j) == {"id", "filename", "preset", "keep_background", "state", "stage", "progress", "message",
                      "error", "duration", "created", "has_output"}
    assert j["filename"] == "My clip (1).mp4" and j["has_output"] is False
    assert any(x["id"] == jid for x in c.get("/api/jobs").json())

    an = c.get(f"/api/jobs/{jid}/analysis").json()
    assert set(an) >= {"duration", "segments", "speakers", "low_conf_threshold"}
    pk = c.get(f"/api/jobs/{jid}/peaks").json()
    assert pk["rate"] == 50 and len(pk["peaks"]) > 50
    for kind in ("original", "vocals", "preview"):
        r = c.get(f"/api/jobs/{jid}/audio/{kind}")
        assert r.status_code == 200 and r.headers["content-type"] == "audio/mp4"
    r = c.get(f"/api/jobs/{jid}/audio/preview", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    assert c.get(f"/api/jobs/{jid}/audio/nope").status_code == 404

    prev_mtime = (store.job_dir(jid) / "preview.m4a").stat().st_mtime_ns
    r = c.post(f"/api/jobs/{jid}/overrides", json={"segments": {"2": "keep"}, "speakers": {"S1": "remove"}})
    assert r.status_code == 200
    an2 = r.json()
    assert an2["segments"][1]["override"] == "keep" and an2["speakers"][0]["override"] == "remove"
    assert c.get(f"/api/jobs/{jid}/analysis").json()["segments"][1]["override"] == "keep"
    assert c.post(f"/api/jobs/{jid}/overrides", json={"segments": {"99": "keep"}}).status_code == 400
    assert c.post(f"/api/jobs/{jid}/overrides", json={"segments": {"1": "maybe"}}).status_code == 400
    t0 = time.time()
    while (store.job_dir(jid) / "preview.m4a").stat().st_mtime_ns == prev_mtime and time.time() - t0 < 20:
        time.sleep(0.1)
    assert (store.job_dir(jid) / "preview.m4a").stat().st_mtime_ns != prev_mtime

    assert c.get(f"/api/jobs/{jid}/download").status_code == 409
    r = c.post(f"/api/jobs/{jid}/export")
    assert r.status_code == 200 and r.json()["state"] == "rendering"
    j = wait_state(c, jid, ("done", "error"))
    assert j["state"] == "done" and j["has_output"], j
    r = c.get(f"/api/jobs/{jid}/download")
    assert r.status_code == 200
    cd = r.headers["content-disposition"]
    assert "My clip (1).voicefilter.mp4" in cd
    out = store.job_dir(jid) / "dl.mp4"
    out.write_bytes(r.content)
    info = media.probe(out)
    assert info["has_video"] and info["audio"]

    # overrides: segment 2 kept (override), S1 removed -> 0.2..0.9 silent, 1.0..1.8 audible
    wav = pipeline.render_from_analysis(store.job_dir(jid), c.get(f"/api/jobs/{jid}/analysis").json(), False)
    y, sr = sf.read(wav, always_2d=True)
    assert np.abs(y[int(0.3 * sr):int(0.8 * sr)]).max() < 1e-4
    assert np.abs(y[int(1.1 * sr):int(1.7 * sr)]).max() > 0.05  # lavfi sine amplitude is 1/8

    assert c.delete(f"/api/jobs/{jid}").status_code == 200
    assert not store.job_dir(jid).exists()
    assert c.get(f"/api/jobs/{jid}").status_code == 404


def test_auto_export_and_errors(client, video, tmp_path):
    c, store = client
    with open(video, "rb") as f:
        jid = c.post("/api/jobs", files={"file": ("a.mp4", f)}, data={"preset": "balanced", "auto_export": "true"}).json()["id"]
    j = wait_state(c, jid, ("done", "error"))
    assert j["state"] == "done"
    assert c.post("/api/jobs", files={"file": ("a.mp4", b"x")}, data={"preset": "nope"}).status_code == 400
    # broken media -> error state with a message
    jid = c.post("/api/jobs", files={"file": ("bad.mp4", b"not a video at all")}, data={"preset": "fast"}).json()["id"]
    j = wait_state(c, jid, ("error", "review"))
    assert j["state"] == "error" and j["error"]
    assert c.get("/api/jobs/doesnotexist").status_code == 404


def test_restart_recovery(tmp_path):
    root = tmp_path / "jobs"
    s = JobStore(root=root, pipeline=FAKE, start_worker=False)
    jid, d = s.new_job_dir()
    (d / "input.mp4").write_bytes(b"x")
    s.create(jid, "x.mp4", d / "input.mp4", "fast")
    jid2, d2 = s.new_job_dir()
    (d2 / "input.mp4").write_bytes(b"x")
    s.create(jid2, "y.mp4", d2 / "input.mp4", "fast")
    s.update(jid2, state="analyzing")
    s2 = JobStore(root=root, pipeline=FAKE, start_worker=False)
    assert s2.get(jid)["state"] == "queued" and s2.q.qsize() == 1
    assert s2.get(jid2)["state"] == "error" and "interrupted" in s2.get(jid2)["error"]


def test_sanitize_and_lan():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("C:\\x\\a<b>.mp4") == "a_b_.mp4"
    assert sanitize_filename("con.mp4") == "_con.mp4"
    assert all(not ip.startswith(("127.", "169.254.")) for ip in lan_ipv4())
