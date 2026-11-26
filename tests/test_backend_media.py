"""media.py: extraction, mux keeps the video bit-identical, durations, A/V offset."""
import subprocess

import numpy as np
import pytest
import soundfile as sf

from voicefilter import config, media

FF = config.FFMPEG


def ff(*args):
    subprocess.run([FF, "-v", "error", "-y", *args], check=True)


def vmd5(path):
    r = subprocess.run([FF, "-v", "error", "-i", str(path), "-map", "0:v", "-c", "copy", "-f", "md5", "-"],
                       capture_output=True, check=True)
    return r.stdout.strip()


def decode(path, sr=48000):
    """Decode the first audio stream placed on the file timeline (leading gap as silence)."""
    r = subprocess.run([FF, "-v", "error", "-i", str(path), "-map", "0:a:0", "-af",
                        "aresample=async=1:first_pts=0", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32)


def onset(x, thr=0.05):
    return int(np.argmax(np.abs(x) > thr))


@pytest.fixture(scope="module")
def srcdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    ff("-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=4", "-c:v", "libx264", "-g", "12",
       "-pix_fmt", "yuv420p", str(d / "v.mp4"))
    # audio: 0.3 s silence then a burst, so the onset is measurable
    ff("-f", "lavfi", "-i", "anoisesrc=d=3.5:c=pink:a=0.3:r=48000", "-af", "adelay=300|300",
       "-ac", "2", "-c:a", "aac", "-b:a", "128k", str(d / "a.m4a"))
    ff("-i", str(d / "v.mp4"), "-i", str(d / "a.m4a"), "-map", "0:v", "-map", "1:a", "-c", "copy",
       str(d / "sync.mp4"))
    ff("-i", str(d / "v.mp4"), "-itsoffset", "0.5", "-i", str(d / "a.m4a"), "-map", "0:v", "-map", "1:a",
       "-c", "copy", str(d / "late.mp4"))
    (d / "s.srt").write_text("1\n00:00:00,500 --> 00:00:02,000\nhello\n", encoding="utf-8")
    ff("-i", str(d / "v.mp4"), "-i", str(d / "a.m4a"), "-i", str(d / "s.srt"), "-map", "0:v", "-map", "1:a",
       "-map", "2:s", "-c:v", "copy", "-c:a", "copy", "-c:s", "srt", str(d / "subs.mkv"))
    # click track for exact timing checks in other containers / codecs
    ff("-f", "lavfi", "-i", "aevalsrc='if(between(t,0.5,0.52),0.8*sin(2*PI*1000*t),0)':s=48000:d=3.7",
       str(d / "click.wav"))
    for name, ac in (("pcm.mov", "pcm_s16le"), ("mp3.mkv", "libmp3lame"), ("pcm.mkv", "pcm_s16le"),
                     ("pcm.avi", "pcm_s16le"), ("mp2.ts", "mp2"), ("aac.mkv", "aac")):
        ff("-i", str(d / "v.mp4"), "-i", str(d / "click.wav"), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
           "-c:a", ac, str(d / name))
    ff("-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=2", "-f", "lavfi",
       "-i", "sine=f=300:d=2:sample_rate=44100", "-c:v", "libvpx-vp9", "-deadline", "realtime",
       "-b:v", "200k", "-c:a", "libopus", "-ac", "1", str(d / "mono.webm"))
    return d


def roundtrip(srcdir, name, tmp_path, out_ext=None):
    src = srcdir / name
    info = media.probe(src)
    wav = tmp_path / "x.wav"
    media.extract_audio(src, wav, info)
    out = tmp_path / ("out" + (out_ext or src.suffix))
    media.export_video(src, wav, out, info)
    return src, out, info, wav


@pytest.mark.parametrize("name", ["sync.mp4", "late.mp4", "subs.mkv", "mono.webm"])
def test_video_bit_identical_and_duration(srcdir, tmp_path, name):
    src, out, info, wav = roundtrip(srcdir, name, tmp_path)
    assert vmd5(src) == vmd5(out)
    a, b = decode(src), decode(out)
    assert abs(len(a) - len(b)) <= 1024  # decoded length within one AAC frame of the source
    oi = media.probe(out)
    assert oi["has_video"] and oi["audio"]
    assert abs((oi["duration"] or 0) - (info["duration"] or 0)) < 0.05


@pytest.mark.parametrize("name", ["pcm.mov", "mp3.mkv", "pcm.mkv", "pcm.avi", "mp2.ts", "aac.mkv"])
def test_containers_length_and_timing(srcdir, tmp_path, name):
    """Decoded output length within one AAC frame of the source and the click at the same time
    (encoder delay compensated where the container does not signal it)."""
    src, out, info, _ = roundtrip(srcdir, name, tmp_path)
    assert vmd5(src) == vmd5(out)
    a, b = decode(src), decode(out)
    assert abs(len(a) - len(b)) <= 1024, (len(a), len(b))
    assert abs(onset(a) - onset(b)) <= 48, (onset(a), onset(b))  # <= 1 ms


def test_av_offset_preserved(srcdir, tmp_path):
    src, out, info, _ = roundtrip(srcdir, "late.mp4", tmp_path)
    assert media.audio_offset(info) == pytest.approx(0.5 - 1024 / 48000, abs=0.01)
    a, b = decode(src), decode(out)
    assert abs(onset(a) - onset(b)) <= 48  # <= 1 ms
    # container start_time may differ by the AAC encoder priming (1024 samples), which the
    # decoder skips: the presented audio (checked above) is what matters
    oi = media.probe(out)
    assert abs(media.audio_offset(oi) - media.audio_offset(info)) <= 1024 / 48000 + 0.002


def test_subtitles_copied(srcdir, tmp_path):
    _, out, _, _ = roundtrip(srcdir, "subs.mkv", tmp_path)
    assert media.probe(out)["subtitles"]


def test_extract_mono_and_format(srcdir, tmp_path):
    wav = tmp_path / "m.wav"
    media.extract_audio(srcdir / "mono.webm", wav)
    i = sf.info(str(wav))
    assert i.samplerate == 44100 and i.channels == 2 and i.subtype == "FLOAT"
    x, _ = sf.read(str(wav))
    assert np.allclose(x[:, 0], x[:, 1])


def test_no_audio_error(srcdir, tmp_path):
    with pytest.raises(media.MediaError, match="no audio"):
        media.extract_audio(srcdir / "v.mp4", tmp_path / "x.wav")


def test_audio_only_roundtrip(srcdir, tmp_path):
    src, out, info, _ = roundtrip(srcdir, "a.m4a", tmp_path)
    oi = media.probe(out)
    assert not oi["has_video"] and oi["audio"]["codec"] == "aac"


def test_preview_and_peaks(srcdir, tmp_path):
    wav = tmp_path / "x.wav"
    media.extract_audio(srcdir / "sync.mp4", wav)
    p = media.encode_preview(wav, tmp_path / "p.m4a")
    assert media.probe(p)["audio"]["codec"] == "aac"
    pk = media.compute_peaks(wav, rate=50)
    n = sf.info(str(wav)).frames
    assert pk["rate"] == 50 and abs(len(pk["peaks"]) - n / 44100 * 50) <= 1
    assert 0 <= min(pk["peaks"]) and max(pk["peaks"]) <= 1
    assert max(pk["peaks"][:10]) < 0.05 < max(pk["peaks"][20:])  # 0.3 s leading silence
