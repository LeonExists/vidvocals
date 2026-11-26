"""Decision stage: per-window features (``extract.Extracted``) -> segments + speakers.

A pure function of the extracted features and the parameters (plus, for the Best preset,
an overlap resolver whose results are cached per piece), so the evaluation can re-score
parameter settings in seconds (``eval/tune.py``).
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from . import activity, gender, overlap

# Parameters of the decision logic (preset key "decision" and config settings override them).
DEFAULT_DECISION: dict[str, Any] = {
    "quiet_scale": 0.8,          # window logit factor for quiet events
    "speaker_k": 1.0,            # strength of the speaker-level shrinkage
    "decision_threshold": 0.5,   # keep if p_female >= this
    "unsure_policy": "model",    # model|keep|remove for low-confidence segments
    "low_conf_threshold": 0.6,   # confidence |2p-1| below this -> low_confidence
}


def kind_from_vad(vr: float) -> str:
    if vr >= 0.5:
        return "speech"
    if vr <= 0.2:
        return "nonverbal"
    return "mixed"


def _split_points(ev_windows, labels: np.ndarray, db: np.ndarray) -> list[tuple[float, int]]:
    """[(time, new_label)] at speaker changes inside one event (lowest-energy frame)."""
    pts = []
    for a, b in zip(ev_windows[:-1], ev_windows[1:]):
        la, lb = labels[a.idx], labels[b.idx]
        if la == lb:
            continue
        lo, hi = b.start, a.end
        if hi <= lo:
            t = (a.end + b.start) / 2
        else:
            i0, i1 = int(lo / activity.HOP_S), max(int(lo / activity.HOP_S) + 1, int(hi / activity.HOP_S))
            seg = db[i0:i1]
            t = (i0 + int(np.argmin(seg))) * activity.HOP_S if len(seg) else (lo + hi) / 2
        pts.append((t, lb))
    return pts


def window_evidence(ex, P: dict, prm: dict) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    """Fused window logits, confidences, kinds and per-window debug info."""
    windows, events = ex.windows, ex.events
    nW = len(windows)
    import copy

    weights = copy.deepcopy(P.get("fusion_weights") or gender.DEFAULT_WEIGHTS)
    for kd in ("speech", "nonverbal", "mixed"):
        for m in ("head", "hf", "tagger", "pitch"):
            if f"w_{m}_{kd}" in prm:
                weights.setdefault(kd, {})[m] = float(prm[f"w_{m}_{kd}"])
    use_tag_gender = bool(prm.get("tagger_gender", "tagger" in P.get("gender_models", [])))
    tag_rel_floor = float(prm.get("tagger_rel_floor", 0.0))
    tag = ex.tag
    wl, wc = np.zeros(nW), np.zeros(nW)
    wkind: list[str] = []
    dbg: list[dict] = []
    for i, w in enumerate(windows):
        ev = events[w.event]
        kind = kind_from_vad(ev.vad_ratio)
        if tag is not None:
            ps, pn = float(tag[i, 2]), float(tag[i, 3])
            if ps >= 0.2 and ps > 2 * pn:
                kind = "speech"
            elif pn > ps and ev.vad_ratio < 0.5:
                kind = "nonverbal"
        wkind.append(kind)
        evd: dict[str, tuple[float, float]] = {}
        if ex.p_head is not None:
            ph = float(ex.p_head[i])
            wa = float(prm.get("head_alt_w", 0.0))
            alt = ex.p_head_alt.get(str(prm.get("head_alt", "gender_head_ecapa")))
            if wa > 0 and alt is not None:
                ph = float(gender.sigmoid((1 - wa) * gender.logit(ph) + wa * gender.logit(float(alt[i]))))
            evd["head"] = (ph, 1.0)
        from . import features

        pp, rel = features.pitch_p_female(ex.pitch[i])
        evd["pitch"] = (pp, rel)
        if ex.p_hf is not None:
            evd["hf"] = (float(ex.p_hf[i]), 1.0 if kind == "speech" else 0.6)
        if tag is not None and use_tag_gender:
            fs, ms = float(tag[i, 0]), float(tag[i, 1])
            if fs + ms > 1e-3:
                pt = float(gender.sigmoid(gender.logit(fs, 1e-4) - gender.logit(ms, 1e-4)))
                evd["tagger"] = (pt, float(np.clip(max(tag[i, 2] * 1.5, tag_rel_floor), 0.0, 1.0)))
        l, _ = gender.fuse(evd, kind, weights, head_available=ex.p_head is not None)
        if ev.quiet:
            l *= float(prm["quiet_scale"])
        v0 = float(prm.get("unvoiced_v", 0.0))
        if v0 > 0:  # unvoiced / breathy windows: the gender cues are unreliable (models at chance)
            sc = float(prm.get("unvoiced_scale", 1.0))
            l *= sc + (1.0 - sc) * min(1.0, float(ex.pitch[i]["voiced"]) / v0)
        l = float(prm.get(f"cal_{kind}_a", 1.0)) * l + float(prm.get(f"cal_{kind}_b", 0.0))
        wl[i] = l
        wc[i] = abs(2 * gender.sigmoid(l) - 1)
        dbg.append({"kind": kind, "evidence": {k: [round(v[0], 3), round(v[1], 2)] for k, v in evd.items()},
                    "f0": round(ex.pitch[i]["f0_med"], 1), "voiced": round(ex.pitch[i]["voiced"], 2)})
    return wl, wc, wkind, dbg


def cluster_windows(E: np.ndarray, durs: np.ndarray, wl: np.ndarray, wc: np.ndarray, wkind: list[str],
                    prm: dict, P: dict) -> tuple[np.ndarray, np.ndarray]:
    """Speaker labels per window and each window's cosine similarity to its speaker centroid.

    ``cluster_on="speech"``: only speech windows >= ``cluster_min_dur`` are clustered (speaker
    embeddings are unreliable on moans, laughs, breaths); the other windows are assigned to the
    nearest centroid afterwards, and their similarity tells how much to trust that assignment.
    """
    nW = len(durs)
    thr = float(prm.get("cluster_threshold", P.get("cluster_threshold", 0.65)))
    if not nW or not E.shape[1]:
        return np.zeros(nW, int), np.ones(nW)
    X = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    sub = np.arange(nW)
    if prm.get("cluster_on", "all") == "speech":
        rel = np.array([k == "speech" and d >= float(prm.get("cluster_min_dur", 0.0)) for k, d in zip(wkind, durs)])
        if rel.sum() >= 4 and durs[rel].sum() >= float(prm.get("cluster_min_total_s", 6.0)):
            sub = np.flatnonzero(rel)
    lab_sub = gender.cluster_embeddings(E[sub], durs[sub], thr, window_logits=wl[sub], window_conf=wc[sub])
    ks = sorted(set(lab_sub.tolist()))
    C = np.stack([X[sub][lab_sub == k].mean(0) for k in ks])
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    S = X @ C.T
    labels = np.array([ks[j] for j in np.argmax(S, axis=1)], int)
    labels[sub] = lab_sub
    wsim = S[np.arange(nW), [ks.index(l) for l in labels]]
    return labels, wsim


def temporal_context(segs, seg_l, seg_c, seg_d, seg_emb, seg_reliable, prm):
    """Mix each segment's own logit with confident neighbours (gap <= ctx_gap s) whose embedding
    is similar: the same person tends to keep moaning / laughing / talking."""
    gap_max = float(prm.get("ctx_gap", 1.0))
    k = float(prm.get("ctx_k", 0.0))
    lo, hi = float(prm.get("ctx_sim_lo", 0.3)), float(prm.get("ctx_sim_hi", 0.6))
    only_unrel = bool(prm.get("ctx_only_unreliable", True))
    order = sorted(range(len(segs)), key=lambda i: segs[i]["start"])
    new_l, new_c = list(seg_l), list(seg_c)
    for pos, i in enumerate(order):
        if segs[i]["overlap"] or (only_unrel and seg_reliable[i]) or seg_emb[i] is None:
            continue
        num = den = 0.0
        for dirn in (-1, 1):
            q = pos + dirn
            while 0 <= q < len(order):
                j = order[q]
                gap = segs[i]["start"] - segs[j]["end"] if dirn < 0 else segs[j]["start"] - segs[i]["end"]
                if gap > gap_max:
                    break
                if not segs[j]["overlap"] and seg_emb[j] is not None:
                    sim = float(seg_emb[i] @ seg_emb[j])
                    f = float(np.clip((sim - lo) / max(1e-6, hi - lo), 0, 1))
                    w = k * f * min(1.0, seg_d[j] / 2.0) * (0.25 + seg_c[j])
                    num += w * seg_l[j]
                    den += w
                q += dirn
        if den > 0:
            a = min(1.0, seg_d[i] / 2.0) * (0.25 + seg_c[i])
            new_l[i] = (a * seg_l[i] + num) / (a + den)
            new_c[i] = abs(2 * float(gender.sigmoid(new_l[i])) - 1)
    return new_l, new_c


def decide(ex, P: dict, params: Optional[dict] = None,
           resolver: Optional[Callable[[list[dict], list[int]], dict[int, dict]]] = None,
           notes: Optional[list[str]] = None) -> dict:
    """Returns {segments, speakers, replacements, debug} (segments in the analysis.json shape)."""
    prm = {**DEFAULT_DECISION, **(P.get("decision") or {}), **(params or {})}
    notes = notes if notes is not None else []
    windows, events = ex.windows, ex.events
    nW = len(windows)
    E = ex.E
    db = ex.db
    wl, wc, wkind, wdbg = window_evidence(ex, P, prm)

    # ---------------------------------------------------------------- speakers
    durs = np.array([w.dur for w in windows])
    labels, wsim = cluster_windows(E, durs, wl, wc, wkind, prm, P)
    for ei in range(len(events)):
        idx = [w.idx for w in windows if w.event == ei]
        if len(idx) >= 3:
            labels[idx] = gender.smooth_labels(labels[idx], 3)

    raw_segs: list[dict] = []
    by_event: dict[int, list] = {}
    for w in windows:
        by_event.setdefault(w.event, []).append(w)
    for ei, ev in enumerate(events):
        ws = by_event.get(ei, [])
        cuts = _split_points(ws, labels, db) if len(ws) > 1 else []
        bounds = [ev.start] + [t for t, _ in cuts] + [ev.end]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a > 0.02:
                raw_segs.append({"start": a, "end": b, "event": ei})

    # ---------------------------------------------------------------- overlap detection
    ov_regions: list[tuple[float, float]] = []
    spk_stats = gender.speaker_posteriors(labels, durs, wl, wc) if nW else {}
    ov_mode = prm.get("overlap_mode") or P.get("overlap", "off")
    if ov_mode != "off" and nW and E.shape[1] and len(spk_stats) >= 2:
        op = overlap.OverlapParams(**{**(P.get("overlap_params") or {}),
                                      **{k[4:]: v for k, v in prm.items() if k.startswith("ovd_")}})
        cand = overlap.candidate_windows(E, labels, {k: s.logit for k, s in spk_stats.items()}, op, tagger=ex.tag)
        ov_regions = overlap.regions_from_windows(windows, cand, op)
    segs: list[dict] = []
    for s in raw_segs:
        pieces = [(s["start"], s["end"], False)]
        for (os_, oe) in ov_regions:
            nxt = []
            for a, b, ov in pieces:
                if ov or oe <= a or os_ >= b:
                    nxt.append((a, b, ov))
                    continue
                if os_ > a + 0.05:
                    nxt.append((a, os_, False))
                nxt.append((max(a, os_), min(b, oe), True))
                if oe < b - 0.05:
                    nxt.append((oe, b, False))
            pieces = nxt
        for a, b, ov in pieces:
            segs.append({"start": a, "end": b, "event": s["event"], "overlap": ov})

    # per-segment evidence from its windows
    starts_w = np.array([w.start for w in windows]) if nW else np.zeros(0)
    ends_w = np.array([w.end for w in windows]) if nW else np.zeros(0)
    seg_l, seg_c, seg_spk, seg_kind, seg_sim = [], [], [], [], []
    Xn = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9) if nW and E.shape[1] else None
    seg_emb = []
    for s in segs:
        ovl = np.clip(np.minimum(ends_w, s["end"]) - np.maximum(starts_w, s["start"]), 0, None) if nW else np.zeros(0)
        sel = np.flatnonzero(ovl > 0)
        if len(sel) == 0:
            seg_l.append(0.0); seg_c.append(0.0); seg_spk.append(0); seg_kind.append("mixed")
            seg_sim.append(0.0); seg_emb.append(None)
            continue
        seg_sim.append(float((ovl[sel] * wsim[sel]).sum() / ovl[sel].sum()))
        if Xn is not None:
            e_ = (ovl[sel, None] * Xn[sel]).sum(0)
            seg_emb.append(e_ / (np.linalg.norm(e_) + 1e-9))
        else:
            seg_emb.append(None)
        wts = ovl[sel] * (0.05 + wc[sel])
        l = float((wts * wl[sel]).sum() / wts.sum())
        seg_l.append(l)
        seg_c.append(abs(2 * gender.sigmoid(l) - 1))
        lab_w: dict[int, float] = {}
        for j in sel:
            lab_w[int(labels[j])] = lab_w.get(int(labels[j]), 0.0) + ovl[j]
        seg_spk.append(max(lab_w, key=lab_w.get))
        kinds: dict[str, float] = {}
        for j in sel:
            kinds[wkind[j]] = kinds.get(wkind[j], 0.0) + ovl[j]
        tot = sum(kinds.values())
        top = max(kinds, key=kinds.get)
        seg_kind.append(top if kinds[top] / tot >= 0.7 else "mixed")

    seg_d = np.array([s["end"] - s["start"] for s in segs])
    seg_reliable = np.array([seg_kind[i] == "speech" and seg_d[i] >= float(prm.get("attach_short_s", 0.0))
                             for i in range(len(segs))], bool)
    # temporal context: same-voice neighbours within ctx_gap seconds
    if float(prm.get("ctx_k", 0.0)) > 0 and len(segs) > 1:
        seg_l, seg_c = temporal_context(segs, seg_l, seg_c, seg_d, seg_emb, seg_reliable, prm)
    seg_l_a, seg_c_a = np.array(seg_l), np.array(seg_c)
    seg_lab = np.array(seg_spk, int)
    non_ov = np.array([not s["overlap"] for s in segs], bool)
    spk_final: dict[int, gender.SpeakerStats] = {}
    if len(segs):
        m = non_ov if non_ov.any() else np.ones(len(segs), bool)
        if prm.get("spk_stats_on", "all") == "reliable":
            # speakers' gender from reliable (speech, long) segments; others only if a speaker has none
            m2 = m & seg_reliable
            have = set(seg_lab[m2].tolist())
            m = m2 | (m & ~np.isin(seg_lab, list(have)))
        spk_final = gender.speaker_posteriors(seg_lab[m], seg_d[m], seg_l_a[m], seg_c_a[m])
        for k in set(seg_lab.tolist()) - set(spk_final):
            sel = seg_lab == k
            spk_final.update({kk: v for kk, v in gender.speaker_posteriors(
                seg_lab[sel], seg_d[sel], seg_l_a[sel], seg_c_a[sel]).items()})

    # ---------------------------------------------------------------- overlap resolution
    thr = float(prm["low_conf_threshold"])
    dthr = float(prm["decision_threshold"])
    policy = prm.get("overlap_policy") or P.get("overlap_policy") or "p_female"
    ov_results: dict[int, dict] = {}
    if ov_mode == "separate":
        smin = float(prm.get("ovl_scan_min_dur", P.get("overlap_scan_min_dur", 0.8)))
        scan_on = bool(prm.get("ovl_scan", True))
        kinds = ("speech", "mixed", "nonverbal") if prm.get("ovl_scan_kinds") == "all" else ("speech", "mixed")
        for i, s in enumerate(segs):
            s["scan"] = scan_on and (not s["overlap"]) and seg_kind[i] in kinds and \
                s["end"] - s["start"] >= smin
        todo = [i for i, s in enumerate(segs) if s["overlap"] or s.get("scan")]
        if todo and resolver is not None:
            try:
                ov_results = resolve_overlaps(resolver(segs, todo), segs, prm)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"overlap separation failed: {type(exc).__name__}: {exc}")

    out_segs, replacements, seg_dbg = [], [], []
    for i, s in enumerate(segs):
        spk = spk_final.get(seg_spk[i])
        dur = s["end"] - s["start"]
        flagged = bool(s["overlap"])
        k_eff = float(prm["speaker_k"])
        if seg_kind[i] != "speech":
            k_eff *= float(prm.get("speaker_k_nonverbal", 1.0))
        if not seg_reliable[i] and prm.get("attach_lo") is not None:
            lo, hi = float(prm["attach_lo"]), float(prm.get("attach_hi", float(prm["attach_lo"]) + 0.1))
            k_eff *= float(np.clip((seg_sim[i] - lo) / max(1e-6, hi - lo), 0.0, 1.0))
        l_plain = gender.combine_with_speaker(seg_l[i], seg_c[i], dur, spk, k=k_eff)
        l_plain = float(prm.get("final_a", 1.0)) * l_plain + float(prm.get("final_b", 0.0))
        r = ov_results.get(i)
        if r and r.get("replacement") and not flagged and \
                float(gender.sigmoid(l_plain)) < float(prm.get("ovl_scan_min_p", 0.0)):
            r = None  # scan found a "female" source in a confidently male segment: not trusted
        if r and r.get("replacement"):
            s["overlap"] = True
        l_fin = l_plain if not s["overlap"] else seg_l[i]
        p = float(gender.sigmoid(l_fin))
        conf = abs(2 * p - 1)
        label = "female" if p >= 0.5 else "male"
        keep = p >= dthr
        decided_by_policy = False
        is_rep = False
        if s["overlap"]:
            label = "overlap"
            if r and r.get("replacement"):
                keep, is_rep = True, True
                p, conf = r["p_female"], r["confidence"]
            elif r and r.get("decision") in ("keep", "remove"):
                keep = r["decision"] == "keep"
                p, conf = r["p_female"], r["confidence"]
            else:
                keep = {"keep": True, "remove": False}.get(policy, p >= dthr)
                conf = min(conf, 0.5)
                decided_by_policy = policy in ("keep", "remove")
        low = bool(conf < thr)
        if low and not decided_by_policy and not is_rep and prm["unsure_policy"] in ("keep", "remove"):
            keep = prm["unsure_policy"] == "keep"
        dbg = {"l_seg": round(float(seg_l[i]), 3), "c_seg": round(float(seg_c[i]), 3),
               "spk": int(seg_spk[i]), "spk_logit": round(spk.logit, 3) if spk else None,
               "spk_purity": round(spk.purity, 3) if spk else None, "l_fin": round(float(l_fin), 3),
               "l_plain": round(float(l_plain), 3), "sim": round(seg_sim[i], 3), "k_eff": round(k_eff, 3),
               "reliable": bool(seg_reliable[i]), "overlap_region": flagged, "replacement": is_rep}
        base = {"speaker": spk.id if spk else "S1", "kind": seg_kind[i]}
        # sub-pieces: a replaced segment is split where the separated female source is silent
        parts = [(s["start"], s["end"], label, p, conf, low, keep, is_rep)]
        if is_rep and r.get("remove_spans"):
            parts = []
            t = s["start"]
            for a_, b_ in r["remove_spans"] + [(s["end"], s["end"])]:
                a_, b_ = max(a_, s["start"]), min(b_, s["end"])
                if a_ - t > 0.02:
                    parts.append((t, a_, "overlap", p, conf, low, True, True))
                if b_ - a_ > 0.02:
                    pm = float(r.get("p_male_source", 0.0))
                    parts.append((a_, b_, "male", pm, abs(2 * pm - 1), abs(2 * pm - 1) < thr, False, False))
                t = max(t, b_)
        rep_ids = []
        for (a_, b_, lab_, p_, c_, low_, keep_, rep_) in parts:
            sid = len(out_segs) + 1
            if rep_:
                rep_ids.append(sid)
            seg_dbg.append(dbg)
            out_segs.append({
                "id": sid, "start": round(a_, 3), "end": round(b_, 3), **base, "label": lab_,
                "p_female": round(p_, 4), "confidence": round(c_, 4), "low_confidence": bool(low_),
                "keep": bool(keep_), "override": None,
            })
        if is_rep and rep_ids:
            for piece in r["pieces"]:
                if piece.get("wav"):
                    replacements.append({"start": piece["start"], "end": piece["end"], "wav": piece["wav"],
                                         "segment": rep_ids[0], "segments": rep_ids})
    for s in out_segs:
        k = next((kk for kk, v in spk_final.items() if v.id == s["speaker"]), None)
        if k is not None:
            spk_final[k].n_segments += 1
    speakers = [{"id": v.id, "p_female": round(v.p_female, 4),
                 "label": "female" if v.p_female >= 0.5 else "male",
                 "total_dur": round(v.total_dur, 3), "n_segments": v.n_segments, "override": None}
                for k, v in sorted(spk_final.items())]
    debug = {
        "windows": [{"start": round(w.start, 3), "end": round(w.end, 3), "speaker": int(labels[w.idx]),
                     "logit": round(float(wl[w.idx]), 3), **wdbg[w.idx]} for w in windows],
        "overlap_regions": ov_regions,
        "segments": seg_dbg,
        "overlap_results": {str(k): v for k, v in ov_results.items()},
    }
    return {"segments": out_segs, "speakers": speakers, "replacements": replacements, "debug": debug,
            "low_conf_threshold": thr, "params": prm}


def resolve_overlaps(pieces: dict[int, list[dict]], segs: list[dict], prm: dict) -> dict[int, dict]:
    """Per segment: decide from the separated sources of its pieces.

    ``pieces[i]`` = [{start, end, ps: [p_female per source], rms: [...], wavs: [file per source]}]
    (from the 2-speaker separator).  A piece with one female and one male source (and both
    sources clearly present) gets a female-only replacement.
    """
    results: dict[int, dict] = {}
    for i, plist in pieces.items():
        if not plist:
            continue
        flagged = segs[i]["overlap"]
        if flagged:
            pf_min, pm_max, db_min = prm.get("ovl_flag_pf", 0.5), prm.get("ovl_flag_pm", 0.5), prm.get("ovl_flag_db", -20.0)
        else:
            pf_min, pm_max, db_min = prm.get("ovl_scan_pf", 0.65), prm.get("ovl_scan_pm", 0.35), prm.get("ovl_scan_db", -12.0)
        out_pieces, confirmed = [], False
        for pc0 in plist:
            ps, rms = np.asarray(pc0["ps"], float), np.asarray(pc0["rms"], float)
            rel_db = 20 * np.log10(rms / max(rms.max(), 1e-12) + 1e-12)
            fi, mi = int(np.argmax(ps)), int(np.argmin(ps))
            pc = {"start": pc0["start"], "end": pc0["end"], "p_sources": [round(float(q), 3) for q in ps],
                  "rel_db": [round(float(r), 1) for r in rel_db]}
            if fi != mi and ps[fi] >= pf_min and ps[mi] < pm_max and rel_db.min() > db_min:
                pc.update(wav=pc0["wavs"][fi], p_female=float(ps[fi]),
                          confidence=float(min(abs(2 * ps[fi] - 1), abs(2 * ps[mi] - 1))))
                confirmed = True
            else:
                dom = int(np.argmax(rms))
                q = float(ps[dom]) if rel_db.min() < -10 else float(np.max(ps))
                pc.update(p_female=q, confidence=float(abs(2 * q - 1)))
            out_pieces.append(pc)
        if confirmed:
            good = [pc for pc in out_pieces if pc.get("wav")]
            results[i] = {"replacement": True, "pieces": out_pieces,
                          "p_female": float(np.mean([pc["p_female"] for pc in good])),
                          "confidence": float(np.min([pc["confidence"] for pc in good]))}
            if prm.get("ovl_split_db") is not None:
                spans, pms = [], []
                for pc0, pc in zip(plist, out_pieces):
                    if not pc.get("wav") or pc0.get("env") is None:
                        continue
                    fi = pc["p_sources"].index(max(pc["p_sources"]))
                    env = np.asarray(pc0["env"], float)  # (2, frames) dB, 10 ms
                    df, dm = env[fi], env[1 - fi]
                    pms.append(pc["p_sources"][1 - fi])
                    off = (df < dm - float(prm["ovl_split_db"])) | (df < df.max() - 35.0)
                    spans += _runs_to_spans(off, pc0["start"], float(prm.get("ovl_split_min_s", 0.3)))
                if spans:
                    results[i]["remove_spans"] = _merge_spans(spans)
                    results[i]["p_male_source"] = float(np.mean(pms)) if pms else 0.0
        elif flagged:
            q = float(np.mean([pc["p_female"] for pc in out_pieces]))
            results[i] = {"decision": "keep" if q >= 0.5 else "remove", "p_female": q,
                          "confidence": float(abs(2 * q - 1)), "pieces": out_pieces}
        else:
            results[i] = {"scan_only": True, "pieces": out_pieces}
    return results


def _runs_to_spans(mask: np.ndarray, t0: float, min_s: float, hop: float = 0.01) -> list[tuple[float, float]]:
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(t0 + a * hop, t0 + b * hop) for a, b in zip(st, en) if (b - a) * hop >= min_s]


def _merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]
