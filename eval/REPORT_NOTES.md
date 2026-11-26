## Dev / test split for tuning (fixed 2026-11-26, before any tuning change)

- **dev** (all tuning and parameter selection): mix_00, mix_02, mix_04, mix_05, mix_06, mix_08
- **test** (scored once at the end, never iterated on): mix_01, mix_03, mix_07, mix_09, mix_10, mix_11
- Both halves contain one female-only mix (mix_04 dev / mix_10 test) and one male-only mix (mix_05 dev /
  mix_11 test; mix_11 also has sung vocals in the background music). The split is also in `eval/splits.json`
  under `"tuning"`.

## Baseline findings (hand-written, 2026-11-26; numbers below are the baseline before tuning)

- **Which mixes**: all `oraclesep_*`, `oracle*` and `random` rows use all 12 mixes (17.0 min, 292 truth
  events). Real-separation rows: `fast_subset` = mix_00 + mix_01 only (3.0 min), `*_smoke` = the 10 s smoke
  mix only. Balanced/best with real separation were deliberately not run on the full set (CPU budget:
  separation runs ~9-35x slower than real time here, and the decision logic is being tuned).
- **Separation is not the main error source on clean-ish mixes**: on mix_00 real separation and oracle
  separation give the same decisions (87.4 % frame accuracy both). It does matter on mix_01 (reverb +
  environmental noise, 0.8 dB SNR): 68.2 % real vs 80.8 % oracle.
- **With perfect stems (oraclesep_fast)**: 87.9 % frame accuracy, 17.2 % male leak, 6.9 % female loss;
  21 % of vocal time is flagged low-confidence and accuracy on the rest is 95.2 %. Errors concentrate in
  non-verbal sounds (31 % of female non-verbal time lost, 33 % of male non-verbal time leaked vs 5 % / 19 %
  for speech), breathy sounds and whisper (52 % / 31 %), short events < 0.4 s (58 % of female time lost) and
  overlaps (40 % of male time inside overlapping events leaked).
- **Decisions lean towards "keep"** (male leak > female loss in every run), so the low-confidence default
  and speaker-level shrinkage are the first knobs to tune. `best` keeps more overlap (78.6 %) at the price of
  more male leak (25.1 %).
- **Gating ceiling**: even a perfect frame-level gate (`oracle_active`) cannot remove the male voice inside
  overlaps (male residual 0 dB there); only the overlap separation in `best` can, and in the baseline it
  lowers the residual in overlaps by only ~1.5 dB vs fast.
- **Speed (CPU, 4 shared cores, no GPU)**: fast with real separation RTF 10.5 (mix_00) / 11.6 (mix_01),
  of which separation is ~9.5x real time; with cached or oracle stems fast / balanced / best run at RTF
  1.1 / 1.3 / 1.7. On the smoke mix balanced and best with real separation (overlap 2 / 4) took RTF 38 / 66.
- **Cached real-separation stems** exist for fast (`num_overlap` 1): mix_00, mix_01, smoke; for balanced
  (`num_overlap` 2) and best (`num_overlap` 4) only for smoke. Stems are not shared between presets because
  the separation parameters differ (the stamp check refuses them).
