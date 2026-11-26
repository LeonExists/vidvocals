# voicefilter/assets

Small trained artifacts used by the backend. No pickle anywhere: weights are plain numpy arrays in
`.npz` files loaded with `np.load(..., allow_pickle=False)`.

| file | what |
|---|---|
| `gender_head.py` | loader + feature code: `load_head()`, `GenderHead.compute_features()`, `GenderHead.predict_proba()`, module-level `compute_features()` / `predict_proba()` for the default head |
| `gender_head.npz` | default head: ECAPA-TDNN embedding (TaoRuijie, 192-d) + 22 pitch/spectral features, logistic regression, Platt-calibrated |
| `gender_head_nofeat.npz` | fallback head without any embedding (22 handcrafted features only); use when no speaker model is available |
| `ecapa_tao.py` | the ECAPA-TDNN speaker encoder the default head expects (`load_ecapa()`, `embed()`); weights (66 MB, MIT) are downloaded from GitHub on first use into `models/ext/ECAPA-TDNN/exps/pretrain.model` (override with `VOICEFILTER_ECAPA_TAO=<path>` or `VOICEFILTER_MODELS_DIR`) |

## Usage

```python
from voicefilter.assets.gender_head import load_head
from voicefilter.assets.ecapa_tao import load_ecapa, embed

head = load_head()                      # gender_head.npz
enc = load_ecapa(device="cuda")         # only if head.embedding_kind == "ecapa_tao_vox2_192"
emb = embed(enc, wav_16k_mono)          # raw 192-d embedding; the head L2-normalises it itself
x = head.compute_features(wav_16k_mono, 16000, emb)
p_female = head.predict_proba(x)        # float, calibrated p(female)
```

`compute_features(wav, sr, embedding)` returns `[L2-normalised embedding | 22 handcrafted features]` and
computes the handcrafted part exactly as in training (pyin F0 60-1000 Hz with 64 ms frames / 10 ms hop,
active frames within 40 dB of the loudest, spectral centroid / roll-off / flatness / ZCR medians, MFCC 1-12
means). Input may be any sample rate (it is resampled to 16 kHz) and mono or stereo. For a head with
`embedding_dim == 0` pass `embedding=None`.

The heads were trained on segments from 0.25 s to 3 s; for longer segments average `p_female` over
~1.5-3 s windows (or compute the logit per window and average) rather than feeding minutes of audio.

## npz format

| key | dtype | meaning |
|---|---|---|
| `feature_names` | str[d] | names of the d input features (embedding dims first, then handcrafted) |
| `embedding_kind` | str | `ecapa_tao_vox2_192`, `wavlm_base_plus_L..._mean` or `none` |
| `embedding_dim` | int | 192 / 768 / 0 |
| `model_type` | str | `logreg` (`param_w`[d], `param_b`) or `mlp` (`param_W1`[d,h], `param_b1`[h], `param_w2`[h], `param_b2`) |
| `mean`, `std` | float64[d] | standardisation applied before the model |
| `calib_a`, `calib_b` | float64 | Platt calibration: `p = sigmoid(calib_a * logit + calib_b)` |
| `info_json` | str | training metadata (features, C, counts, seed, trainer script) |

Retrain with `python eval/train_gender_head.py` (needs the features from `eval/extract_features.py`);
numbers are in `eval/GENDER_MODELS.md`. Training data (Deeply non-verbal set) is CC BY-NC-ND, so these
weights are for non-commercial use, see `eval/DATA.md`.
