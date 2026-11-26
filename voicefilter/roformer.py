"""BS-RoFormer and Mel-Band RoFormer (inference only).

Self-contained re-implementation compatible with the checkpoints trained with
lucidrains/BS-RoFormer and ZFTurbo/Music-Source-Separation-Training (both MIT licensed),
which is what the public vocal RoFormer checkpoints on Hugging Face use.  Kept free of
extra dependencies (no beartype / rotary-embedding-torch / einops) so that it installs
everywhere and does not break when those packages change their parameter names.

State-dict layout (must match the checkpoints)::

    layers.{i}.{j}.layers.{k}.0  Attention (norm.gamma, to_qkv, to_gates, to_out.0,
                                  rotary_embed.freqs)
    layers.{i}.{j}.layers.{k}.1  FeedForward (net.0.gamma, net.1, net.4)
    layers.{i}.{j}.norm.gamma    transformer output norm (older checkpoints) or
    final_norm.gamma             (newer BS-RoFormer checkpoints)
    band_split.to_features.{b}.{0: RMSNorm, 1: Linear}
    mask_estimators.{n}.to_freqs.{b}.0.{2*l: Linear}
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_FREQS_PER_BANDS = ((2,) * 24 + (4,) * 12 + (12,) * 8 + (12,) * 8 + (24,) * 8
                           + (48,) * 8 + (128, 129))


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class RotaryEmbedding(nn.Module):
    """Interleaved rotary embedding (same math/param as rotary-embedding-torch)."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        self.freqs = nn.Parameter(freqs, requires_grad=False)
        self._cache: dict = {}

    def _cos_sin(self, n: int, device, dtype):
        key = (n, device, dtype)
        c = self._cache.get(key)
        if c is None:
            t = torch.arange(n, device=device, dtype=torch.float32)
            f = torch.einsum("i,j->ij", t, self.freqs.float().to(device))
            f = f.repeat_interleave(2, dim=-1)
            c = (f.cos().to(dtype), f.sin().to(dtype))
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[key] = c
        return c

    def rotate(self, x):  # x: (..., n, d)
        n, d = x.shape[-2], x.shape[-1]
        cos, sin = self._cos_sin(n, x.device, x.dtype)
        rd = cos.shape[-1]
        xr, xp = x[..., :rd], x[..., rd:]
        x1 = xr[..., 0::2]
        x2 = xr[..., 1::2]
        rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        out = xr * cos + rot * sin
        return torch.cat([out, xp], dim=-1) if xp.shape[-1] else out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.Sequential(RMSNorm(dim), nn.Linear(dim, inner), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, rotary_embed: Optional[RotaryEmbedding] = None):
        super().__init__()
        self.heads = heads
        inner = heads * dim_head
        self.rotary_embed = rotary_embed
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(inner, dim, bias=False), nn.Dropout(0.0))

    def forward(self, x):
        x = self.norm(x)
        b, n, _ = x.shape
        qkv = self.to_qkv(x).view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.rotary_embed is not None:
            q = self.rotary_embed.rotate(q)
            k = self.rotary_embed.rotate(k)
        out = F.scaled_dot_product_attention(q, k, v)
        gates = self.to_gates(x)  # b n h
        out = out * gates.permute(0, 2, 1).unsqueeze(-1).sigmoid()
        out = out.permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, *, dim, depth, dim_head=64, heads=8, ff_mult=4, norm_output=True,
                 rotary_embed=None):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([Attention(dim, heads, dim_head, rotary_embed), FeedForward(dim, ff_mult)])
            for _ in range(depth)])
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class BandSplit(nn.Module):
    def __init__(self, dim, dim_inputs: Sequence[int]):
        super().__init__()
        self.dim_inputs = tuple(dim_inputs)
        self.to_features = nn.ModuleList([nn.Sequential(RMSNorm(d), nn.Linear(d, dim)) for d in dim_inputs])

    def forward(self, x):
        xs = x.split(self.dim_inputs, dim=-1)
        return torch.stack([f(s) for s, f in zip(xs, self.to_features)], dim=-2)


def _mlp(dim_in, dim_out, dim_hidden, depth):
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class MaskEstimator(nn.Module):
    """``depth`` = number of hidden layers of the per-band MLP.

    Note: the BS-RoFormer and Mel-Band code bases interpret ``mask_estimator_depth``
    differently (depth-1 vs depth hidden layers); :func:`build_and_load` reads the real
    number from the checkpoint.
    """

    def __init__(self, dim, dim_inputs: Sequence[int], depth, mlp_expansion_factor=4):
        super().__init__()
        hidden = dim * mlp_expansion_factor
        self.to_freqs = nn.ModuleList([nn.Sequential(_mlp(dim, d * 2, hidden, depth), nn.GLU(dim=-1))
                                       for d in dim_inputs])

    def forward(self, x):
        xs = x.unbind(dim=-2)
        return torch.cat([m(b) for b, m in zip(xs, self.to_freqs)], dim=-1)


class _RoformerBase(nn.Module):
    """Shared transformer stack + STFT handling."""

    def _build_common(self, dim, depth, stereo, num_stems, time_transformer_depth,
                      freq_transformer_depth, dim_head, heads, stft_n_fft, stft_hop_length,
                      stft_win_length, stft_normalized, final_norm, ff_mult=4):
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        tkw = dict(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, norm_output=not final_norm)
        time_rot = RotaryEmbedding(dim_head)
        freq_rot = RotaryEmbedding(dim_head)
        self.layers = nn.ModuleList([
            nn.ModuleList([Transformer(depth=time_transformer_depth, rotary_embed=time_rot, **tkw),
                           Transformer(depth=freq_transformer_depth, rotary_embed=freq_rot, **tkw)])
            for _ in range(depth)])
        self.final_norm = RMSNorm(dim) if final_norm else None
        self.stft_kwargs = dict(n_fft=stft_n_fft, hop_length=stft_hop_length, win_length=stft_win_length,
                                normalized=stft_normalized)
        self.stft_win_length = stft_win_length
        self.n_freqs = stft_n_fft // 2 + 1

    def _window(self, device):
        return torch.hann_window(self.stft_win_length, device=device)

    def _stft(self, raw):  # raw: b s t -> stft b (f s) t c, window
        b, s, t = raw.shape
        win = self._window(raw.device)
        with torch.autocast(device_type=raw.device.type, enabled=False):
            spec = torch.stft(raw.float().reshape(b * s, t), **self.stft_kwargs, window=win,
                              return_complex=True)
        spec = torch.view_as_real(spec)  # (b s) f t c
        f, n = spec.shape[1], spec.shape[2]
        spec = spec.reshape(b, s, f, n, 2).permute(0, 2, 1, 3, 4).reshape(b, f * s, n, 2)
        return spec, win

    def _axial(self, x):  # x: b t f d
        b, t, f, d = x.shape
        for time_tr, freq_tr in self.layers:
            x = x.permute(0, 2, 1, 3).reshape(b * f, t, d)
            x = time_tr(x)
            x = x.reshape(b, f, t, d).permute(0, 2, 1, 3).reshape(b * t, f, d)
            x = freq_tr(x)
            x = x.reshape(b, t, f, d)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x

    def _istft(self, spec_c, win, length, batch):  # spec_c: b n (f s) t complex
        s = self.audio_channels
        b, n, fs, t = spec_c.shape
        spec_c = spec_c.reshape(b, n, fs // s, s, t).permute(0, 1, 3, 2, 4).reshape(b * n * s, fs // s, t)
        with torch.autocast(device_type=spec_c.device.type, enabled=False):
            audio = torch.istft(spec_c, **self.stft_kwargs, window=win, return_complex=False, length=length)
        return audio.reshape(batch, n, s, -1)


class BSRoformer(_RoformerBase):
    def __init__(self, dim, *, depth, stereo=False, num_stems=1, time_transformer_depth=2,
                 freq_transformer_depth=2, freqs_per_bands=DEFAULT_FREQS_PER_BANDS, dim_head=64,
                 heads=8, stft_n_fft=2048, stft_hop_length=512, stft_win_length=2048,
                 stft_normalized=False, mask_estimator_depth=2, mlp_expansion_factor=4,
                 final_norm=False, **_ignored):
        super().__init__()
        self._build_common(dim, depth, stereo, num_stems, time_transformer_depth, freq_transformer_depth,
                           dim_head, heads, stft_n_fft, stft_hop_length, stft_win_length,
                           stft_normalized, final_norm)
        assert len(freqs_per_bands) > 1 and sum(freqs_per_bands) == self.n_freqs, \
            f"freqs_per_bands must sum to {self.n_freqs}"
        dims = tuple(2 * f * self.audio_channels for f in freqs_per_bands)
        self.band_split = BandSplit(dim, dims)
        self.mask_estimators = nn.ModuleList([MaskEstimator(dim, dims, mask_estimator_depth, mlp_expansion_factor)
                                              for _ in range(num_stems)])

    def forward(self, raw):  # raw: b s t
        if raw.ndim == 2:
            raw = raw.unsqueeze(1)
        b, s, t = raw.shape
        spec, win = self._stft(raw)  # b F n 2
        x = spec.permute(0, 2, 1, 3).reshape(b, spec.shape[2], -1)  # b n (F c)
        x = self.band_split(x)
        x = self._axial(x)
        mask = torch.stack([m(x) for m in self.mask_estimators], dim=1)  # b n t (F c)
        mask = mask.reshape(b, self.num_stems, mask.shape[2], -1, 2).permute(0, 1, 3, 2, 4).float()
        spec_c = torch.view_as_complex(spec.float().contiguous()).unsqueeze(1)
        out = spec_c * torch.view_as_complex(mask.contiguous())
        audio = self._istft(out, win, t, b)
        return audio[:, 0] if self.num_stems == 1 else audio


class MelBandRoformer(_RoformerBase):
    def __init__(self, dim, *, depth, stereo=False, num_stems=1, time_transformer_depth=2,
                 freq_transformer_depth=2, num_bands=60, dim_head=64, heads=8, sample_rate=44100,
                 stft_n_fft=2048, stft_hop_length=512, stft_win_length=2048, stft_normalized=False,
                 mask_estimator_depth=1, mlp_expansion_factor=4, final_norm=False, **_ignored):
        super().__init__()
        self._build_common(dim, depth, stereo, num_stems, time_transformer_depth, freq_transformer_depth,
                           dim_head, heads, stft_n_fft, stft_hop_length, stft_win_length,
                           stft_normalized, final_norm)
        import librosa

        mel = torch.from_numpy(librosa.filters.mel(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands))
        mel[0][0] = 1.0
        mel[-1, -1] = 1.0
        fpb = mel > 0
        assert fpb.any(dim=0).all(), "all frequencies need to be covered by the mel bands"
        freq_idx = torch.arange(self.n_freqs).unsqueeze(0).expand(num_bands, -1)[fpb]
        if stereo:
            freq_idx = (freq_idx.unsqueeze(-1) * 2 + torch.arange(2)).reshape(-1)
        self.register_buffer("freq_indices", freq_idx, persistent=False)
        nfb = fpb.sum(dim=1)
        self.register_buffer("num_bands_per_freq", fpb.sum(dim=0), persistent=False)
        dims = tuple(2 * int(f) * self.audio_channels for f in nfb.tolist())
        self.band_split = BandSplit(dim, dims)
        self.mask_estimators = nn.ModuleList([MaskEstimator(dim, dims, mask_estimator_depth, mlp_expansion_factor)
                                              for _ in range(num_stems)])

    def forward(self, raw):
        if raw.ndim == 2:
            raw = raw.unsqueeze(1)
        b, s, t = raw.shape
        spec, win = self._stft(raw)  # b (f s) n 2
        x = spec[:, self.freq_indices]  # b F' n 2
        x = x.permute(0, 2, 1, 3).reshape(b, x.shape[2], -1)
        x = self.band_split(x)
        x = self._axial(x)
        masks = torch.stack([m(x) for m in self.mask_estimators], dim=1)  # b n t (F' c)
        masks = masks.reshape(b, self.num_stems, masks.shape[2], -1, 2).permute(0, 1, 3, 2, 4).float()
        masks = torch.view_as_complex(masks.contiguous())  # b n F' t
        spec_c = torch.view_as_complex(spec.float().contiguous())  # b (f s) t
        n_t = spec_c.shape[-1]
        idx = self.freq_indices.view(1, 1, -1, 1).expand(b, self.num_stems, -1, n_t)
        summed = torch.zeros((b, self.num_stems) + spec_c.shape[1:], dtype=masks.dtype, device=masks.device)
        summed.scatter_add_(2, idx, masks)
        denom = self.num_bands_per_freq.repeat_interleave(self.audio_channels).view(1, 1, -1, 1)
        out = spec_c.unsqueeze(1) * (summed / denom.clamp(min=1e-8))
        audio = self._istft(out, win, t, b)
        return audio[:, 0] if self.num_stems == 1 else audio


def build_from_config(model_type: str, cfg: dict) -> nn.Module:
    """model_type: 'mel_band_roformer' | 'bs_roformer'; cfg: the yaml 'model' section."""
    cfg = dict(cfg)
    if model_type in ("mel_band_roformer", "melband", "mel_band"):
        return MelBandRoformer(**cfg)
    if model_type in ("bs_roformer", "bs"):
        if "freqs_per_bands" in cfg:
            cfg["freqs_per_bands"] = tuple(cfg["freqs_per_bands"])
        return BSRoformer(**cfg)
    raise ValueError(f"unknown roformer type {model_type}")


def load_checkpoint(model: nn.Module, path: str, strict_report: bool = True) -> dict:
    """Load weights, tolerating the known key variants.  Returns {'missing', 'unexpected'}."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file

        sd = load_file(str(path))
    else:
        sd = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    # RMSNorm param naming variants
    own = model.state_dict()
    fixed = {}
    for k, v in sd.items():
        if k not in own and k.endswith(".weight") and k[:-7] + ".gamma" in own:
            k = k[:-7] + ".gamma"
        fixed[k] = v
    res = model.load_state_dict(fixed, strict=False)
    missing = [k for k in res.missing_keys if not k.endswith("rotary_embed.freqs")]
    unexpected = list(res.unexpected_keys)
    if strict_report and (missing or unexpected):
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:8]} (n={len(missing)}), "
                           f"unexpected={unexpected[:8]} (n={len(unexpected)})")
    return {"missing": missing, "unexpected": unexpected}


def _read_state_dict(path: str) -> dict:
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file

        sd = load_file(str(path))
    else:
        sd = torch.load(str(path), map_location="cpu", weights_only=False)
    for key in ("state_dict", "model"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def build_and_load(model_type: str, cfg: dict, ckpt_path: str) -> nn.Module:
    """Build the model matching ``cfg`` and the checkpoint's key layout, load the weights."""
    sd = _read_state_dict(ckpt_path)
    cfg = dict(cfg)
    if int(cfg.get("linear_transformer_depth", 0) or 0) > 0:
        raise ValueError("linear_transformer_depth > 0 is not supported")
    cfg["final_norm"] = any(k.startswith("final_norm.") for k in sd)
    pref = "mask_estimators.0.to_freqs.0.0."
    n_lin = len({k[len(pref):].split(".")[0] for k in sd if k.startswith(pref) and k.endswith(".weight")})
    if n_lin:
        cfg["mask_estimator_depth"] = n_lin - 1
    model = build_from_config(model_type, cfg)
    own = model.state_dict()
    fixed = {}
    for k, v in sd.items():
        if k not in own and k.endswith(".weight") and k[:-7] + ".gamma" in own:
            k = k[:-7] + ".gamma"
        fixed[k] = v
    res = model.load_state_dict(fixed, strict=False)
    missing = [k for k in res.missing_keys if not k.endswith("rotary_embed.freqs")]
    if missing or res.unexpected_keys:
        raise RuntimeError(f"checkpoint {ckpt_path} does not match the {model_type} config: "
                           f"missing={missing[:6]} (n={len(missing)}), "
                           f"unexpected={list(res.unexpected_keys)[:6]} (n={len(res.unexpected_keys)})")
    model.eval()
    return model
