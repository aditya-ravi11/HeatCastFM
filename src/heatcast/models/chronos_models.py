"""Chronos family forecasters (Ansari et al., 2024; Chronos-2, 2025).

Chronos-T5    scales and quantizes the series into tokens and samples future tokens
              from a T5 language model, one step at a time (generative sample paths).
Chronos-Bolt  T5 encoder-decoder that emits 9 quantiles for all steps at once.
Chronos-2     encoder-only model with group attention; takes past and known future
              covariates and emits 21 quantiles.
"""
import numpy as np
import torch

from ..probability import LEVELS, quantiles_from_cdf_grid, quantiles_from_samples

BOLT_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def _device():
    import os

    forced = os.environ.get("HEATCAST_DEVICE")
    if forced:
        return forced
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _ctx(arr, s_idx, t_pos, L):
    offs = np.arange(-L + 1, 1)
    return arr[s_idx[:, None], t_pos[:, None] + offs]


def _fut(arr, s_idx, t_pos, H):
    return arr[s_idx[:, None], t_pos[:, None] + np.arange(1, H + 1)]


class Chronos2:
    """Chronos-2 zero-shot. covariates: None, 'basic' (Tmin, season, normal) or 'rh' (basic + humidity)."""

    def __init__(self, model_id: str, covariates: str | None = None, context: int = 512,
                 batch: int = 256, pipeline=None, label: str | None = None):
        from chronos import Chronos2Pipeline

        self.pipe = pipeline or Chronos2Pipeline.from_pretrained(model_id, device_map=_device())
        self.cov = covariates
        self.L, self.batch = context, batch
        self.name = label or {None: "Chronos-2", "basic": "Chronos-2+cov", "rh": "Chronos-2+cov+RH"}[covariates]

    def fit(self, store, cfg):
        return self

    def _inputs(self, store, s_idx, t_pos, H):
        L = self.L
        tgt = _ctx(store.tmax_in, s_idx, t_pos, L)
        if self.cov is None:
            return torch.tensor(tgt[:, None, :])
        n_s = store.tmax_in.shape[0]
        sin = np.broadcast_to(store.doy_sin, (n_s, len(store.dates)))
        cos = np.broadcast_to(store.doy_cos, (n_s, len(store.dates)))
        items = []
        tmin = _ctx(store.tmin_in, s_idx, t_pos, L)
        sin_c, cos_c = _ctx(sin, s_idx, t_pos, L), _ctx(cos, s_idx, t_pos, L)
        nor_c = _ctx(store.normal, s_idx, t_pos, L)
        sin_f, cos_f = _fut(sin, s_idx, t_pos, H), _fut(cos, s_idx, t_pos, H)
        nor_f = _fut(store.normal, s_idx, t_pos, H)
        rh = _ctx(store.rh_in, s_idx, t_pos, L) if self.cov == "rh" else None
        for i in range(len(s_idx)):
            past = {"tmin": tmin[i], "doy_sin": sin_c[i], "doy_cos": cos_c[i], "normal": nor_c[i]}
            if rh is not None:
                past["rh"] = rh[i]
            items.append({
                "target": tgt[i],
                "past_covariates": past,
                "future_covariates": {"doy_sin": sin_f[i], "doy_cos": cos_f[i], "normal": nor_f[i]},
            })
        return items

    def predict(self, store, s_idx, t_pos, H):
        out = []
        for i in range(0, len(s_idx), self.batch):
            inp = self._inputs(store, s_idx[i:i + self.batch], t_pos[i:i + self.batch], H)
            q, _ = self.pipe.predict_quantiles(inp, prediction_length=H, quantile_levels=LEVELS.tolist(),
                                               batch_size=self.batch)
            out.append(torch.stack([x[0] for x in q]).float().cpu().numpy())
        return np.concatenate(out)


class ChronosBolt:
    name = "Chronos-Bolt"

    def __init__(self, model_id: str, context: int = 512, batch: int = 512):
        from chronos import ChronosBoltPipeline

        self.pipe = ChronosBoltPipeline.from_pretrained(model_id, device_map=_device())
        self.L, self.batch = context, batch

    def fit(self, store, cfg):
        return self

    def predict(self, store, s_idx, t_pos, H):
        out = []
        for i in range(0, len(s_idx), self.batch):
            ctx = torch.tensor(_ctx(store.tmax_in, s_idx[i:i + self.batch], t_pos[i:i + self.batch], self.L))
            q, _ = self.pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=BOLT_LEVELS.tolist())
            out.append(q.float().cpu().numpy())
        q9 = np.concatenate(out)
        # Bolt only emits 0.1..0.9; outer levels come from exponential tails fitted to those
        return quantiles_from_cdf_grid(q9, BOLT_LEVELS, LEVELS).astype("float32")


def t5_sample(pipe, context: torch.Tensor, H: int, S: int) -> torch.Tensor:
    """Same sampling as ChronosPipeline.predict, but the encoder runs once per series.

    The stock pipeline repeats every context S times before the encoder, which is S times
    the work and memory. Here the encoder states are computed once and repeated instead.
    Returns (B, S, H) in the original units.
    """
    from transformers import GenerationConfig
    from transformers.modeling_outputs import BaseModelOutput

    cfg = pipe.model.config
    t5 = pipe.model.model
    token_ids, mask, scale = pipe.tokenizer.context_input_transform(context)
    dev = pipe.model.device
    token_ids, mask = token_ids.to(dev), mask.to(dev)
    with torch.no_grad():
        enc = t5.get_encoder()(input_ids=token_ids, attention_mask=mask).last_hidden_state
        enc = enc.repeat_interleave(S, dim=0)
        mask_r = mask.repeat_interleave(S, dim=0)
        preds = t5.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc),
            attention_mask=mask_r,
            generation_config=GenerationConfig(
                min_new_tokens=H, max_new_tokens=H, do_sample=True, num_return_sequences=1,
                eos_token_id=cfg.eos_token_id, pad_token_id=cfg.pad_token_id,
                temperature=cfg.temperature, top_k=cfg.top_k, top_p=cfg.top_p,
            ),
        )
    preds = preds[..., 1:].reshape(context.shape[0], S, -1)  # drop the decoder start token
    return pipe.tokenizer.output_transform(preds.cpu(), scale)


class ChronosT5:
    """Generative sampling: returns quantiles and keeps the raw sample paths."""

    name = "Chronos-T5"

    def __init__(self, model_id: str, context: int = 512, batch: int = 8, num_samples: int = 50):
        from chronos import ChronosPipeline

        self.pipe = ChronosPipeline.from_pretrained(model_id, device_map=_device(), torch_dtype=torch.float32)
        self.L, self.batch, self.S = context, batch, num_samples
        self.last_samples = None

    def fit(self, store, cfg):
        return self

    def predict(self, store, s_idx, t_pos, H):
        torch.manual_seed(0)
        out = []
        for i in range(0, len(s_idx), self.batch):
            ctx = torch.tensor(_ctx(store.tmax_in, s_idx[i:i + self.batch], t_pos[i:i + self.batch], self.L))
            smp = t5_sample(self.pipe, ctx, H, self.S)  # (B, S, H)
            out.append(smp.float().cpu().numpy().transpose(0, 2, 1))  # (B, H, S)
        samples = np.concatenate(out).astype("float16")
        self.last_samples = samples
        return quantiles_from_samples(samples.astype("float32")).astype("float32")
