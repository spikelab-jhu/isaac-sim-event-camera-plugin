"""
noise.py
========

Credits & References
--------------------
The noise model here follows the mechanisms characterised in v2e and ESIM.
--------------------

It hooks the event model in two places:
  1. ``thresholds()`` — per-pixel ON/OFF thresholds (fixed-pattern mismatch),
     used in place of the scalar threshold when deciding which pixels fire.
  2. ``apply()`` — after the signal masks are computed: inject background/leak/
     shot/hot-pixel events, then enforce the refractory dead time.

Intensity note: shot noise rises in the dark, so ``apply`` scales it by a
darkness factor from the (linear) intensity. With the default HDR event source
the intensity is linear radiance ~[0,1+]; for LDR set ``intensity_scale`` to the
LDR white level (≈255) so "dark" is measured correctly.
"""
from dataclasses import dataclass

import torch


@dataclass
class DVSNoiseCfg:
    """Noise parameters (all per-pixel rates are Hz)."""
    sigma_threshold: float = 0.03      # per-pixel threshold std, log units (fixed-pattern mismatch)
    on_off_ratio: float = 1.0          # mean theta_on / theta_off (ON/OFF asymmetry; 1.0 = symmetric)
    refractory_s: float = 1.0e-3       # dead time after a pixel fires (seconds)
    leak_rate_hz: float = 0.1          # per-pixel steady ON "leak" event rate
    shot_rate_hz: float = 0.1          # per-pixel shot-noise rate at full darkness (scaled down in bright regions)
    hot_pixel_frac: float = 1.0e-4     # fraction of pixels that are "hot" (stuck firing)
    hot_pixel_rate_hz: float = 300.0   # event rate of a hot pixel
    intensity_scale: float = 1.0       # divide intensity by this before the darkness factor (LDR: ~255)
    min_threshold: float = 0.01        # clamp per-pixel thresholds to at least this
    # intensity-dependent photoreceptor bandwidth (v2e-style 1st-order IIR lowpass on
    # the log frame; effective cutoff = cutoff_hz * inten01, so dark pixels respond
    # slower). 0 = OFF (the exact clean signal path).
    cutoff_hz: float = 0.0             # 3 dB cutoff at FULL intensity (inten01 = 1)
    bandwidth_floor: float = 0.02      # min inten01, keeps zero-light pixels from freezing
    seed: int = 0


class DVSNoiseModel:
    """Stateful per-pixel noise model for ONE camera (all envs). Lazily sized on
    the first call so it adapts to the (num_envs, H, W) it actually sees."""

    def __init__(self, cfg: DVSNoiseCfg, threshold: float):
        self.cfg = cfg
        self.threshold = threshold          # nominal (scalar) threshold from the processor
        self.gen = None                     # seeded RNG (device-bound, made on first use)
        self.theta_on = None                # (H, W) per-pixel thresholds (broadcast over envs)
        self.theta_off = None
        self.hot_mask = None                # (H, W) bool
        self.last_t = None                  # (num_envs, H, W) last fire time, for refractory
        self.prev_t = None
        # intensity-dependent bandwidth state (see bandwidth_filter)
        self.lp_log = None                  # (num_envs, H, W) lowpassed log frame
        self.prev_t_bw = None
        self._bw_reset_ids = None           # env ids whose lp state re-seeds on the next call

    # ── intensity-dependent photoreceptor bandwidth (hook 0) ─────
    def reset_envs(self, env_ids):
        """Re-seed the bandwidth filter state for reset envs on their next frame."""
        if self.cfg.cutoff_hz > 0.0 and len(env_ids) > 0:
            self._bw_reset_ids = env_ids if self._bw_reset_ids is None else \
                torch.cat([self._bw_reset_ids, env_ids])

    def bandwidth_filter(self, log_frame, intensity, current_time):
        """
        The photoreceptor's bandwidth scales with photocurrent: bright pixels track
        the signal, dark pixels respond slowly. Discrete update (v2e emulator_utils
        ``low_pass_filter``):

            tau  = 1 / (2*pi*cutoff_hz)
            eps  = clamp(inten01 * dt / tau, max=1)
            lp   = (1 - eps) * lp + eps * log_frame

        so the effective cutoff is ``cutoff_hz * inten01``. ``inten01`` is the
        intensity normalized by ``intensity_scale``, floored at ``bandwidth_floor``
        so zero-light pixels do not freeze. ``cutoff_hz <= 0`` returns the input
        untouched (exact clean path).
        """
        if self.cfg.cutoff_hz <= 0.0:
            return log_frame
        if self.lp_log is None:                        # first frame seeds the state
            self.lp_log = log_frame.clone()
            self.prev_t_bw = current_time
            return log_frame
        if self._bw_reset_ids is not None:             # reset envs snap to the new frame
            self.lp_log[self._bw_reset_ids] = log_frame[self._bw_reset_ids]
            self._bw_reset_ids = None
        dt = max(0.0, current_time - self.prev_t_bw)
        self.prev_t_bw = current_time
        if dt > 0.0:
            tau = 1.0 / (2.0 * torch.pi * self.cfg.cutoff_hz)
            inten01 = (intensity / self.cfg.intensity_scale).clamp(self.cfg.bandwidth_floor, 1.0)
            eps = (inten01 * (dt / tau)).clamp(max=1.0)
            self.lp_log = (1.0 - eps) * self.lp_log + eps * log_frame
        return self.lp_log

    # ── fixed-pattern per-pixel thresholds (hook 1) ──────────────
    def thresholds(self, hw, device):
        """Return (theta_on, theta_off), each (H, W), sampled once and cached."""
        if self.theta_on is None:
            g = torch.Generator(device=device); g.manual_seed(self.cfg.seed)
            self.gen = g
            b, sig, r = self.threshold, self.cfg.sigma_threshold, self.cfg.on_off_ratio
            self.theta_on = (b * r + sig * torch.randn(hw, generator=g, device=device)).clamp_min(self.cfg.min_threshold)
            self.theta_off = (b + sig * torch.randn(hw, generator=g, device=device)).clamp_min(self.cfg.min_threshold)
            self.hot_mask = torch.rand(hw, generator=g, device=device) < self.cfg.hot_pixel_frac
        return self.theta_on, self.theta_off

    # ── background events + hot pixels + refractory (hook 2) ─────
    def apply(self, pos_mask, neg_mask, intensity, current_time):
        """Take the signal masks, add sensor-noise events, enforce refractory.
        ``intensity`` is (num_envs, H, W) linear luminance; masks are the same shape."""
        device = pos_mask.device
        if self.theta_on is None:                       # ensure fixed pattern exists
            self.thresholds(pos_mask.shape[1:], device)
        if self.last_t is None:
            self.last_t = torch.full(pos_mask.shape, -1e30, device=device)

        dt = 0.0 if self.prev_t is None else max(0.0, current_time - self.prev_t)
        self.prev_t = current_time
        g = self.gen

        if dt > 0.0:
            def rand():
                return torch.rand(pos_mask.shape, generator=g, device=device)
            # leak: steady stream of ON events
            leak = rand() < (self.cfg.leak_rate_hz * dt)
            # shot: random ON/OFF, more in the dark (darkness in [0,1])
            dark = (1.0 - (intensity / self.cfg.intensity_scale).clamp(0.0, 1.0))
            shot = rand() < (self.cfg.shot_rate_hz * dt * dark)
            shot_on = shot & (rand() < 0.5)
            # hot pixels: high-rate random ON/OFF at the stuck pixels
            hot = self.hot_mask.unsqueeze(0) & (rand() < (self.cfg.hot_pixel_rate_hz * dt))
            hot_on = hot & (rand() < 0.5)
            pos_mask = pos_mask | leak | shot_on | hot_on
            neg_mask = neg_mask | (shot & ~shot_on) | (hot & ~hot_on)

        # one comparator per pixel: it can't fire ON and OFF at once (ON wins)
        neg_mask = neg_mask & ~pos_mask

        # refractory: suppress events within refractory_s of this pixel's last fire
        if self.cfg.refractory_s > 0.0:
            ready = (current_time - self.last_t) >= self.cfg.refractory_s
            pos_mask = pos_mask & ready
            neg_mask = neg_mask & ready
            fired = pos_mask | neg_mask
            self.last_t = torch.where(fired, torch.full_like(self.last_t, float(current_time)), self.last_t)

        return pos_mask, neg_mask
