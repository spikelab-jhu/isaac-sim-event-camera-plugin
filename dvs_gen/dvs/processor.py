"""
processor.py
============
``BatchedMultiCamProcessor`` — converts a batched RGB tensor (all envs of ONE
camera) into DVS events using the standard log-intensity-difference model, and
forwards them to a :class:`~dvs_gen.dvs.recorder.GeneralDVSRecorder`.

Event model: per pixel, an event fires when the change in ``log(intensity)``
since the last reference crosses ``±threshold``; the reference is then latched to
the new value (per pixel). Pure torch — no Omniverse dependency.
"""
import torch

from .recorder import GeneralDVSRecorder


class BatchedMultiCamProcessor:
    """Processes batched RGB tensors for ONE specific camera across ALL envs."""
    def __init__(self, recorder: GeneralDVSRecorder, camera_name: str, threshold: float = 0.15,
                 noise=None):
        self.recorder = recorder
        self.camera_name = camera_name
        self.threshold = threshold
        # optional DVSNoiseModel (dvs_gen.dvs.noise). None = the ideal clean model.
        self.noise = noise
        self.ref_log_intensity = None
        self.needs_reset_mask = None # Tracks which envs need a new reference frame
        # opt-in: keep this call's (pos_mask, neg_mask) so a caller can render the
        # per-frame event image aligned to the warped RGB frame. Off = zero overhead.
        self.stash_events = False
        self.last_masks = None

    def reset_envs(self, env_ids: torch.Tensor):
        """Flags specific environments to grab a fresh reference frame."""
        if self.needs_reset_mask is not None and len(env_ids) > 0:
            self.needs_reset_mask[env_ids] = True
            if self.noise is not None:
                self.noise.reset_envs(env_ids)         # re-seed the bandwidth filter state

    def __call__(self, rgb_batch: torch.Tensor, current_time: float):
        # rgb_batch: (num_envs, H, W, C)
        num_envs = rgb_batch.shape[0]
        device = rgb_batch.device
        if self.stash_events:
            self.last_masks = None      # cleared per call; set below once masks exist

        if rgb_batch.shape[-1] == 4:
            rgb_batch = rgb_batch[..., :3]

        intensity = (0.2126 * rgb_batch[..., 0] + 0.7152 * rgb_batch[..., 1] + 0.0722 * rgb_batch[..., 2])
        log_intensity = torch.log(intensity + 1e-5)

        # Intensity-dependent photoreceptor bandwidth (noise model hook 0): lowpass
        # the log frame BEFORE the reference logic so the whole event model sees the
        # filtered signal. cutoff_hz == 0 (default) returns it untouched.
        if self.noise is not None:
            log_intensity = self.noise.bandwidth_filter(log_intensity, intensity, current_time)

        # Initialization
        if self.ref_log_intensity is None:
            self.ref_log_intensity = log_intensity.clone()
            self.needs_reset_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
            return

        # 1. Update references for environments that just reset
        if self.needs_reset_mask.any():
            self.ref_log_intensity[self.needs_reset_mask] = log_intensity[self.needs_reset_mask]
            self.needs_reset_mask.fill_(False)
            # We don't generate events for the reset frame itself

        # 2. Compute differences. With a noise model, use PER-PIXEL thresholds
        # (fixed-pattern mismatch); otherwise the single scalar threshold.
        diff = log_intensity - self.ref_log_intensity
        if self.noise is not None:
            th_on, th_off = self.noise.thresholds(log_intensity.shape[1:], device)
            pos_mask = diff >= th_on
            neg_mask = diff <= -th_off
        else:
            pos_mask = diff >= self.threshold
            neg_mask = diff <= -self.threshold

        # Update reference where the SIGNAL crossed threshold (before noise). Refractory
        # only suppresses the recorded event, not the reference — avoids drift.
        self.ref_log_intensity[pos_mask] = log_intensity[pos_mask]
        self.ref_log_intensity[neg_mask] = log_intensity[neg_mask]

        # Sensor noise: inject background/leak/shot/hot events + enforce refractory.
        if self.noise is not None:
            pos_mask, neg_mask = self.noise.apply(pos_mask, neg_mask, intensity, current_time)

        if self.stash_events:               # final recorded masks (incl. noise) for the event image
            self.last_masks = (pos_mask, neg_mask)

        # 3. Extract events
        if pos_mask.any() or neg_mask.any():
            envs_pos, ys_pos, xs_pos = torch.where(pos_mask)
            envs_neg, ys_neg, xs_neg = torch.where(neg_mask)

            all_envs = torch.cat([envs_pos, envs_neg])
            all_xs = torch.cat([xs_pos, xs_neg])
            all_ys = torch.cat([ys_pos, ys_neg])
            all_ps = torch.cat([
                torch.ones(xs_pos.shape[0], dtype=torch.int8, device=device),
                -torch.ones(xs_neg.shape[0], dtype=torch.int8, device=device)
            ])

            self.recorder.record(self.camera_name, all_envs, all_xs, all_ys, all_ps, current_time)
