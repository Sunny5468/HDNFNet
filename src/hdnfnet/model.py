import torch
import torch.nn as nn


class HemoDelayNeuroFusionNet(nn.Module):
    """
    HemoDelayNeuroFusionNet (HDNFNet)

    A multi-modal architecture for EEG-fNIRS fusion that explicitly models the hemodynamic delay constraints in cross-modal attention. The model consists of:
- EEG temporal-spatial stem with convolutional and TCN layers to extract robust EEG features.
- fNIRS encoder to capture temporal patterns in the hemodynamic signals.
- Delay-constrained cross-modal attention that aligns EEG tokens with fNIRS context while respecting physiologically plausible delays.
- Delay-aware temporal masking that modulates EEG features based on the aligned fNIRS context, allowing the model to focus on EEG time-steps that are more likely to be influenced by the hemodynamic response.
- Sample-wise physiological quality gating to down-weight low-quality samples during training.

    Inputs:
        eeg_x:   (B, 1, C_eeg, T_eeg)
        fnirs_x: (B, C_fnirs, T_fnirs)

    Outputs:
        logits: (B, num_classes)
        aux: dict with sample_weight / quality_reg / attention maps for analysis.
    """

    def __init__(
        self,
        eeg_channels: int,
        eeg_samples: int,
        fnirs_channels: int,
        fnirs_samples: int,
        num_classes: int = 2,
        delay_min_s: float = 3.0,
        delay_max_s: float = 8.0,
        trial_duration_s: float = 10.0,
        mask_max_impact: float = 0.4,
    ):
        super().__init__()

        self.eeg_channels = int(eeg_channels)
        self.eeg_samples = int(eeg_samples)
        self.fnirs_channels = int(fnirs_channels)
        self.fnirs_samples = int(fnirs_samples)
        self.num_classes = int(num_classes)

        self.delay_min_s = float(delay_min_s)
        self.delay_max_s = float(delay_max_s)
        self.trial_duration_s = float(trial_duration_s)
        self.mask_max_impact = float(mask_max_impact)

        if self.delay_min_s < 0.0:
            raise ValueError(f"delay_min_s must be non-negative, got {self.delay_min_s}")
        if self.delay_max_s < self.delay_min_s:
            raise ValueError(
                f"delay_max_s must be >= delay_min_s, got {self.delay_max_s} < {self.delay_min_s}"
            )
        if self.trial_duration_s <= 0.0:
            raise ValueError(f"trial_duration_s must be positive, got {self.trial_duration_s}")
        if self.mask_max_impact < 0.0:
            raise ValueError(f"mask_max_impact must be non-negative, got {self.mask_max_impact}")

        # EEG temporal-spatial stem
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(self.eeg_channels, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(0.25),
        )
        self.temporal_pool = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))

        self.tcn_block1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.25),
        )
        self.tcn_block2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.25),
        )

        # fNIRS encoder
        self.fnirs_encoder = nn.Sequential(
            nn.Conv1d(self.fnirs_channels, 32, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Conv1d(32, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.25),
        )

        # Delay-constrained cross-modal alignment
        self.eeg_q_proj = nn.Linear(64, 64)
        self.fnirs_k_proj = nn.Linear(32, 64)
        self.fnirs_v_proj = nn.Linear(32, 64)
        self.cross_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        # Delay-aware temporal mask head (aligned context -> scalar per EEG time-step)
        self.mask_proj = nn.Linear(64, 1)

        # EEG self-attention integration
        self.self_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        # Sample-wise physiological quality gate
        self.gate_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.classifier = nn.Linear(64, self.num_classes)

    @staticmethod
    def _quality_regularization(quality_score: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        q = torch.clamp(quality_score, eps, 1.0 - eps)
        entropy = -(q * torch.log(q) + (1.0 - q) * torch.log(1.0 - q))
        return entropy.mean()

    def _build_delay_mask(self, q_len: int, k_len: int, device, dtype) -> torch.Tensor:
        eeg_dt_s = float(self.trial_duration_s) / max(int(q_len), 1)
        fnirs_dt_s = float(self.trial_duration_s) / max(int(k_len), 1)

        t_q = torch.arange(q_len, device=device, dtype=dtype) * eeg_dt_s
        t_k = torch.arange(k_len, device=device, dtype=dtype) * fnirs_dt_s
        delay = t_k.unsqueeze(0) - t_q.unsqueeze(1)

        valid = (delay >= self.delay_min_s) & (delay <= self.delay_max_s)
        mask = torch.full((q_len, k_len), float("-inf"), device=device, dtype=dtype)
        mask = mask.masked_fill(valid, 0.0)

        # Fallback for rows without valid keys: keep full visibility to avoid NaN attention rows.
        row_has_valid = valid.any(dim=1)
        if not torch.all(row_has_valid):
            mask[~row_has_valid] = 0.0
        return mask

    def _extract_eeg_tokens(self, eeg_x: torch.Tensor):
        x = self.spatial_conv(eeg_x)      # (B,16,1,T)
        x = self.temporal_conv(x)         # (B,32,1,T)
        x = self.temporal_pool(x)         # (B,32,1,Tp)
        x = x.squeeze(2)                  # (B,32,Tp)
        x = self.tcn_block1(x)            # (B,64,Tp)
        eeg_seq = self.tcn_block2(x)      # (B,64,Tp)
        eeg_tokens = eeg_seq.transpose(1, 2).contiguous()  # (B,Tp,64)
        return eeg_seq, eeg_tokens

    def _extract_fnirs_tokens(self, fnirs_x: torch.Tensor):
        fnirs_tokens = self.fnirs_encoder(fnirs_x).transpose(1, 2).contiguous()  # (B,Tf,32)
        return fnirs_tokens

    def forward_with_aux(self, eeg_x: torch.Tensor, fnirs_x: torch.Tensor):
        if eeg_x.dim() != 4:
            raise ValueError(f"Expected eeg_x as (B,1,C,T), got shape={tuple(eeg_x.shape)}")
        if eeg_x.shape[1] != 1:
            raise ValueError(f"Expected eeg_x channel dim=1 at axis=1, got shape={tuple(eeg_x.shape)}")
        if eeg_x.shape[2] != self.eeg_channels:
            raise ValueError(
                f"EEG channel mismatch: expected {self.eeg_channels}, got {eeg_x.shape[2]}"
            )
        if fnirs_x.dim() != 3:
            raise ValueError(f"Expected fnirs_x as (B,C,T), got shape={tuple(fnirs_x.shape)}")
        if fnirs_x.shape[1] != self.fnirs_channels:
            raise ValueError(
                f"fNIRS channel mismatch: expected {self.fnirs_channels}, got {fnirs_x.shape[1]}"
            )

        eeg_seq, eeg_tokens = self._extract_eeg_tokens(eeg_x)
        fnirs_tokens = self._extract_fnirs_tokens(fnirs_x)

        q = self.eeg_q_proj(eeg_tokens)
        k = self.fnirs_k_proj(fnirs_tokens)
        v = self.fnirs_v_proj(fnirs_tokens)
        delay_mask = self._build_delay_mask(q.size(1), k.size(1), q.device, q.dtype)

        aligned_context, cross_attn_weights = self.cross_attn(q, k, v, attn_mask=delay_mask)

        # Delay-aware mask from aligned context (not same-time fNIRS)
        mask_t = torch.sigmoid(self.mask_proj(aligned_context)).transpose(1, 2).contiguous()  # (B,1,Te)
        mask_centered = 2.0 * mask_t - 1.0
        eeg_seq = eeg_seq * (1.0 + self.mask_max_impact * mask_centered)
        eeg_tokens = eeg_seq.transpose(1, 2).contiguous()

        # Alignment fusion residual
        eeg_tokens = eeg_tokens + aligned_context

        refined_tokens, self_attn_weights = self.self_attn(eeg_tokens, eeg_tokens, eeg_tokens)
        feat = refined_tokens.mean(dim=1)  # (B,64)

        sample_weight = self.gate_head(feat)  # (B,1)
        quality_reg = self._quality_regularization(sample_weight)

        logits = self.classifier(feat)
        aux = {
            "sample_weight": sample_weight,
            "quality_reg": quality_reg,
            "features": feat,
            "delay_aware_mask": mask_t,
            "attn_weights": {
                "cross": cross_attn_weights,
                "self": self_attn_weights,
            },
        }
        return logits, aux

    def forward(self, eeg_x: torch.Tensor, fnirs_x: torch.Tensor):
        logits, _ = self.forward_with_aux(eeg_x, fnirs_x)
        return logits
