import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None

from strict_riemann_alignment import (
    extract_epochs_eeg,
    extract_epochs_nirs,
    load_cells,
    paired_by_label_order,
)
from models import AsymmetricfNIRSGuidedEEG
from model_16 import AsymmetricfNIRSGuidedEEG16


def parse_args():
    parser = argparse.ArgumentParser(
        description="Torch pipeline: Asymmetric fNIRS-guided EEG models"
    )
    parser.add_argument("--eeg", action="store_true", help="Run EEG-only branch derived from asym model")
    parser.add_argument("--eeg16", action="store_true", help="Run EEG-only branch derived from asym16 model")
    parser.add_argument("--atcnet", action="store_true", help="Run ATCNet-style EEG-only baseline")
    parser.add_argument("--atcnet_use_gating", dest="atcnet_use_gating", action="store_true", default=True, help="Enable quality gating in atcnet mode (default: on)")
    parser.add_argument("--no_atcnet_use_gating", dest="atcnet_use_gating", action="store_false", help="Disable quality gating in atcnet mode")
    parser.add_argument("--atcnet_use_alignment", dest="atcnet_use_alignment", action="store_true", default=True, help="Enable physiological alignment in atcnet mode (default: on)")
    parser.add_argument("--no_atcnet_use_alignment", dest="atcnet_use_alignment", action="store_false", help="Disable physiological alignment in atcnet mode")
    parser.add_argument("--atcnet_use_mask", dest="atcnet_use_mask", action="store_true", default=True, help="Enable temporal mask guidance in atcnet mode (default: on)")
    parser.add_argument("--no_atcnet_use_mask", dest="atcnet_use_mask", action="store_false", help="Disable temporal mask guidance in atcnet mode")
    parser.add_argument("--asym", action="store_true", help="Run Asymmetric fNIRS-guided EEG model from models.py")
    parser.add_argument("--asym16", action="store_true", help="Run 16x16 spatial Asymmetric fNIRS-guided EEG model from model_16.py")
    parser.add_argument("--asym16_use_gating", dest="asym16_use_gating", action="store_true", default=True, help="Enable gating branch in asym16 mode (default: on)")
    parser.add_argument("--no_asym16_use_gating", dest="asym16_use_gating", action="store_false", help="Disable gating branch in asym16 mode")
    parser.add_argument("--asym16_use_alignment", dest="asym16_use_alignment", action="store_true", default=True, help="Enable temporal alignment in asym16 mode (default: on)")
    parser.add_argument("--no_asym16_use_alignment", dest="asym16_use_alignment", action="store_false", help="Disable temporal alignment in asym16 mode")
    parser.add_argument("--asym16_use_dynamic_mask", dest="asym16_use_dynamic_mask", action="store_true", default=True, help="Enable dynamic spatial mask in asym16 mode (default: on)")
    parser.add_argument("--no_asym16_use_dynamic_mask", dest="asym16_use_dynamic_mask", action="store_false", help="Disable dynamic spatial mask in asym16 mode")
    parser.add_argument("--asym_quality_weight", type=float, default=0.02, help="Weight of quality regularization for asym mode")
    parser.add_argument("--asym_gate_mean_weight", type=float, default=0.01, help="Weight of gate-mean regularization for asym mode")
    parser.add_argument("--asym_gate_mean_target", type=float, default=0.6, help="Target mean gate value in asym mode")
    parser.add_argument("--asym_reg_warmup_epochs", type=int, default=10, help="Warm-up epochs with zero gate/quality regularization")
    parser.add_argument("--asym_power_frame_rate_hz", type=float, default=5.0, help="Target frame rate for power-envelope extraction in asym16")
    parser.add_argument("--asym_detach_sample_weight", dest="asym_detach_sample_weight", action="store_true", default=True, help="Detach gate weights before weighted CE in asym mode (default: on)")
    parser.add_argument("--no_asym_detach_sample_weight", dest="asym_detach_sample_weight", action="store_false", help="Disable gate-weight detaching in asym mode")
    parser.add_argument("--asym_delay_min_s", type=float, default=3.0, help="Minimum physiological EEG->fNIRS delay (seconds) for asym alignment")
    parser.add_argument("--asym_delay_max_s", type=float, default=8.0, help="Maximum physiological EEG->fNIRS delay (seconds) for asym alignment")
    parser.add_argument("--asym_trial_duration_s", type=float, default=10.0, help="Trial duration used to map token index to physical time in asym alignment")
    parser.add_argument("--sliding_window", dest="sliding_window", action="store_true", default=True, help="Enable sliding-window augmentation on each trial (default: on)")
    parser.add_argument("--no_sliding_window", dest="sliding_window", action="store_false", help="Disable sliding-window augmentation")
    parser.add_argument("--window_size_s", type=float, default=8.0, help="Sliding window size in seconds")
    parser.add_argument("--window_step_s", type=float, default=1.0, help="Sliding window step in seconds")
    parser.add_argument("--data_trial_duration_s", type=float, default=10.0, help="Physical duration (seconds) of loaded EEG/fNIRS trial segments")
    parser.add_argument("--loso", action="store_true", help="Use LOSO cross-subject evaluation")
    parser.add_argument("--lsso", action="store_true", help="Use LSSO cross-session evaluation")
    parser.add_argument("--hold_out", action="store_true", help="Use subject-specific hold-out across sessions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping pipeline")
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--early_stopping_max_epochs", type=int, default=200, help="Maximum epochs when early stopping is enabled")
    parser.add_argument("--hold_out_val_ratio", type=float, default=0.2, help="Validation ratio inside hold-out stage-1 training")
    parser.add_argument("--hold_out_stage1_patience", type=int, default=50, help="Patience for hold-out stage-1 early stopping")
    parser.add_argument("--hold_out_stage1_max_epochs", type=int, default=400, help="Maximum epochs for hold-out stage-1")
    parser.add_argument("--hold_out_stage2_max_epochs", type=int, default=200, help="Maximum epochs for hold-out stage-2")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--progress_bar", dest="progress_bar", action="store_true", default=True, help="Enable progress bar (default: on)")
    parser.add_argument("--no_progress_bar", dest="progress_bar", action="store_false", help="Disable progress bar")
    parser.add_argument(
        "--write_final_results",
        action="store_true",
        default=False,
        help="Write final aggregated all-results json at the end (default: off)",
    )
    return parser.parse_args()


def selected_modes(args) -> List[str]:
    modes = []
    if args.eeg:
        modes.append("eeg")
    if args.eeg16:
        modes.append("eeg16")
    if args.atcnet:
        modes.append("atcnet")
    if args.asym:
        modes.append("asym")
    if args.asym16:
        modes.append("asym16")
    if not modes:
        modes = ["asym", "asym16"]
    return modes


def sliding_window_augment_paired(
    eeg_epochs: np.ndarray,
    nirs_epochs: np.ndarray,
    labels: np.ndarray,
    window_size_s: float,
    window_step_s: float,
    trial_duration_s: float,
):
    if eeg_epochs.shape[0] == 0:
        return eeg_epochs, nirs_epochs, labels
    if window_size_s <= 0 or window_step_s <= 0 or trial_duration_s <= 0:
        raise ValueError("window_size_s, window_step_s and trial_duration_s must be > 0")

    eeg_t = int(eeg_epochs.shape[-1])
    nirs_t = int(nirs_epochs.shape[-1])
    if eeg_t <= 1 or nirs_t <= 1:
        return eeg_epochs, nirs_epochs, labels

    n_windows = int(np.floor((trial_duration_s - window_size_s) / window_step_s)) + 1
    if n_windows <= 1:
        return eeg_epochs, nirs_epochs, labels

    eeg_fs = eeg_t / trial_duration_s
    nirs_fs = nirs_t / trial_duration_s
    eeg_win = int(round(window_size_s * eeg_fs))
    nirs_win = int(round(window_size_s * nirs_fs))
    eeg_win = max(1, min(eeg_win, eeg_t))
    nirs_win = max(1, min(nirs_win, nirs_t))

    eeg_out, nirs_out, y_out = [], [], []
    for i in range(labels.shape[0]):
        eeg_trial = eeg_epochs[i]
        nirs_trial = nirs_epochs[i]
        y_i = labels[i]

        for w in range(n_windows):
            start_s = w * window_step_s
            eeg_start = int(round(start_s * eeg_fs))
            nirs_start = int(round(start_s * nirs_fs))

            eeg_start = min(max(0, eeg_start), max(0, eeg_t - eeg_win))
            nirs_start = min(max(0, nirs_start), max(0, nirs_t - nirs_win))
            eeg_end = eeg_start + eeg_win
            nirs_end = nirs_start + nirs_win

            eeg_out.append(eeg_trial[:, eeg_start:eeg_end])
            nirs_out.append(nirs_trial[:, nirs_start:nirs_end])
            y_out.append(y_i)

    return np.asarray(eeg_out), np.asarray(nirs_out), np.asarray(y_out)


def load_subject_trials(
    root: Path,
    sid: int,
    sliding_window: bool = True,
    window_size_s: float = 8.0,
    window_step_s: float = 1.0,
    data_trial_duration_s: float = 10.0,
):
    subject_name = f"subject {sid:02d}"
    eeg_root = root / "EEG-fNIRs异构数据集" / "EEG_01-29" / subject_name / "with occular artifact"
    nirs_root = root / "EEG-fNIRs异构数据集" / "NIRS_01-29" / subject_name

    eeg_cnt = eeg_root / "cnt.mat"
    eeg_mrk = eeg_root / "mrk.mat"
    nirs_cnt = nirs_root / "cnt.mat"
    nirs_mrk = nirs_root / "mrk.mat"
    for p in [eeg_cnt, eeg_mrk, nirs_cnt, nirs_mrk]:
        if not p.exists():
            return None

    cnt_eeg = load_cells(str(eeg_cnt), "cnt")
    mrk_eeg = load_cells(str(eeg_mrk), "mrk")
    cnt_nirs = load_cells(str(nirs_cnt), "cnt")
    mrk_nirs = load_cells(str(nirs_mrk), "mrk")

    sessions_mi = [0, 2, 4]
    eeg_map = {16: 0, 32: 1}
    nirs_map = {1: 0, 2: 1}

    eeg_all, nirs_all, y_all, sess_all = [], [], [], []
    for sid_session, s in enumerate(sessions_mi):
        eeg_epochs, eeg_labels = extract_epochs_eeg(
            cnt_eeg,
            mrk_eeg,
            sessions=[s],
            label_map=eeg_map,
            window_s=(0.0, 10.0),
        )
        nirs_epochs, nirs_labels = extract_epochs_nirs(
            cnt_nirs,
            mrk_nirs,
            sessions=[s],
            label_map=nirs_map,
            window_s=(2.0, 12.0),
            baseline_s=(-2.0, 0.0),
        )
        eeg_epochs, nirs_epochs, labels = paired_by_label_order(eeg_epochs, eeg_labels, nirs_epochs, nirs_labels)
        if len(labels) == 0:
            continue
        if sliding_window:
            eeg_epochs, nirs_epochs, labels = sliding_window_augment_paired(
                eeg_epochs,
                nirs_epochs,
                labels,
                window_size_s=window_size_s,
                window_step_s=window_step_s,
                trial_duration_s=data_trial_duration_s,
            )
        eeg_all.append(eeg_epochs)
        nirs_all.append(nirs_epochs)
        y_all.append(labels)
        sess_all.append(np.full(len(labels), sid_session, dtype=np.int32))

    if not y_all or int(np.sum([len(x) for x in y_all])) < 20:
        return None

    eeg_epochs = np.concatenate(eeg_all, axis=0)
    nirs_epochs = np.concatenate(nirs_all, axis=0)
    labels = np.concatenate(y_all, axis=0)
    session_ids = np.concatenate(sess_all, axis=0)
    return eeg_epochs, nirs_epochs, labels, session_ids


def build_dataset(
    root: Path,
    progress_bar: bool,
    sliding_window: bool = True,
    window_size_s: float = 8.0,
    window_step_s: float = 1.0,
    data_trial_duration_s: float = 10.0,
    return_session_ids: bool = False,
):
    eeg_all, nirs_all, y_all, subj_all, sess_all = [], [], [], [], []
    sid_iter = range(1, 30)
    if progress_bar and tqdm is not None:
        sid_iter = tqdm(sid_iter, desc="Loading subjects", leave=True, dynamic_ncols=True)

    for sid in sid_iter:
        out = load_subject_trials(
            root,
            sid,
            sliding_window=sliding_window,
            window_size_s=window_size_s,
            window_step_s=window_step_s,
            data_trial_duration_s=data_trial_duration_s,
        )
        if out is None:
            if progress_bar and tqdm is not None:
                tqdm.write(f"[skip] subject {sid:02d}: missing file or too few trials")
            else:
                print(f"[skip] subject {sid:02d}: missing file or too few trials")
            continue
        eeg_epochs, nirs_epochs, labels, session_ids = out
        n = len(labels)
        eeg_all.append(eeg_epochs)
        nirs_all.append(nirs_epochs)
        y_all.append(labels)
        subj_all.append(np.full(n, sid, dtype=np.int32))
        sess_all.append(session_ids)
        if progress_bar and tqdm is not None:
            tqdm.write(f"[ok] subject {sid:02d}: paired_trials={n}")
        else:
            print(f"[ok] subject {sid:02d}: paired_trials={n}")

    if not y_all:
        raise RuntimeError("No valid subject could be loaded.")

    eeg = np.concatenate(eeg_all, axis=0)
    nirs = np.concatenate(nirs_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    subjects = np.concatenate(subj_all, axis=0)
    sessions = np.concatenate(sess_all, axis=0)
    if return_session_ids:
        return eeg, nirs, y, subjects, sessions
    return eeg, nirs, y, subjects


class AsymPipelineModel(nn.Module):
    def __init__(self, eeg_ch, eeg_t, nirs_ch, nirs_t, asym_cfg=None):
        super().__init__()
        cfg = asym_cfg or {}
        self.net = AsymmetricfNIRSGuidedEEG(
            eeg_channels=eeg_ch,
            eeg_samples=eeg_t,
            fnirs_channels=nirs_ch,
            fnirs_samples=nirs_t,
            num_classes=2,
            use_gating=True,
            use_alignment=True,
            use_spatial_mask=True,
            delay_min_s=float(cfg.get("asym_delay_min_s", 3.0)),
            delay_max_s=float(cfg.get("asym_delay_max_s", 8.0)),
            trial_duration_s=float(cfg.get("asym_trial_duration_s", 10.0)),
        )

    def forward(self, eeg, nirs):
        logits, _ = self.net(eeg, nirs)
        return logits

    def forward_with_aux(self, eeg, nirs):
        logits, aux = self.net(eeg, nirs)
        if aux is None:
            aux = {}
        return logits, aux


class AsymEEGOnlyPipelineModel(nn.Module):
    def __init__(self, eeg_ch, eeg_t, nirs_ch, nirs_t, asym_cfg=None):
        super().__init__()
        cfg = asym_cfg or {}
        self.fnirs_ch = int(nirs_ch)
        self.fnirs_t = int(nirs_t)
        self.net = AsymmetricfNIRSGuidedEEG(
            eeg_channels=eeg_ch,
            eeg_samples=eeg_t,
            fnirs_channels=nirs_ch,
            fnirs_samples=nirs_t,
            num_classes=2,
            use_gating=False,
            use_alignment=False,
            use_spatial_mask=False,
            delay_min_s=float(cfg.get("asym_delay_min_s", 3.0)),
            delay_max_s=float(cfg.get("asym_delay_max_s", 8.0)),
            trial_duration_s=float(cfg.get("asym_trial_duration_s", 10.0)),
        )

    def _zero_fnirs(self, eeg):
        b = eeg.shape[0]
        return torch.zeros((b, self.fnirs_ch, self.fnirs_t), device=eeg.device, dtype=eeg.dtype)

    def forward(self, eeg, nirs):
        logits, _ = self.net(eeg, self._zero_fnirs(eeg))
        return logits

    def forward_with_aux(self, eeg, nirs):
        logits, aux = self.net(eeg, self._zero_fnirs(eeg))
        if aux is None:
            aux = {}
        return logits, aux


class Asym16PipelineModel(nn.Module):
    def __init__(self, eeg_ch, eeg_t, nirs_ch, nirs_t, asym_cfg=None):
        super().__init__()
        cfg = asym_cfg or {}
        self.net = AsymmetricfNIRSGuidedEEG16(
            eeg_channels=eeg_ch,
            eeg_samples=eeg_t,
            fnirs_channels=nirs_ch,
            fnirs_samples=nirs_t,
            num_classes=2,
            grid_size=8,
            use_gating=bool(cfg.get("asym16_use_gating", True)),
            use_alignment=bool(cfg.get("asym16_use_alignment", True)),
            use_dynamic_mask=bool(cfg.get("asym16_use_dynamic_mask", True)),
            delay_min_s=float(cfg.get("asym_delay_min_s", 3.0)),
            delay_max_s=float(cfg.get("asym_delay_max_s", 8.0)),
            trial_duration_s=float(cfg.get("asym_trial_duration_s", 10.0)),
            power_frame_rate_hz=float(cfg.get("asym_power_frame_rate_hz", 5.0)),
        )

    def forward(self, eeg, nirs):
        logits, _ = self.net(eeg, nirs)
        return logits

    def forward_with_aux(self, eeg, nirs):
        logits, aux = self.net(eeg, nirs)
        if aux is None:
            aux = {}
        return logits, aux


class Asym16EEGOnlyPipelineModel(nn.Module):
    def __init__(self, eeg_ch, eeg_t, nirs_ch, nirs_t, asym_cfg=None):
        super().__init__()
        cfg = asym_cfg or {}
        self.fnirs_ch = int(nirs_ch)
        self.fnirs_t = int(nirs_t)
        self.net = AsymmetricfNIRSGuidedEEG16(
            eeg_channels=eeg_ch,
            eeg_samples=eeg_t,
            fnirs_channels=nirs_ch,
            fnirs_samples=nirs_t,
            num_classes=2,
            grid_size=8,
            use_gating=False,
            use_alignment=False,
            use_dynamic_mask=False,
            delay_min_s=float(cfg.get("asym_delay_min_s", 3.0)),
            delay_max_s=float(cfg.get("asym_delay_max_s", 8.0)),
            trial_duration_s=float(cfg.get("asym_trial_duration_s", 10.0)),
            power_frame_rate_hz=float(cfg.get("asym_power_frame_rate_hz", 5.0)),
        )

    def _zero_fnirs(self, eeg):
        b = eeg.shape[0]
        return torch.zeros((b, self.fnirs_ch, self.fnirs_t), device=eeg.device, dtype=eeg.dtype)

    def forward(self, eeg, nirs):
        logits, _ = self.net(eeg, self._zero_fnirs(eeg))
        return logits

    def forward_with_aux(self, eeg, nirs):
        logits, aux = self.net(eeg, self._zero_fnirs(eeg))
        if aux is None:
            aux = {}
        return logits, aux


class ATCNetEEGOnlyPipelineModel(nn.Module):
    """ATCNet-style EEG-only model with temporal attention and TCN blocks."""

    def __init__(self, eeg_ch, eeg_t, nirs_ch, nirs_t, asym_cfg=None):
        super().__init__()
        cfg = asym_cfg or {}
        self.fnirs_ch = int(nirs_ch)
        self.fnirs_t = int(nirs_t)
        self.eeg_t = int(eeg_t)
        self.use_gating = bool(cfg.get("atcnet_use_gating", True))
        self.use_alignment = bool(cfg.get("atcnet_use_alignment", True))
        self.use_mask = bool(cfg.get("atcnet_use_mask", True))
        self.mask_max_impact = 0.4
        self.delay_min_s = float(cfg.get("asym_delay_min_s", 3.0))
        self.delay_max_s = float(cfg.get("asym_delay_max_s", 8.0))
        self.trial_duration_s = float(cfg.get("asym_trial_duration_s", 10.0))

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(eeg_ch, 1), bias=False),
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
        self.fnirs_encoder = nn.Sequential(
            nn.Conv1d(self.fnirs_ch, 32, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Conv1d(32, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.25),
        )
        self.eeg_q_proj = nn.Linear(64, 64)
        self.fnirs_k_proj = nn.Linear(32, 64)
        self.fnirs_v_proj = nn.Linear(32, 64)
        self.cross_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.mask_proj = nn.Linear(64, 1)
        self.gate_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.classifier = nn.Linear(64, 2)

    def _zero_fnirs(self, eeg):
        b = eeg.shape[0]
        return torch.zeros((b, self.fnirs_ch, self.fnirs_t), device=eeg.device, dtype=eeg.dtype)

    @staticmethod
    def _quality_regularization(quality_score):
        eps = 1e-6
        q = torch.clamp(quality_score, eps, 1.0 - eps)
        entropy = -(q * torch.log(q) + (1.0 - q) * torch.log(1.0 - q))
        return entropy.mean()

    def _build_delay_mask(self, q_len, k_len, device, dtype):
        eeg_dt_s = float(self.trial_duration_s) / max(int(q_len), 1)
        fnirs_dt_s = float(self.trial_duration_s) / max(int(k_len), 1)
        t_q = torch.arange(q_len, device=device, dtype=dtype) * eeg_dt_s
        t_k = torch.arange(k_len, device=device, dtype=dtype) * fnirs_dt_s
        delay = t_k.unsqueeze(0) - t_q.unsqueeze(1)
        valid = (delay >= self.delay_min_s) & (delay <= self.delay_max_s)
        mask = torch.full((q_len, k_len), float("-inf"), device=device, dtype=dtype)
        mask = mask.masked_fill(valid, 0.0)
        row_has_valid = valid.any(dim=1)
        if not torch.all(row_has_valid):
            mask[~row_has_valid] = 0.0
        return mask

    def _forward_features(self, eeg, nirs):
        # Input eeg: (B,1,C,T)
        x = self.spatial_conv(eeg)          # (B,16,1,T)
        x = self.temporal_conv(x)           # (B,32,1,T)
        x = self.temporal_pool(x)           # (B,32,1,Tp)
        x = x.squeeze(2)                    # (B,32,Tp)
        x = self.tcn_block1(x)              # (B,64,Tp)
        eeg_seq = self.tcn_block2(x)        # (B,64,Tp)
        eeg_tokens = eeg_seq.transpose(1, 2)  # (B,Tp,64)

        fnirs_tokens = self.fnirs_encoder(nirs).transpose(1, 2).contiguous()  # (B,Tf,32)

        attn_weights = None
        aligned_context = None
        if self.use_alignment or self.use_mask:
            q = self.eeg_q_proj(eeg_tokens)
            k = self.fnirs_k_proj(fnirs_tokens)
            v = self.fnirs_v_proj(fnirs_tokens)
            delay_mask = self._build_delay_mask(q.size(1), k.size(1), q.device, q.dtype)
            aligned_context, attn_weights = self.cross_attn(q, k, v, attn_mask=delay_mask)

        if self.use_mask and aligned_context is not None:
            # Delay-aware mask: derive per-EEG-step mask from physiologically aligned context.
            mask_t = torch.sigmoid(self.mask_proj(aligned_context)).transpose(1, 2).contiguous()  # (B,1,Te)
            # Center mask to [-1,1] and cap impact to +/-40% to avoid over-suppress/amplify.
            mask_centered = 2.0 * mask_t - 1.0
            eeg_seq = eeg_seq * (1.0 + float(self.mask_max_impact) * mask_centered)
            eeg_tokens = eeg_seq.transpose(1, 2).contiguous()

        if self.use_alignment and aligned_context is not None:
            eeg_tokens = eeg_tokens + aligned_context

        attn_out, self_attn_weights = self.attn(eeg_tokens, eeg_tokens, eeg_tokens)
        feat = attn_out.mean(dim=1)         # (B,64)

        sample_weight = None
        quality_reg = torch.zeros((), device=feat.device)
        if self.use_gating:
            sample_weight = self.gate_head(feat)  # (B,1)
            quality_reg = self._quality_regularization(sample_weight)

        aux = {
            "sample_weight": sample_weight,
            "quality_reg": quality_reg,
            "features": feat,
            "attn_weights": {"cross": attn_weights, "self": self_attn_weights},
        }
        return feat, aux

    def forward(self, eeg, nirs):
        feat, _ = self._forward_features(eeg, nirs)
        logits = self.classifier(feat)
        return logits

    def forward_with_aux(self, eeg, nirs):
        feat, aux = self._forward_features(eeg, nirs)
        logits = self.classifier(feat)
        return logits, aux


def make_model(mode: str, eeg_ch: int, eeg_t: int, nirs_ch: int, nirs_t: int, asym_cfg=None):
    if mode == "eeg":
        return AsymEEGOnlyPipelineModel(eeg_ch, eeg_t, nirs_ch, nirs_t, asym_cfg=asym_cfg)
    if mode == "eeg16":
        return Asym16EEGOnlyPipelineModel(eeg_ch, eeg_t, nirs_ch, nirs_t, asym_cfg=asym_cfg)
    if mode == "atcnet":
        return ATCNetEEGOnlyPipelineModel(eeg_ch, eeg_t, nirs_ch, nirs_t, asym_cfg=asym_cfg)
    if mode == "asym":
        return AsymPipelineModel(eeg_ch, eeg_t, nirs_ch, nirs_t, asym_cfg=asym_cfg)
    if mode == "asym16":
        return Asym16PipelineModel(eeg_ch, eeg_t, nirs_ch, nirs_t, asym_cfg=asym_cfg)
    raise ValueError(f"Unsupported mode '{mode}'. Only 'eeg', 'eeg16', 'atcnet', 'asym' and 'asym16' are allowed.")


def standardize_nirs_by_train_channel(nirs_train: np.ndarray, nirs_test: np.ndarray):
    # Fit channel-wise stats on training set only to avoid leakage.
    ch_mean = nirs_train.mean(axis=(0, 2), keepdims=True)
    ch_std = nirs_train.std(axis=(0, 2), keepdims=True) + 1e-8
    nirs_train = (nirs_train - ch_mean) / ch_std
    nirs_test = (nirs_test - ch_mean) / ch_std
    return nirs_train, nirs_test


def standardize_eeg_by_train_channel(eeg_train: np.ndarray, eeg_test: np.ndarray):
    # Fit channel-wise stats on training set only to avoid leakage.
    ch_mean = eeg_train.mean(axis=(0, 2), keepdims=True)
    ch_std = eeg_train.std(axis=(0, 2), keepdims=True) + 1e-8
    eeg_train = (eeg_train - ch_mean) / ch_std
    eeg_test = (eeg_test - ch_mean) / ch_std
    return eeg_train, eeg_test


def _inv_sqrt_spd(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mat = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, eps, None)
    inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return inv_sqrt.astype(np.float32)


def euclidean_align_eeg_by_subject(eeg_epochs: np.ndarray, subjects: np.ndarray, eps: float = 1e-6):
    """Subject-wise Euclidean Alignment for EEG epochs (N,C,T)."""
    out = np.empty_like(eeg_epochs, dtype=np.float32)
    subjects = np.asarray(subjects)
    for sid in np.unique(subjects):
        idx = np.where(subjects == sid)[0]
        x_sid = eeg_epochs[idx].astype(np.float32)
        c = x_sid.shape[1]
        cov_mean = np.zeros((c, c), dtype=np.float32)
        for i in range(x_sid.shape[0]):
            xi = x_sid[i]
            cov = (xi @ xi.T) / max(xi.shape[1] - 1, 1)
            cov = cov + eps * np.eye(c, dtype=np.float32)
            cov_mean += cov
        cov_mean /= max(x_sid.shape[0], 1)
        ref_inv_sqrt = _inv_sqrt_spd(cov_mean, eps=eps)
        out[idx] = np.einsum("ab,nbt->nat", ref_inv_sqrt, x_sid)
    return out


def prepare_fold_data(mode, eeg_train, eeg_test, nirs_train, nirs_test, subj_train, subj_test, seed):
    if mode not in {"eeg", "eeg16", "atcnet", "asym", "asym16"}:
        raise ValueError(f"Unsupported mode '{mode}'. Only 'eeg', 'eeg16', 'atcnet', 'asym' and 'asym16' are allowed.")

    eeg_train_t = eeg_train.astype(np.float32)
    eeg_test_t = eeg_test.astype(np.float32)
    nirs_train_t = nirs_train.astype(np.float32)
    nirs_test_t = nirs_test.astype(np.float32)

    eeg_train_t = euclidean_align_eeg_by_subject(eeg_train_t, subj_train)
    eeg_test_t = euclidean_align_eeg_by_subject(eeg_test_t, subj_test)

    eeg_train_t, eeg_test_t = standardize_eeg_by_train_channel(eeg_train_t, eeg_test_t)
    nirs_train_t, nirs_test_t = standardize_nirs_by_train_channel(nirs_train_t, nirs_test_t)

    # EEGNet expects (B,1,C,T)
    eeg_train_t = np.expand_dims(eeg_train_t, axis=1)
    eeg_test_t = np.expand_dims(eeg_test_t, axis=1)

    return eeg_train_t, eeg_test_t, nirs_train_t, nirs_test_t


def compute_metrics(y_true, y_pred, y_prob):
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    try:
        result["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        result["auc"] = float("nan")
    return result


def save_binary_confusion_matrix(y_true, y_pred, save_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def save_training_curve(histories, save_path: Path, title: str):
    if not histories:
        return
    max_len = max(len(h.get("loss", [])) for h in histories)
    if max_len == 0:
        return

    def stack_with_nan(key: str):
        mat = np.full((len(histories), max_len), np.nan, dtype=np.float64)
        for i, h in enumerate(histories):
            vals = np.asarray(h.get(key, []), dtype=np.float64)
            n = min(len(vals), max_len)
            if n > 0:
                mat[i, :n] = vals[:n]
        return mat

    loss_mat = stack_with_nan("loss")
    acc_mat = stack_with_nan("acc")
    val_loss_mat = stack_with_nan("val_loss")
    val_acc_mat = stack_with_nan("val_acc")

    loss_mean = np.nanmean(loss_mat, axis=0)
    loss_std = np.nanstd(loss_mat, axis=0, ddof=1) if loss_mat.shape[0] > 1 else np.zeros(loss_mat.shape[1])
    acc_mean = np.nanmean(acc_mat, axis=0)
    acc_std = np.nanstd(acc_mat, axis=0, ddof=1) if acc_mat.shape[0] > 1 else np.zeros(acc_mat.shape[1])
    val_loss_mean = np.nanmean(val_loss_mat, axis=0)
    val_loss_std = np.nanstd(val_loss_mat, axis=0, ddof=1) if val_loss_mat.shape[0] > 1 else np.zeros(val_loss_mat.shape[1])
    val_acc_mean = np.nanmean(val_acc_mat, axis=0)
    val_acc_std = np.nanstd(val_acc_mat, axis=0, ddof=1) if val_acc_mat.shape[0] > 1 else np.zeros(val_acc_mat.shape[1])

    epochs = np.arange(1, max_len + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, loss_mean, color="#d62728", label="train")
    axes[0].fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std, color="#d62728", alpha=0.2)
    axes[0].plot(epochs, val_loss_mean, color="#ff7f0e", label="val")
    axes[0].fill_between(epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, color="#ff7f0e", alpha=0.2)
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, acc_mean, color="#1f77b4", label="train")
    axes[1].fill_between(epochs, acc_mean - acc_std, acc_mean + acc_std, color="#1f77b4", alpha=0.2)
    axes[1].plot(epochs, val_acc_mean, color="#2ca02c", label="val")
    axes[1].fill_between(epochs, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std, color="#2ca02c", alpha=0.2)
    axes[1].set_title("Accuracy vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def save_single_fold_curves(history: Dict[str, List[float]], loss_path: Path, acc_path: Path, title_prefix: str):
    loss_vals = np.asarray(history.get("loss", []), dtype=np.float64)
    val_loss_vals = np.asarray(history.get("val_loss", []), dtype=np.float64)
    acc_vals = np.asarray(history.get("acc", []), dtype=np.float64)
    val_acc_vals = np.asarray(history.get("val_acc", []), dtype=np.float64)

    if len(loss_vals) > 0:
        x = np.arange(1, len(loss_vals) + 1)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, loss_vals, color="#d62728", label="train")
        if len(val_loss_vals) > 0:
            n = min(len(x), len(val_loss_vals))
            ax.plot(x[:n], val_loss_vals[:n], color="#ff7f0e", label="val")
        ax.set_title(f"{title_prefix} Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(loss_path, dpi=180)
        plt.close(fig)

    if len(acc_vals) > 0:
        x = np.arange(1, len(acc_vals) + 1)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, acc_vals, color="#1f77b4", label="train")
        if len(val_acc_vals) > 0:
            n = min(len(x), len(val_acc_vals))
            ax.plot(x[:n], val_acc_vals[:n], color="#2ca02c", label="val")
        ax.set_title(f"{title_prefix} Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        fig.tight_layout()
        fig.savefig(acc_path, dpi=180)
        plt.close(fig)


def fmt_minutes_seconds(sec: float) -> str:
    minutes = sec / 60.0
    if minutes >= 1.0:
        return f"{minutes:.2f}m"
    return f"{sec:.2f}s"


def fmt_ms_per_trial(sec: float, n_trials: int) -> float:
    if n_trials <= 0:
        return float("nan")
    return float(1000.0 * sec / n_trials)


def write_mode_config_yaml(mode_dir: Path, args, mode: str, protocol: str):
    cfg_lines = [
        f"dataset_name: EEG-fNIRs_01-29_{protocol.lower()}",
        f"model: {mode}",
        f"seed: {args.seed}",
        f"gpu_id: 0",
        f"max_epochs: {args.epochs}",
        f"max_epochs_loso: {args.epochs}",
        "preprocessing:",
        f"  batch_size: {args.batch_size}",
        "  z_scale: true",
        "  split_seed: 42",
        "progress_bar: true" if args.progress_bar else "progress_bar: false",
        "subject_ids: all",
    ]

    cfg_path = mode_dir / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cfg_lines) + "\n")


def write_mode_results_txt(mode_dir: Path, mode: str, protocol: str, subject_ids: List[int], fold_records: List[Dict], summary: Dict):
    lines = []
    lines.append(f"Results for model: {mode}")
    lines.append("#Params: N/A")
    lines.append(f"Dataset: EEG-fNIRs_01-29_{protocol.lower()}")
    lines.append(f"Subject IDs: {subject_ids}")
    lines.append("")
    lines.append("Results for each subject:")

    for rec in fold_records:
        sid = rec.get("subject_id", -1)
        m = rec.get("metrics", {})
        train_s = float(rec.get("train_time_seconds", 0.0))
        test_s = float(rec.get("test_time_seconds", 0.0))
        acc = float(m.get("accuracy", float("nan")))
        loss = float(m.get("loss", float("nan")))
        kappa = float(m.get("kappa", float("nan")))
        lines.append(
            "Subject {} => Train Time: {}, Test Time: {}, Test Acc: {:.4f}, Test Loss: {:.4f}, Test Kappa: {:.4f}".format(
                sid,
                fmt_minutes_seconds(train_s),
                fmt_minutes_seconds(test_s),
                acc,
                loss,
                kappa,
            )
        )

    lines.append("")
    lines.append("--- Summary Statistics ---")
    acc_mean = float(summary.get("accuracy", {}).get("mean", float("nan")))
    acc_std = float(summary.get("accuracy", {}).get("std", float("nan")))
    kappa_mean = float(summary.get("kappa", {}).get("mean", float("nan")))
    kappa_std = float(summary.get("kappa", {}).get("std", float("nan")))
    loss_mean = float(summary.get("loss", {}).get("mean", float("nan")))
    loss_std = float(summary.get("loss", {}).get("std", float("nan")))
    total_train_sec = float(np.nansum([float(r.get("train_time_seconds", 0.0)) for r in fold_records]))
    rt_vals = [fmt_ms_per_trial(float(r.get("test_time_seconds", 0.0)), int(r.get("n_test", 0))) for r in fold_records]
    rt_vals = [v for v in rt_vals if not math.isnan(v)]
    avg_resp_ms = float(np.nanmean(rt_vals)) if rt_vals else float("nan")

    lines.append(f"Average Test Accuracy: {acc_mean*100:.2f} ± {acc_std*100:.2f}")
    lines.append(f"Average Test Kappa:    {kappa_mean:.3f} ± {kappa_std:.3f}")
    lines.append(f"Average Test Loss:     {loss_mean:.3f} ± {loss_std:.3f}")
    lines.append(f"Total Training Time: {total_train_sec/60.0:.2f} min")
    lines.append(f"Average Response Time: {avg_resp_ms:.2f} ms")

    with open(mode_dir / "results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_single_subject_result_txt(mode_dir: Path, mode: str, protocol: str, rec: Dict):
    sid = int(rec.get("subject_id", -1))
    m = rec.get("metrics", {})
    train_s = float(rec.get("train_time_seconds", 0.0))
    test_s = float(rec.get("test_time_seconds", 0.0))
    acc = float(m.get("accuracy", float("nan")))
    loss = float(m.get("loss", float("nan")))
    kappa = float(m.get("kappa", float("nan")))

    lines = [
        f"Results for model: {mode}",
        f"Dataset: EEG-fNIRs_01-29_{protocol.lower()}",
        f"Subject: {sid}",
        "",
        "Result:",
        "Subject {} => Train Time: {}, Test Time: {}, Test Acc: {:.4f}, Test Loss: {:.4f}, Test Kappa: {:.4f}".format(
            sid,
            fmt_minutes_seconds(train_s),
            fmt_minutes_seconds(test_s),
            acc,
            loss,
            kappa,
        ),
    ]

    subject_dir = mode_dir / "subjects"
    subject_dir.mkdir(parents=True, exist_ok=True)
    with open(subject_dir / f"subject_{sid:02d}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def train_one_fold(
    mode,
    eeg_train,
    y_train,
    nirs_train,
    eeg_test,
    nirs_test,
    y_test,
    seed,
    epochs,
    batch_size,
    lr,
    device,
    asym_cfg=None,
    fold_label="",
    epoch_logger=None,
    asym_quality_weight: float = 0.02,
    asym_gate_mean_weight: float = 0.01,
    asym_gate_mean_target: float = 0.6,
    asym_detach_sample_weight: bool = False,
    early_stopping: bool = False,
    early_stopping_patience: int = 20,
    early_stopping_max_epochs: int = 200,
):
    torch.manual_seed(seed)
    model = make_model(
        mode,
        eeg_train.shape[2],
        eeg_train.shape[3],
        nirs_train.shape[1],
        nirs_train.shape[2],
        asym_cfg=asym_cfg,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    if len(np.unique(y_train)) >= 2 and len(y_train) >= 10:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_local_idx, val_local_idx = next(splitter.split(np.zeros(len(y_train)), y_train))
    else:
        train_local_idx = np.arange(len(y_train))
        val_local_idx = np.arange(len(y_train))

    ds = TensorDataset(
        torch.from_numpy(eeg_train[train_local_idx]),
        torch.from_numpy(nirs_train[train_local_idx]),
        torch.from_numpy(y_train[train_local_idx].astype(np.int64)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    val_ds = TensorDataset(
        torch.from_numpy(eeg_train[val_local_idx]),
        torch.from_numpy(nirs_train[val_local_idx]),
        torch.from_numpy(y_train[val_local_idx].astype(np.int64)),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "val_eeg_acc": []}

    train_start_time = time.perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    reg_warmup_epochs = int((asym_cfg or {}).get("asym_reg_warmup_epochs", 10))
    total_epochs = int(epochs)
    if early_stopping:
        total_epochs = min(int(early_stopping_max_epochs), max(1, int(early_stopping_max_epochs)))
    best_state = None
    best_val_acc = -1.0
    stale_epochs = 0
    model.train()
    for ep in range(1, total_epochs + 1):
        reg_scale = 0.0 if ep <= reg_warmup_epochs else 1.0
        running_loss = 0.0
        running_cls_loss = 0.0
        running_qreg_loss = 0.0
        running_meanreg_loss = 0.0
        running_correct = 0
        total = 0
        for eeg_b, nirs_b, y_b in loader:
            eeg_b = eeg_b.to(device)
            nirs_b = nirs_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            logits, aux = model.forward_with_aux(eeg_b, nirs_b)
            sample_weight = aux.get("sample_weight", None)
            if sample_weight is not None:
                w = torch.clamp(sample_weight.squeeze(1), min=1e-3, max=1.0)
                if asym_detach_sample_weight:
                    w = w.detach()
                ce_per_sample = F.cross_entropy(logits, y_b, reduction="none")
                l_cls = torch.sum(ce_per_sample * w) / torch.sum(w)
                gate_mean_reg = (w.mean() - float(asym_gate_mean_target)) ** 2
            else:
                l_cls = criterion(logits, y_b)
                gate_mean_reg = torch.zeros((), device=device)
            qreg = aux.get("quality_reg", torch.zeros((), device=device))
            loss = l_cls + reg_scale * (
                float(asym_quality_weight) * qreg + float(asym_gate_mean_weight) * gate_mean_reg
            )

            running_cls_loss += l_cls.item() * y_b.size(0)
            running_qreg_loss += qreg.item() * y_b.size(0)
            running_meanreg_loss += gate_mean_reg.item() * y_b.size(0)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y_b.size(0)
            pred = torch.argmax(logits, dim=1)
            running_correct += (pred == y_b).sum().item()
            total += y_b.size(0)

        history["loss"].append(running_loss / max(total, 1))
        history["acc"].append(running_correct / max(total, 1))

        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            for eeg_v, nirs_v, y_v in val_loader:
                eeg_v = eeg_v.to(device)
                nirs_v = nirs_v.to(device)
                y_v = y_v.to(device)

                val_logits, val_aux = model.forward_with_aux(eeg_v, nirs_v)
                val_l_cls = criterion(val_logits, y_v)
                val_qreg = val_aux.get("quality_reg", torch.zeros((), device=device))
                val_loss_batch = val_l_cls + reg_scale * float(asym_quality_weight) * val_qreg

                val_loss_sum += val_loss_batch.item() * y_v.size(0)
                val_pred = torch.argmax(val_logits, dim=1)
                val_correct += (val_pred == y_v).sum().item()
                val_total += y_v.size(0)

            val_loss = val_loss_sum / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_eeg_acc"].append(np.nan)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1

        if epoch_logger is not None:
            epoch_logger(
                "{} Epoch {}/{} reg_scale={:.1f} train_loss={:.4f} train_cls={:.4f} train_qreg={:.4f} train_meanreg={:.4f} train_acc={:.4f} val_loss={:.4f} val_acc={:.4f}".format(
                    fold_label,
                    ep,
                    total_epochs,
                    reg_scale,
                    history["loss"][-1],
                    running_cls_loss / max(total, 1),
                    running_qreg_loss / max(total, 1),
                    running_meanreg_loss / max(total, 1),
                    history["acc"][-1],
                    val_loss,
                    val_acc,
                )
            )
        if early_stopping and stale_epochs >= int(early_stopping_patience):
            if epoch_logger is not None:
                epoch_logger(
                    "{} Early stopping triggered at epoch {} (patience={})".format(
                        fold_label,
                        ep,
                        int(early_stopping_patience),
                    )
                )
            break
        model.train()

    if best_state is not None:
        model.load_state_dict(best_state)

    train_time_seconds = float(time.perf_counter() - train_start_time)
    model.eval()
    with torch.no_grad():
        infer_start_time = time.perf_counter()
        test_ds = TensorDataset(
            torch.from_numpy(eeg_test),
            torch.from_numpy(nirs_test),
            torch.from_numpy(y_test.astype(np.int64)),
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        test_loss_sum = 0.0
        test_total = 0
        prob_list = []
        pred_list = []
        for eeg_t, nirs_t, y_t in test_loader:
            eeg_t = eeg_t.to(device)
            nirs_t = nirs_t.to(device)
            y_t = y_t.to(device)

            logits = model(eeg_t, nirs_t)
            test_loss_sum += criterion(logits, y_t).item() * y_t.size(0)
            test_total += y_t.size(0)
            prob_list.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            pred_list.append(torch.argmax(logits, dim=1).cpu().numpy())

        test_loss = float(test_loss_sum / max(test_total, 1))
        prob = np.concatenate(prob_list, axis=0) if prob_list else np.empty((0,), dtype=np.float32)
        pred = np.concatenate(pred_list, axis=0) if pred_list else np.empty((0,), dtype=np.int64)
        test_time_seconds = float(time.perf_counter() - infer_start_time)

    return pred, prob, history, test_loss, train_time_seconds, test_time_seconds


def _split_train_val_for_holdout(y_train: np.ndarray, val_ratio: float, seed: int):
    val_ratio = float(val_ratio)
    val_ratio = min(max(val_ratio, 0.0), 0.5)
    if val_ratio <= 0.0 or len(y_train) < 10 or len(np.unique(y_train)) < 2:
        idx = np.arange(len(y_train))
        return idx, idx

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    try:
        train_idx, val_idx = next(splitter.split(np.zeros(len(y_train)), y_train))
    except ValueError:
        idx = np.arange(len(y_train))
        return idx, idx
    if len(train_idx) == 0 or len(val_idx) == 0:
        idx = np.arange(len(y_train))
        return idx, idx
    return train_idx, val_idx


def train_one_fold_holdout_two_stage(
    mode,
    eeg_train,
    y_train,
    nirs_train,
    eeg_test,
    nirs_test,
    y_test,
    seed,
    batch_size,
    lr,
    device,
    asym_cfg=None,
    fold_label="",
    epoch_logger=None,
    hold_out_val_ratio: float = 0.2,
    hold_out_stage1_patience: int = 50,
    hold_out_stage1_max_epochs: int = 400,
    hold_out_stage2_max_epochs: int = 200,
    asym_quality_weight: float = 0.02,
    asym_gate_mean_weight: float = 0.01,
    asym_gate_mean_target: float = 0.6,
    asym_detach_sample_weight: bool = False,
):
    torch.manual_seed(seed)
    model = make_model(
        mode,
        eeg_train.shape[2],
        eeg_train.shape[3],
        nirs_train.shape[1],
        nirs_train.shape[2],
        asym_cfg=asym_cfg,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    reg_warmup_epochs = int((asym_cfg or {}).get("asym_reg_warmup_epochs", 10))

    train_local_idx, val_local_idx = _split_train_val_for_holdout(y_train, hold_out_val_ratio, seed)
    loader_stage1 = DataLoader(
        TensorDataset(
            torch.from_numpy(eeg_train[train_local_idx]),
            torch.from_numpy(nirs_train[train_local_idx]),
            torch.from_numpy(y_train[train_local_idx].astype(np.int64)),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(eeg_train[val_local_idx]),
            torch.from_numpy(nirs_train[val_local_idx]),
            torch.from_numpy(y_train[val_local_idx].astype(np.int64)),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    loader_stage2 = DataLoader(
        TensorDataset(
            torch.from_numpy(eeg_train),
            torch.from_numpy(nirs_train),
            torch.from_numpy(y_train.astype(np.int64)),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "val_eeg_acc": []}
    train_start_time = time.perf_counter()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val_acc = -1.0
    stale_epochs = 0
    stage1_end_train_loss = float("inf")

    model.train()
    for ep in range(1, int(hold_out_stage1_max_epochs) + 1):
        reg_scale = 0.0 if ep <= reg_warmup_epochs else 1.0
        running_loss = 0.0
        running_cls_loss = 0.0
        running_qreg_loss = 0.0
        running_meanreg_loss = 0.0
        running_correct = 0
        total = 0

        for eeg_b, nirs_b, y_b in loader_stage1:
            eeg_b = eeg_b.to(device)
            nirs_b = nirs_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            logits, aux = model.forward_with_aux(eeg_b, nirs_b)
            sample_weight = aux.get("sample_weight", None)
            if sample_weight is not None:
                w = torch.clamp(sample_weight.squeeze(1), min=1e-3, max=1.0)
                if asym_detach_sample_weight:
                    w = w.detach()
                ce_per_sample = F.cross_entropy(logits, y_b, reduction="none")
                l_cls = torch.sum(ce_per_sample * w) / torch.sum(w)
                gate_mean_reg = (w.mean() - float(asym_gate_mean_target)) ** 2
            else:
                l_cls = criterion(logits, y_b)
                gate_mean_reg = torch.zeros((), device=device)
            qreg = aux.get("quality_reg", torch.zeros((), device=device))
            loss = l_cls + reg_scale * (
                float(asym_quality_weight) * qreg + float(asym_gate_mean_weight) * gate_mean_reg
            )

            running_cls_loss += l_cls.item() * y_b.size(0)
            running_qreg_loss += qreg.item() * y_b.size(0)
            running_meanreg_loss += gate_mean_reg.item() * y_b.size(0)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y_b.size(0)
            pred = torch.argmax(logits, dim=1)
            running_correct += (pred == y_b).sum().item()
            total += y_b.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = running_correct / max(total, 1)
        history["loss"].append(train_loss)
        history["acc"].append(train_acc)

        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            for eeg_v, nirs_v, y_v in val_loader:
                eeg_v = eeg_v.to(device)
                nirs_v = nirs_v.to(device)
                y_v = y_v.to(device)

                val_logits, val_aux = model.forward_with_aux(eeg_v, nirs_v)
                val_l_cls = criterion(val_logits, y_v)
                val_qreg = val_aux.get("quality_reg", torch.zeros((), device=device))
                val_loss_batch = val_l_cls + reg_scale * float(asym_quality_weight) * val_qreg

                val_loss_sum += val_loss_batch.item() * y_v.size(0)
                val_pred = torch.argmax(val_logits, dim=1)
                val_correct += (val_pred == y_v).sum().item()
                val_total += y_v.size(0)

            val_loss = val_loss_sum / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_eeg_acc"].append(np.nan)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1

        if epoch_logger is not None:
            epoch_logger(
                "{} Stage1 Epoch {}/{} reg_scale={:.1f} train_loss={:.4f} train_cls={:.4f} train_qreg={:.4f} train_meanreg={:.4f} train_acc={:.4f} val_loss={:.4f} val_acc={:.4f}".format(
                    fold_label,
                    ep,
                    int(hold_out_stage1_max_epochs),
                    reg_scale,
                    train_loss,
                    running_cls_loss / max(total, 1),
                    running_qreg_loss / max(total, 1),
                    running_meanreg_loss / max(total, 1),
                    train_acc,
                    val_loss,
                    val_acc,
                )
            )

        stage1_end_train_loss = float(train_loss)
        if stale_epochs >= int(hold_out_stage1_patience):
            if epoch_logger is not None:
                epoch_logger(
                    "{} Stage1 early stopping at epoch {} (patience={})".format(
                        fold_label,
                        ep,
                        int(hold_out_stage1_patience),
                    )
                )
            break
        model.train()

    if best_state is not None:
        model.load_state_dict(best_state)

    optimizer_stage2 = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep2 in range(1, int(hold_out_stage2_max_epochs) + 1):
        running_loss = 0.0
        running_correct = 0
        total = 0

        for eeg_b, nirs_b, y_b in loader_stage2:
            eeg_b = eeg_b.to(device)
            nirs_b = nirs_b.to(device)
            y_b = y_b.to(device)

            optimizer_stage2.zero_grad()
            logits, aux = model.forward_with_aux(eeg_b, nirs_b)
            sample_weight = aux.get("sample_weight", None)
            if sample_weight is not None:
                w = torch.clamp(sample_weight.squeeze(1), min=1e-3, max=1.0)
                if asym_detach_sample_weight:
                    w = w.detach()
                ce_per_sample = F.cross_entropy(logits, y_b, reduction="none")
                l_cls = torch.sum(ce_per_sample * w) / torch.sum(w)
                gate_mean_reg = (w.mean() - float(asym_gate_mean_target)) ** 2
            else:
                l_cls = criterion(logits, y_b)
                gate_mean_reg = torch.zeros((), device=device)
            qreg = aux.get("quality_reg", torch.zeros((), device=device))
            loss = l_cls + float(asym_quality_weight) * qreg + float(asym_gate_mean_weight) * gate_mean_reg

            loss.backward()
            optimizer_stage2.step()

            running_loss += loss.item() * y_b.size(0)
            pred = torch.argmax(logits, dim=1)
            running_correct += (pred == y_b).sum().item()
            total += y_b.size(0)

        train_loss_stage2 = running_loss / max(total, 1)
        train_acc_stage2 = running_correct / max(total, 1)
        history["loss"].append(train_loss_stage2)
        history["acc"].append(train_acc_stage2)
        history["val_loss"].append(np.nan)
        history["val_acc"].append(np.nan)
        history["val_eeg_acc"].append(np.nan)

        if epoch_logger is not None:
            epoch_logger(
                "{} Stage2 Epoch {}/{} train_loss={:.4f} train_acc={:.4f} stop_threshold={:.4f}".format(
                    fold_label,
                    ep2,
                    int(hold_out_stage2_max_epochs),
                    train_loss_stage2,
                    train_acc_stage2,
                    stage1_end_train_loss,
                )
            )

        if train_loss_stage2 < stage1_end_train_loss:
            if epoch_logger is not None:
                epoch_logger(
                    "{} Stage2 stopping: train_loss {:.4f} < stage1_end_train_loss {:.4f}".format(
                        fold_label,
                        train_loss_stage2,
                        stage1_end_train_loss,
                    )
                )
            break

    train_time_seconds = float(time.perf_counter() - train_start_time)
    model.eval()
    with torch.no_grad():
        infer_start_time = time.perf_counter()
        test_ds = TensorDataset(
            torch.from_numpy(eeg_test),
            torch.from_numpy(nirs_test),
            torch.from_numpy(y_test.astype(np.int64)),
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        test_loss_sum = 0.0
        test_total = 0
        prob_list = []
        pred_list = []
        for eeg_t, nirs_t, y_t in test_loader:
            eeg_t = eeg_t.to(device)
            nirs_t = nirs_t.to(device)
            y_t = y_t.to(device)

            logits = model(eeg_t, nirs_t)
            test_loss_sum += criterion(logits, y_t).item() * y_t.size(0)
            test_total += y_t.size(0)
            prob_list.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            pred_list.append(torch.argmax(logits, dim=1).cpu().numpy())

        test_loss = float(test_loss_sum / max(test_total, 1))
        prob = np.concatenate(prob_list, axis=0) if prob_list else np.empty((0,), dtype=np.float32)
        pred = np.concatenate(pred_list, axis=0) if pred_list else np.empty((0,), dtype=np.int64)
        test_time_seconds = float(time.perf_counter() - infer_start_time)

    return pred, prob, history, test_loss, train_time_seconds, test_time_seconds


def aggregate_metrics(metric_list: List[Dict[str, float]]):
    keys = list(metric_list[0].keys())
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in metric_list], dtype=np.float64)
        out[k] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }
    return out


def evaluate_loso(mode, eeg, nirs, y, subjects, seed, epochs, batch_size, lr, progress_bar, device, asym_cfg=None, on_fold_end=None):
    fold_metrics, histories = [], []
    y_true_all, y_pred_all = [], []
    fold_records = []
    fold_y_true, fold_y_pred = [], []
    fold_subjects = []

    sid_values = np.unique(subjects)
    sid_iter = sid_values
    if progress_bar and tqdm is not None:
        sid_iter = tqdm(sid_iter, desc=f"LOSO-{mode}", leave=True, dynamic_ncols=True)

    total_folds = int(len(sid_values))
    for fold_idx, sid in enumerate(sid_iter, start=1):
        test_idx = np.where(subjects == sid)[0]
        train_idx = np.where(subjects != sid)[0]

        fold_label = f"[{mode}][Fold {fold_idx}/{total_folds}][Subject {int(sid):02d}]"
        if progress_bar and tqdm is not None:
            epoch_logger = tqdm.write
        else:
            epoch_logger = print

        eeg_tr, eeg_te, nirs_tr, nirs_te = prepare_fold_data(
            mode,
            eeg[train_idx],
            eeg[test_idx],
            nirs[train_idx],
            nirs[test_idx],
            subjects[train_idx],
            subjects[test_idx],
            seed,
        )

        pred, prob, hist, test_loss, train_t, test_t = train_one_fold(
            mode,
            eeg_tr,
            y[train_idx],
            nirs_tr,
            eeg_te,
            nirs_te,
            y[test_idx],
            seed,
            epochs,
            batch_size,
            lr,
            device,
            asym_cfg,
            fold_label,
            epoch_logger,
            asym_quality_weight=float(asym_cfg.get("asym_quality_weight", 0.02)) if asym_cfg else 0.02,
            asym_gate_mean_weight=float(asym_cfg.get("asym_gate_mean_weight", 0.01)) if asym_cfg else 0.01,
            asym_gate_mean_target=float(asym_cfg.get("asym_gate_mean_target", 0.6)) if asym_cfg else 0.6,
            asym_detach_sample_weight=bool(asym_cfg.get("asym_detach_sample_weight", False)) if asym_cfg else False,
            early_stopping=bool(asym_cfg.get("early_stopping", False)) if asym_cfg else False,
            early_stopping_patience=int(asym_cfg.get("early_stopping_patience", 20)) if asym_cfg else 20,
            early_stopping_max_epochs=int(asym_cfg.get("early_stopping_max_epochs", 200)) if asym_cfg else 200,
        )

        metric = compute_metrics(y[test_idx], pred, prob)
        metric["loss"] = float(test_loss)
        fold_metrics.append(metric)
        histories.append(hist)
        y_true_all.append(y[test_idx])
        y_pred_all.append(pred)
        fold_y_true.append(y[test_idx].copy())
        fold_y_pred.append(pred.copy())
        fold_subjects.append(int(sid))
        fold_records.append(
            {
                "subject_id": int(sid),
                "n_test": int(len(test_idx)),
                "train_time_seconds": float(train_t),
                "test_time_seconds": float(test_t),
                "metrics": metric,
            }
        )

        msg = (
            f"{fold_label} done: test_acc={metric['accuracy']:.4f} "
            f"test_loss={metric['loss']:.4f} test_kappa={metric['kappa']:.4f} "
            f"train_time={fmt_minutes_seconds(train_t)} test_time={fmt_minutes_seconds(test_t)}"
        )
        if progress_bar and tqdm is not None:
            tqdm.write(msg)
        else:
            print(msg)

        if on_fold_end is not None:
            on_fold_end(
                fold_idx,
                total_folds,
                int(sid),
                y[test_idx].copy(),
                pred.copy(),
                hist,
                fold_records.copy(),
                fold_metrics.copy(),
            )

    artifacts = {
        "y_true_all": np.concatenate(y_true_all, axis=0),
        "y_pred_all": np.concatenate(y_pred_all, axis=0),
        "histories": histories,
        "fold_subjects": fold_subjects,
        "fold_y_true": fold_y_true,
        "fold_y_pred": fold_y_pred,
    }
    return aggregate_metrics(fold_metrics), fold_records, artifacts


def evaluate_within_session(mode, eeg, nirs, y, subjects, seed, epochs, batch_size, lr, progress_bar, device, asym_cfg=None):
    fold_metrics, histories = [], []
    y_true_all, y_pred_all = [], []
    fold_records = []
    fold_y_true, fold_y_pred = [], []
    fold_subjects = []

    sid_iter = np.unique(subjects)
    if progress_bar and tqdm is not None:
        sid_iter = tqdm(sid_iter, desc=f"Within-{mode}", leave=True, dynamic_ncols=True)

    for sid in sid_iter:
        idx = np.where(subjects == sid)[0]
        y_sid = y[idx]
        if len(np.unique(y_sid)) < 2 or len(y_sid) < 10:
            continue

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        local_train, local_test = next(splitter.split(np.zeros(len(y_sid)), y_sid))
        train_idx = idx[local_train]
        test_idx = idx[local_test]

        eeg_tr, eeg_te, nirs_tr, nirs_te = prepare_fold_data(
            mode,
            eeg[train_idx],
            eeg[test_idx],
            nirs[train_idx],
            nirs[test_idx],
            subjects[train_idx],
            subjects[test_idx],
            seed,
        )

        pred, prob, hist, test_loss, train_t, test_t = train_one_fold(
            mode,
            eeg_tr,
            y[train_idx],
            nirs_tr,
            eeg_te,
            nirs_te,
            y[test_idx],
            seed,
            epochs,
            batch_size,
            lr,
            device,
            asym_cfg,
            asym_quality_weight=float(asym_cfg.get("asym_quality_weight", 0.02)) if asym_cfg else 0.02,
            asym_gate_mean_weight=float(asym_cfg.get("asym_gate_mean_weight", 0.01)) if asym_cfg else 0.01,
            asym_gate_mean_target=float(asym_cfg.get("asym_gate_mean_target", 0.6)) if asym_cfg else 0.6,
            asym_detach_sample_weight=bool(asym_cfg.get("asym_detach_sample_weight", False)) if asym_cfg else False,
            early_stopping=bool(asym_cfg.get("early_stopping", False)) if asym_cfg else False,
            early_stopping_patience=int(asym_cfg.get("early_stopping_patience", 20)) if asym_cfg else 20,
            early_stopping_max_epochs=int(asym_cfg.get("early_stopping_max_epochs", 200)) if asym_cfg else 200,
        )

        metric = compute_metrics(y[test_idx], pred, prob)
        metric["loss"] = float(test_loss)
        fold_metrics.append(metric)
        histories.append(hist)
        y_true_all.append(y[test_idx])
        y_pred_all.append(pred)
        fold_y_true.append(y[test_idx].copy())
        fold_y_pred.append(pred.copy())
        fold_subjects.append(int(sid))
        fold_records.append(
            {
                "subject_id": int(sid),
                "n_test": int(len(test_idx)),
                "train_time_seconds": float(train_t),
                "test_time_seconds": float(test_t),
                "metrics": metric,
            }
        )

    artifacts = {
        "y_true_all": np.concatenate(y_true_all, axis=0),
        "y_pred_all": np.concatenate(y_pred_all, axis=0),
        "histories": histories,
        "fold_subjects": fold_subjects,
        "fold_y_true": fold_y_true,
        "fold_y_pred": fold_y_pred,
    }
    return aggregate_metrics(fold_metrics), fold_records, artifacts


def evaluate_lsso(mode, eeg, nirs, y, subjects, sessions, seed, epochs, batch_size, lr, progress_bar, device, asym_cfg=None):
    fold_metrics, histories = [], []
    y_true_all, y_pred_all = [], []
    fold_records = []
    fold_y_true, fold_y_pred = [], []
    fold_subjects = []

    session_values = np.unique(sessions)
    session_iter = session_values
    if progress_bar and tqdm is not None:
        session_iter = tqdm(session_iter, desc=f"LSSO-{mode}", leave=True, dynamic_ncols=True)

    total_folds = int(len(session_values))
    for fold_idx, sess in enumerate(session_iter, start=1):
        test_idx = np.where(sessions == sess)[0]
        train_idx = np.where(sessions != sess)[0]

        fold_label = f"[{mode}][LSSO Fold {fold_idx}/{total_folds}][Session {int(sess)}]"
        epoch_logger = tqdm.write if (progress_bar and tqdm is not None) else print

        eeg_tr, eeg_te, nirs_tr, nirs_te = prepare_fold_data(
            mode,
            eeg[train_idx],
            eeg[test_idx],
            nirs[train_idx],
            nirs[test_idx],
            subjects[train_idx],
            subjects[test_idx],
            seed,
        )

        pred, prob, hist, test_loss, train_t, test_t = train_one_fold(
            mode,
            eeg_tr,
            y[train_idx],
            nirs_tr,
            eeg_te,
            nirs_te,
            y[test_idx],
            seed,
            epochs,
            batch_size,
            lr,
            device,
            asym_cfg,
            fold_label,
            epoch_logger,
            asym_quality_weight=float(asym_cfg.get("asym_quality_weight", 0.02)) if asym_cfg else 0.02,
            asym_gate_mean_weight=float(asym_cfg.get("asym_gate_mean_weight", 0.01)) if asym_cfg else 0.01,
            asym_gate_mean_target=float(asym_cfg.get("asym_gate_mean_target", 0.6)) if asym_cfg else 0.6,
            asym_detach_sample_weight=bool(asym_cfg.get("asym_detach_sample_weight", False)) if asym_cfg else False,
            early_stopping=bool(asym_cfg.get("early_stopping", False)) if asym_cfg else False,
            early_stopping_patience=int(asym_cfg.get("early_stopping_patience", 20)) if asym_cfg else 20,
            early_stopping_max_epochs=int(asym_cfg.get("early_stopping_max_epochs", 200)) if asym_cfg else 200,
        )

        metric = compute_metrics(y[test_idx], pred, prob)
        metric["loss"] = float(test_loss)
        fold_metrics.append(metric)
        histories.append(hist)
        y_true_all.append(y[test_idx])
        y_pred_all.append(pred)
        fold_y_true.append(y[test_idx].copy())
        fold_y_pred.append(pred.copy())
        fold_subjects.append(int(sess))
        fold_records.append(
            {
                "subject_id": int(sess),
                "n_test": int(len(test_idx)),
                "train_time_seconds": float(train_t),
                "test_time_seconds": float(test_t),
                "metrics": metric,
            }
        )

    artifacts = {
        "y_true_all": np.concatenate(y_true_all, axis=0),
        "y_pred_all": np.concatenate(y_pred_all, axis=0),
        "histories": histories,
        "fold_subjects": fold_subjects,
        "fold_y_true": fold_y_true,
        "fold_y_pred": fold_y_pred,
    }
    return aggregate_metrics(fold_metrics), fold_records, artifacts


def evaluate_hold_out(
    mode,
    eeg,
    nirs,
    y,
    subjects,
    sessions,
    seed,
    batch_size,
    lr,
    progress_bar,
    device,
    asym_cfg=None,
    hold_out_cfg=None,
):
    hold_out_cfg = hold_out_cfg or {}
    subject_metrics = []
    subject_records = []

    y_true_all, y_pred_all = [], []
    histories = []
    fold_subjects, fold_sessions = [], []
    fold_y_true, fold_y_pred = [], []
    session_fold_records = []

    sid_values = np.unique(subjects)
    sid_iter = sid_values
    if progress_bar and tqdm is not None:
        sid_iter = tqdm(sid_iter, desc=f"HoldOut-{mode}", leave=True, dynamic_ncols=True)

    for sid in sid_iter:
        sid_mask = (subjects == sid)
        sid_sessions = np.unique(sessions[sid_mask])
        sid_session_metrics = []
        sid_train_time = 0.0
        sid_test_time = 0.0

        for sess in sid_sessions:
            test_idx = np.where((subjects == sid) & (sessions == sess))[0]
            train_idx = np.where((subjects == sid) & (sessions != sess))[0]
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue

            fold_label = f"[{mode}][HoldOut][Subject {int(sid):02d}][Session {int(sess)}]"
            epoch_logger = tqdm.write if (progress_bar and tqdm is not None) else print

            eeg_tr, eeg_te, nirs_tr, nirs_te = prepare_fold_data(
                mode,
                eeg[train_idx],
                eeg[test_idx],
                nirs[train_idx],
                nirs[test_idx],
                subjects[train_idx],
                subjects[test_idx],
                seed,
            )

            pred, prob, hist, test_loss, train_t, test_t = train_one_fold_holdout_two_stage(
                mode,
                eeg_tr,
                y[train_idx],
                nirs_tr,
                eeg_te,
                nirs_te,
                y[test_idx],
                seed,
                batch_size,
                lr,
                device,
                asym_cfg,
                fold_label,
                epoch_logger,
                hold_out_val_ratio=float(hold_out_cfg.get("val_ratio", 0.2)),
                hold_out_stage1_patience=int(hold_out_cfg.get("stage1_patience", 50)),
                hold_out_stage1_max_epochs=int(hold_out_cfg.get("stage1_max_epochs", 400)),
                hold_out_stage2_max_epochs=int(hold_out_cfg.get("stage2_max_epochs", 200)),
                asym_quality_weight=float(asym_cfg.get("asym_quality_weight", 0.02)) if asym_cfg else 0.02,
                asym_gate_mean_weight=float(asym_cfg.get("asym_gate_mean_weight", 0.01)) if asym_cfg else 0.01,
                asym_gate_mean_target=float(asym_cfg.get("asym_gate_mean_target", 0.6)) if asym_cfg else 0.6,
                asym_detach_sample_weight=bool(asym_cfg.get("asym_detach_sample_weight", False)) if asym_cfg else False,
            )

            metric = compute_metrics(y[test_idx], pred, prob)
            metric["loss"] = float(test_loss)
            sid_session_metrics.append(metric)
            sid_train_time += float(train_t)
            sid_test_time += float(test_t)

            y_true_all.append(y[test_idx])
            y_pred_all.append(pred)
            histories.append(hist)
            fold_subjects.append(int(sid))
            fold_sessions.append(int(sess))
            fold_y_true.append(y[test_idx].copy())
            fold_y_pred.append(pred.copy())

            session_fold_records.append(
                {
                    "subject_id": int(sid),
                    "session_id": int(sess),
                    "n_test": int(len(test_idx)),
                    "train_time_seconds": float(train_t),
                    "test_time_seconds": float(test_t),
                    "metrics": metric,
                }
            )

            msg = (
                f"{fold_label} done: test_acc={metric['accuracy']:.4f} "
                f"test_loss={metric['loss']:.4f} test_kappa={metric['kappa']:.4f} "
                f"train_time={fmt_minutes_seconds(train_t)} test_time={fmt_minutes_seconds(test_t)}"
            )
            if progress_bar and tqdm is not None:
                tqdm.write(msg)
            else:
                print(msg)

        if not sid_session_metrics:
            continue

        sid_summary = aggregate_metrics(sid_session_metrics)
        sid_metric_mean = {k: float(v["mean"]) for k, v in sid_summary.items()}
        subject_metrics.append(sid_metric_mean)
        subject_records.append(
            {
                "subject_id": int(sid),
                "n_test": int(np.sum([(r["n_test"]) for r in session_fold_records if int(r["subject_id"]) == int(sid)])),
                "n_sessions": int(len(sid_session_metrics)),
                "train_time_seconds": float(sid_train_time),
                "test_time_seconds": float(sid_test_time),
                "metrics": sid_metric_mean,
            }
        )

    if not subject_metrics:
        raise RuntimeError("Hold-out evaluation produced no valid subject records.")

    artifacts = {
        "y_true_all": np.concatenate(y_true_all, axis=0),
        "y_pred_all": np.concatenate(y_pred_all, axis=0),
        "histories": histories,
        "fold_subjects": fold_subjects,
        "fold_session_ids": fold_sessions,
        "fold_y_true": fold_y_true,
        "fold_y_pred": fold_y_pred,
        "session_fold_records": session_fold_records,
    }
    return aggregate_metrics(subject_metrics), subject_records, artifacts


def main():
    args = parse_args()
    np.random.seed(args.seed)
    train_start_dt = datetime.now()

    if torch is None:
        raise ImportError("PyTorch is required now. Please install torch in your environment.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modes = selected_modes(args)
    protocol = "within_session"
    protocol_flags = [bool(args.loso), bool(args.lsso), bool(args.hold_out)]
    if sum(protocol_flags) > 1:
        raise ValueError("--loso, --lsso and --hold_out are mutually exclusive. Please choose one.")
    if args.loso:
        protocol = "LOSO"
    elif args.lsso:
        protocol = "LSSO"
    elif args.hold_out:
        protocol = "HOLD_OUT"

    if args.sliding_window and args.window_size_s < args.asym_delay_max_s:
        raise ValueError(
            "Invalid configuration: window_size_s ({:.2f}s) is smaller than asym_delay_max_s ({:.2f}s). "
            "Either set --no_sliding_window or increase --window_size_s to be >= --asym_delay_max_s."
            .format(args.window_size_s, args.asym_delay_max_s)
        )

    effective_sliding_window = args.sliding_window
    if (not args.loso) and (not args.lsso) and (not args.hold_out) and args.sliding_window:
        print("[warn] within_session detected: sliding-window augmentation is auto-disabled to prevent train/test leakage.")
        effective_sliding_window = False

    root = Path(__file__).resolve().parent
    results_root = root / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    run_stamp = train_start_dt.strftime("%m%d_%H%M")

    print(f"Selected modes: {modes}")
    print(f"Protocol: {protocol}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Device: {device}")
    print(f"Progress bar: {args.progress_bar}")
    print(f"Write final all-results json: {args.write_final_results}")
    print(f"Sliding window augmentation: {effective_sliding_window}")
    print(
        "ATCNet switches: use_gating={} use_alignment={} use_mask={}".format(
            args.atcnet_use_gating,
            args.atcnet_use_alignment,
            args.atcnet_use_mask,
        )
    )
    print(
        "Asym16 switches: use_gating={} use_alignment={} use_dynamic_mask={}".format(
            args.asym16_use_gating,
            args.asym16_use_alignment,
            args.asym16_use_dynamic_mask,
        )
    )
    if effective_sliding_window:
        print(
            "Sliding window config: window_size_s={} step_s={} trial_duration_s={}".format(
                args.window_size_s,
                args.window_step_s,
                args.data_trial_duration_s,
            )
        )
    if args.progress_bar and tqdm is None:
        print("[warn] tqdm is not installed. Running without progress bar.")

    eeg, nirs, y, subjects, sessions = build_dataset(
        root,
        args.progress_bar,
        sliding_window=effective_sliding_window,
        window_size_s=args.window_size_s,
        window_step_s=args.window_step_s,
        data_trial_duration_s=args.data_trial_duration_s,
        return_session_ids=True,
    )
    print(f"Total paired trials: {len(y)}")
    print(f"Subjects used: {len(np.unique(subjects))}")

    all_results = {
        "run_stamp": run_stamp,
        "train_start_time": train_start_dt.isoformat(timespec="seconds"),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "protocol": protocol,
        "modes": modes,
        "n_trials": int(len(y)),
        "n_subjects": int(len(np.unique(subjects))),
        "results": {},
    }

    asym_cfg = {
        "asym_quality_weight": args.asym_quality_weight,
        "asym_gate_mean_weight": args.asym_gate_mean_weight,
        "asym_gate_mean_target": args.asym_gate_mean_target,
        "asym_reg_warmup_epochs": args.asym_reg_warmup_epochs,
        "asym_power_frame_rate_hz": args.asym_power_frame_rate_hz,
        "atcnet_use_gating": args.atcnet_use_gating,
        "atcnet_use_alignment": args.atcnet_use_alignment,
        "atcnet_use_mask": args.atcnet_use_mask,
        "asym16_use_gating": args.asym16_use_gating,
        "asym16_use_alignment": args.asym16_use_alignment,
        "asym16_use_dynamic_mask": args.asym16_use_dynamic_mask,
        "asym_detach_sample_weight": args.asym_detach_sample_weight,
        "asym_delay_min_s": args.asym_delay_min_s,
        "asym_delay_max_s": args.asym_delay_max_s,
        "asym_trial_duration_s": args.asym_trial_duration_s,
        "early_stopping": args.early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_max_epochs": args.early_stopping_max_epochs,
    }
    hold_out_cfg = {
        "val_ratio": args.hold_out_val_ratio,
        "stage1_patience": args.hold_out_stage1_patience,
        "stage1_max_epochs": args.hold_out_stage1_max_epochs,
        "stage2_max_epochs": args.hold_out_stage2_max_epochs,
    }

    mode_iter = modes
    if args.progress_bar and tqdm is not None:
        mode_iter = tqdm(modes, desc="Running modes", leave=True, dynamic_ncols=True)

    for mode in mode_iter:
        msg = f"\n=== Running mode: {mode} ==="
        if args.progress_bar and tqdm is not None:
            tqdm.write(msg)
        else:
            print(msg)
        mode_dir = results_root / f"{run_stamp}_{mode}_{protocol.lower()}_seed{args.seed}"
        checkpoint_dir = mode_dir / "checkpoints"
        confmat_dir = mode_dir / "confmats"
        curves_dir = mode_dir / "curves"
        tsne_dir = mode_dir / "tsne"
        fig_dir = mode_dir / "figures"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        confmat_dir.mkdir(parents=True, exist_ok=True)
        curves_dir.mkdir(parents=True, exist_ok=True)
        tsne_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

        if args.loso:
            def on_fold_end(
                fold_idx,
                total_folds,
                sid,
                y_true_fold,
                y_pred_fold,
                hist,
                running_fold_records,
                running_fold_metrics,
            ):
                save_binary_confusion_matrix(
                    y_true_fold,
                    y_pred_fold,
                    confmat_dir / f"confmat_subject_{sid}.png",
                    title=f"Confusion Matrix Subject {sid}",
                )
                save_single_fold_curves(
                    hist,
                    curves_dir / f"subject_{sid}_loss.png",
                    curves_dir / f"subject_{sid}_acc.png",
                    title_prefix=f"Subject {sid}",
                )

                rec = next((r for r in running_fold_records if int(r.get("subject_id", -1)) == int(sid)), None)
                if rec is not None:
                    write_single_subject_result_txt(mode_dir, mode, protocol, rec)

                if running_fold_records and running_fold_metrics:
                    running_summary = aggregate_metrics(running_fold_metrics)
                    write_mode_results_txt(
                        mode_dir,
                        mode,
                        protocol,
                        [int(r.get("subject_id", -1)) for r in running_fold_records],
                        running_fold_records,
                        running_summary,
                    )

            summary, fold_records, artifacts = evaluate_loso(
                mode,
                eeg,
                nirs,
                y,
                subjects,
                args.seed,
                args.epochs,
                args.batch_size,
                args.lr,
                args.progress_bar,
                device,
                asym_cfg,
                on_fold_end,
            )
        elif args.lsso:
            summary, fold_records, artifacts = evaluate_lsso(
                mode,
                eeg,
                nirs,
                y,
                subjects,
                sessions,
                args.seed,
                args.epochs,
                args.batch_size,
                args.lr,
                args.progress_bar,
                device,
                asym_cfg,
            )
        elif args.hold_out:
            summary, fold_records, artifacts = evaluate_hold_out(
                mode,
                eeg,
                nirs,
                y,
                subjects,
                sessions,
                args.seed,
                args.batch_size,
                args.lr,
                args.progress_bar,
                device,
                asym_cfg,
                hold_out_cfg,
            )
        else:
            summary, fold_records, artifacts = evaluate_within_session(
                mode,
                eeg,
                nirs,
                y,
                subjects,
                args.seed,
                args.epochs,
                args.batch_size,
                args.lr,
                args.progress_bar,
                device,
                asym_cfg,
            )

        cm_path = confmat_dir / "avg_confusion_matrix.png"
        curve_path = fig_dir / f"curve_{mode}.png"
        save_binary_confusion_matrix(
            artifacts["y_true_all"],
            artifacts["y_pred_all"],
            cm_path,
            title=f"Confusion Matrix ({protocol}, {mode})",
        )
        save_training_curve(
            artifacts["histories"],
            curve_path,
            title=f"Training Curves ({protocol}, {mode})",
        )

        if args.hold_out:
            for sid, sess, y_true_fold, y_pred_fold, hist in zip(
                artifacts["fold_subjects"],
                artifacts["fold_session_ids"],
                artifacts["fold_y_true"],
                artifacts["fold_y_pred"],
                artifacts["histories"],
            ):
                save_binary_confusion_matrix(
                    y_true_fold,
                    y_pred_fold,
                    confmat_dir / f"confmat_subject_{sid}_session_{sess}.png",
                    title=f"Confusion Matrix Subject {sid} Session {sess}",
                )
                save_single_fold_curves(
                    hist,
                    curves_dir / f"subject_{sid}_session_{sess}_loss.png",
                    curves_dir / f"subject_{sid}_session_{sess}_acc.png",
                    title_prefix=f"Subject {sid} Session {sess}",
                )
        elif not args.loso:
            for sid, y_true_fold, y_pred_fold, hist in zip(
                artifacts["fold_subjects"],
                artifacts["fold_y_true"],
                artifacts["fold_y_pred"],
                artifacts["histories"],
            ):
                save_binary_confusion_matrix(
                    y_true_fold,
                    y_pred_fold,
                    confmat_dir / f"confmat_subject_{sid}.png",
                    title=f"Confusion Matrix Subject {sid}",
                )
                save_single_fold_curves(
                    hist,
                    curves_dir / f"subject_{sid}_loss.png",
                    curves_dir / f"subject_{sid}_acc.png",
                    title_prefix=f"Subject {sid}",
                )

        write_mode_config_yaml(mode_dir, args, mode, protocol)
        if args.hold_out:
            write_mode_results_txt(
                mode_dir,
                mode,
                protocol,
                [int(r.get("subject_id", -1)) for r in fold_records],
                fold_records,
                summary,
            )
        elif not args.loso:
            write_mode_results_txt(
                mode_dir,
                mode,
                protocol,
                [int(s) for s in artifacts["fold_subjects"]],
                fold_records,
                summary,
            )

        mode_payload = {
            "run_stamp": run_stamp,
            "mode": mode,
            "protocol": protocol,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "n_trials": int(len(y)),
            "n_subjects": int(len(np.unique(subjects))),
            "n_folds": len(fold_records),
            "summary": summary,
            "fold_results": fold_records,
            "session_fold_results": artifacts.get("session_fold_records", []),
            "config_yaml": str(mode_dir / "config.yaml"),
            "results_txt": str(mode_dir / "results.txt"),
            "confusion_matrix_png": str(cm_path),
            "train_curve_png": str(curve_path),
        }
        mode_json = mode_dir / f"summary_{mode}.json"
        with open(mode_json, "w", encoding="utf-8") as f:
            json.dump(mode_payload, f, ensure_ascii=False, indent=2)

        all_results["results"][mode] = {
            "summary": summary,
            "n_folds": len(fold_records),
            "fold_results": fold_records,
            "result_dir": str(mode_dir),
            "mode_summary_json": str(mode_json),
            "config_yaml": str(mode_dir / "config.yaml"),
            "results_txt": str(mode_dir / "results.txt"),
            "confusion_matrix_png": str(cm_path),
            "train_curve_png": str(curve_path),
        }

        print(
            "accuracy={:.4f}, f1_macro={:.4f}, auc={:.4f}".format(
                summary["accuracy"]["mean"],
                summary["f1_macro"]["mean"],
                summary["auc"]["mean"],
            )
        )

    train_end_dt = datetime.now()
    all_results["train_end_time"] = train_end_dt.isoformat(timespec="seconds")
    all_results["train_duration_seconds"] = float((train_end_dt - train_start_dt).total_seconds())
    if args.write_final_results:
        out_json = results_root / f"{run_stamp}_all_{protocol.lower()}_seed{args.seed}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved summary: {out_json}")


if __name__ == "__main__":
    main()
