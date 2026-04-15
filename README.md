# HDNFNet

HDNFNet (Hemodynamic-Delay-aware NeuroFusion Network) is a hybrid EEG-fNIRS decoding framework for BCI tasks. The core idea is to model neurovascular delay explicitly during cross-modal fusion, instead of aligning EEG and fNIRS at identical timestamps.

This repository packages the current paper draft, core model implementation, training pipeline script, and representative LOSO result summaries for MI and MA tasks.

## Highlights

- Delay-constrained cross-modal attention (`3-8 s`) for physiologically plausible EEG-fNIRS alignment.
- Delay-aware temporal mask modulation that modulates EEG features using aligned fNIRS context.
- Sample-wise physiological quality gating to down-weight noisy trials during training.
- Asymmetric design: EEG as primary decoding stream, fNIRS as guidance stream.

## Repository Structure

```text
.
├── src/hdnfnet/
│   ├── __init__.py
│   └── model.py
├── scripts/
│   └── run_hybrid_pipeline.py
├── paper/
│   ├── HDNFNet.tex
│   ├── references.bib
│   └── figures/
├── results/
│   ├── MI/
│   └── MA/
├── docs/
│   ├── dataset.md
│   └── baselines.md
├── .gitignore
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

## Quick Start

1. Create environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Import model:

```python
from hdnfnet import HemoDelayNeuroFusionNet

model = HemoDelayNeuroFusionNet(
    eeg_channels=30,
    eeg_samples=2000,
    fnirs_channels=36,
    fnirs_samples=100,
    num_classes=2,
)
```

3. Training pipeline script:

```bash
python scripts/run_hybrid_pipeline.py --atcnet --loso --epochs 40 --seed 42
```

## Current Included Results (LOSO)

- MI full/pure baseline summary: `results/MI/`
- MA full model summary: `results/MA/summary_full_model.json`
- MA full model mean accuracy: `0.7557`
- MA full model mean kappa: `0.5115`

## Notes

- `scripts/run_hybrid_pipeline.py` references local modules (`strict_riemann_alignment`, `models`, `model_16`) from the original development workspace. Those modules are not included in this snapshot.
- Core HDNFNet architecture is provided in `src/hdnfnet/model.py`.

## Citation

See `CITATION.cff`.
