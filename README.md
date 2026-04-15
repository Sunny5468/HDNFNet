# HDNFNet

HDNFNet (Hemodynamic-Delay-aware NeuroFusion Network) is a hybrid EEG-fNIRS decoding framework for BCI tasks. The core idea is to model neurovascular delay explicitly during cross-modal fusion, instead of aligning EEG and fNIRS at identical timestamps.
<img width="340" height="242" alt="image" src="https://github.com/user-attachments/assets/9ad222b5-0917-4598-83f9-7fd659b891b4" />
<img width="392" height="184" alt="image" src="https://github.com/user-attachments/assets/e4c27e9f-b8e4-4281-8ec1-8004153e5583" />
<img width="252" height="151" alt="image" src="https://github.com/user-attachments/assets/6a971fea-1f8e-4ce7-9bf1-e6b90235bbbe" />
<img width="335" height="199" alt="image" src="https://github.com/user-attachments/assets/8a20e840-bc8f-4a33-8e67-6b64ef047337" />
<img width="257" height="118" alt="image" src="https://github.com/user-attachments/assets/a239113e-1ee1-42cf-8cfe-819309e10f53" />
<img width="232" height="185" alt="image" src="https://github.com/user-attachments/assets/a80d4b9c-6b10-44e7-922c-23f2d424b68e" />

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

## Current Included Results (LOSO)

- MA full model mean accuracy: `0.7557`
- MA full model mean kappa: `0.5115`

## Notes

- Core HDNFNet architecture is provided in `src/hdnfnet/model.py`.

## Citation

See `CITATION.cff`.
