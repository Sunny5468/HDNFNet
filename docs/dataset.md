# Dataset Note

This project targets the public simultaneous EEG-fNIRS BCI dataset introduced by Shin et al., containing:

- Dataset A: Motor imagery (left vs right hand)
- Dataset B: Mental arithmetic vs baseline
- 29 subjects
- EEG and fNIRS recorded simultaneously

The original local training pipeline expects dataset files under a path similar to:

```text
<data_root>/EEG-fNIRs异构数据集/
  EEG_01-29/subject xx/with occular artifact/{cnt.mat,mrk.mat}
  NIRS_01-29/subject xx/{cnt.mat,mrk.mat}
```

This repository does not redistribute raw dataset files.

## Reference

- Shin, J. et al. Open access dataset for EEG-NIRS hybrid BCI.
