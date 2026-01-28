# Experiments Overview

This directory contains all experiments conducted in order to optimize the watermarking system.

## Experiment Details

### Experiment 00: Baseline 
**Config**:  
**Results**: Clean accuracy: , attacked accuracy: 77%, PSNR: 24.89 dB, ASR: 0%

---

### Experiment 01: Test AdamW optimizer
**Config**: 
**Results**: Clean accuracy: , attacked accuracy: 79%, PSNR: 23.14 dB, ASR: 0%

---

### Experiment 02: Step decay LR
**Config**: 
**Results**: Clean accuracy: , attacked accuracy: 73%, PSNR: 25.44 dB, ASR: 0%

---

### Experiment 03: Increased secret weight to 5.0
**Config**: 
**Results**: Clean accuracy: , attacked accuracy: 66%, PSNR: 25.36 dB, ASR: 0%

---

### Experiment 04: Increased secret weight to 10.0
**Config**: 
**Results**: Clean accuracy: , attacked accuracy: 66%, PSNR: 22.48 dB, ASR: 0%

---

### Experiment 05: 0.1 image penalty
**Config**: 
**Results**: Clean accuracy: , attacked accuracy: 66%, PSNR: 22.48 dB, ASR: 0%

---

### Experiment 06: Stronger watermark alpha = 0.20
**Config**: 
**Results**: Clean accuracy: , attacked accuracy: 66%, PSNR: 22.48 dB, ASR: 0%

---

### Experiment 07: Stronger watermark alpha = 0.25
**Config**: 
**Results**: Clean accuracy: , attacked accuracy: 66%, PSNR: 22.48 dB, ASR: 0%

---

### Experiment 08: AdamW optimizer + secret weight = 5.0
**Config**: 
**Results**: Clean accuracy: , attacked accuracy: 66%, PSNR: 22.48 dB, ASR: 0%

---

### Experiment 09: AdamW optimizer + secret weight = 5.0 + image penalty = 0.1
**Config**: 
**Results**: Clean accuracy: , attacked accuracy: 66%, PSNR: 22.48 dB, ASR: 0%

---

### Experiment 10: AdamW optimizer + secret weight = 5.0 + image penalty = 0.1 + alpha = 0.20
**Config**: 
**Results**: Clean accuracy: , attacked accuracy: 66%, PSNR: 22.48 dB, ASR: 0%

---

### Experiment 10: CosineAnnealingLR scheduler
**Config**: 
**Results**: Clean accuracy: , attacked accuracy: 66%, PSNR: 22.48 dB, ASR: 0%

---

## Summary Table


## Reproducibility

(скорректировать )Each experiment can be reproduced with:
```bash
cd experiments/exp_XX_name
python run.py
```

All experiments use the same:
- Random seed: 42
- Dataset: COCO val2017 (1500 images)
- Target secret: `0xBC4CA0EdA7`
- Batch size: 16
