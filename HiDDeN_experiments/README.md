# Experiments with StegaStamp Overview

This directory contains all experiments conducted in order to optimize the watermarking system.

## Experiment Details

## Experiment 01: Baseline 

### Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW (lr=2e-3, weight_decay=1e-4) |
| **Scheduler** | OneCycleLR (max_lr=2e-3, epochs=140) |
| **Image Size** | 128×128 pixels |
| **Batch Size** | 32 (train) / 16 (test) |
| **Loss Function** | `3.0 × MSE(message) + 0.3 × MSE(image)` |
| **Message Length** | 48 bits (12 hex chars) |
| **Architecture** | Encoder: 6×ConvBNRelu(64ch) + Decoder: 8×ConvBNRelu(64ch) |
| **Training Epochs** | 70 |
| **Device** | CUDA |

## Results

### Bit Accuracy (%)

| Condition | Accuracy |
|-----------|----------|
| **Clean** (no attack) | ~83% |
| **Blur σ=0.5** | ~75% |
| **Blur σ=1.0** | ~56% |
| **Blur σ=2.0** | ~45% |
| **Rotation 5°** | ~80% |
| **Rotation 15°** | ~74% |
| **Rotation 30°** | ~55% |

### Image Quality & Exact Recovery

| Metric | Value |
|--------|-------|
| **PSNR** | ~35-40 dB |
| **Exact Recovery (Blur σ=1.0)** | Low (<20%) |
| **Exact Recovery (Rot 15°)** | Moderate (~40-50%) |
| **ASR (Attack Success Rate)** | ~0% (message still partially decodable) |

### Key Observations

1. **Clean Performance**: Model achieves ~83% bit accuracy on unperturbed images
2. **Blur Sensitivity**: Gaussian blur is the most damaging attack; accuracy drops ~30% at σ=1.0
3. **Rotation Robustness**: Moderate resilience to rotation (~74% at 15°)
4. **Progressive Training**: Helps but doesn't achieve strong robustness against aggressive attacks
5. **Exact Recovery**: Bit-level accuracy doesn't guarantee full secret recovery; practical deployment needs >95% bit accuracy

<img width="2100" height="750" alt="metrics_plot_1" src="https://github.com/user-attachments/assets/6b7f9c6a-327a-4238-b531-7303a1b94f3c" />
<img width="2400" height="600" alt="qualitative_plot_1" src="https://github.com/user-attachments/assets/681ab25d-bae7-4dd9-ba21-c5ee9ee3dfcc" />

