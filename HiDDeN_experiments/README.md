# Experiments with HiDDeN Overview

This directory contains all experiments conducted in order to optimize the watermarking system.

## Experiment Details

### Experiment 1: Baseline 

#### Configuration

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

### Results

#### Bit Accuracy (%)

| Condition | Accuracy |
|-----------|----------|
| **Clean** (no attack) | ~83% |
| **Blur σ=0.5** | ~75% |
| **Blur σ=1.0** | ~56% |
| **Blur σ=2.0** | ~45% |
| **Rotation 5°** | ~80% |
| **Rotation 15°** | ~74% |
| **Rotation 30°** | ~55% |

#### Image Quality & Exact Recovery

| Metric | Value |
|--------|-------|
| **PSNR** | ~35-40 dB |
| **Exact Recovery (Blur σ=1.0)** | Low (<20%) |
| **Exact Recovery (Rot 15°)** | Moderate (~40-50%) |
| **ASR (Attack Success Rate)** | ~0% (message still partially decodable) |

#### Key Observations

1. **Clean Performance**: Model achieves ~83% bit accuracy on unperturbed images
2. **Blur Sensitivity**: Gaussian blur is the most damaging attack; accuracy drops ~30% at σ=1.0
3. **Rotation Robustness**: Moderate resilience to rotation (~74% at 15°)
4. **Progressive Training**: Helps but doesn't achieve strong robustness against aggressive attacks
5. **Exact Recovery**: Bit-level accuracy doesn't guarantee full secret recovery; practical deployment needs >95% bit accuracy

<img width="2100" height="750" alt="metrics_plot_1" src="https://github.com/user-attachments/assets/6b7f9c6a-327a-4238-b531-7303a1b94f3c" />

<img width="2400" height="600" alt="qualitative_plot_1" src="https://github.com/user-attachments/assets/681ab25d-bae7-4dd9-ba21-c5ee9ee3dfcc" />

### Experiment 2: BCE message loss optimizer

#### Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW (lr=2e-3, weight_decay=1e-4) |
| **Scheduler** | OneCycleLR (max_lr=2e-3, epochs=140) |
| **Image Size** | 128×128 pixels |
| **Batch Size** | 32 (train) / 16 (test) |
| **Loss Function** | `3.0 × BCEWithLogits(message) + 0.3 × MSE(image)` |
| **Message Length** | 48 bits (12 hex chars) |
| **Architecture** | Encoder: 6×ConvBNRelu(64ch) + Decoder: 8×ConvBNRelu(64ch) |
| **Training Epochs** | 80 |
| **Device** | CUDA|

### Results

#### Bit Accuracy (%)

| Condition | Accuracy |
|-----------|----------|
| **Clean** (σ=0.0 / 0°) | ~72% |
| **Blur σ=0.5** | ~72% |
| **Blur σ=1.0** | ~64% |
| **Blur σ=1.5** | ~64% |
| **Blur σ=2.0** | ~62% |
| **Blur σ=3.0** | ~62% |
| **Rotation 5°** | ~72% |
| **Rotation 10°** | ~70% |
| **Rotation 15°** | ~70% |
| **Rotation 20°** | ~71% |
| **Rotation 30°** | ~73% |

#### Image Quality & Exact Recovery

| Metric | Value |
|--------|-------|
| **PSNR** | ~32-36 dB (visible watermark pattern) |
| **Exact Recovery (Blur σ=1.0)** | Low-Moderate (~25-40%) |
| **Exact Recovery (Rot 15°)** | Moderate (~35-50%) |
| **ASR (Attack Success Rate)** | ~0% |

#### Key Observations

1. **Clean Performance**: Model achieves ~72% bit accuracy on unperturbed images (lower than expected for BCE loss)
2. **Blur Sensitivity**: Significant drop at σ=1.0 (~72% → ~64%, -8%); plateaus at ~62% for higher sigma
3. **Rotation Robustness**: Remarkably stable across all angles (70-73%); rotation is less damaging than blur
4. **BCE vs MSE**: Despite BCE loss theoretical advantages, actual performance is modest (~72% clean vs ~83% in Exp.01)
5. **Watermark Visibility**: Qualitative images show visible pattern/noise in watermarked images (trade-off for robustness)
6. **Attack Recovery**: After blur (σ=1) and rotation (15°), watermark pattern remains partially decodable
7. **Plateau Effect**: Accuracy stabilizes at ~62-64% for blur σ≥1.0, suggesting model hits robustness ceilingith BCEWithLogits for message loss yields consistent improvements in both clean accuracy (+5-10%) and attack robustness (+8-15%). This confirms that aligning the loss function with the binary nature of the task is a low-cost, high-impact optimization. The model shows better gradient flow and more reliable bit predictions, though at a minor cost to image quality (PSNR). Next steps should include JPEG/noise augmentation and perceptual loss for better image fidelity.

<img width="2100" height="750" alt="metrics_plot_2" src="https://github.com/user-attachments/assets/e64700e5-34f4-4e11-ad31-e003d316d415" />
<img width="2400" height="600" alt="qualitative_plot_2" src="https://github.com/user-attachments/assets/9cc97236-b5eb-4b1d-9dae-df5ac78bfd42" />

### Experiment 3: heavier message weight optimizer

#### Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW (lr=2e-3, weight_decay=1e-4) |
| **Scheduler** | OneCycleLR (max_lr=2e-3, epochs=140) |
| **Image Size** | 128×128 pixels |
| **Batch Size** | 32 (train) / 16 (test) |
| **Loss Function** | `5.0 × MSE(message) + 0.3 × MSE(image)` |
| **Message Weight** | **5.0** (increased from 3.0 in Exp.01) |
| **Message Length** | 48 bits (12 hex chars) |
| **Architecture** | Encoder: 6×ConvBNRelu(64ch) + Decoder: 8×ConvBNRelu(64ch) |
| **Training Epochs** | 80 |
| **Device** | CUDA / CPU auto-detect |

### Results

#### Bit Accuracy (%) - From Metrics Plot

| Condition | Accuracy |
|-----------|----------|
| **Clean** (σ=0.0 / 0°) | ~90% |
| **Blur σ=0.5** | ~90% |
| **Blur σ=1.0** | ~63% |
| **Blur σ=1.5** | ~54% |
| **Blur σ=2.0** | ~53% |
| **Blur σ=3.0** | ~54% |
| **Rotation 0°** | ~90% |
| **Rotation 5°** | ~89% |
| **Rotation 10°** | ~87% |
| **Rotation 15°** | ~81% |
| **Rotation 20°** | ~76% |
| **Rotation 30°** | ~62% |

#### Image Quality & Exact Recovery

| Metric | Value |
|--------|-------|
| **PSNR** | ~30-35 dB (visible watermark pattern) |
| **Exact Recovery (Blur σ=1.0)** | Low (~20-35%) |
| **Exact Recovery (Rot 15°)** | Moderate (~40-55%) |
| **ASR (Attack Success Rate)** | ~0% |

#### Key Observations

1. **Highest Clean Performance**: Model achieves ~90% bit accuracy on unperturbed images - best among all experiments (+7% vs Exp.01, +18% vs Exp.02)
2. **Message Weight Impact**: Increasing message loss weight from 3.0 to 5.0 significantly improves clean accuracy but doesn't prevent blur degradation
3. **Blur Vulnerability**: Sharp drop at σ=1.0 (~90% → ~63%, -27%); plateaus at ~53-54% for σ≥1.5 (similar floor to previous experiments)
4. **Rotation Robustness**: Superior rotation performance with gradual decline (~90% → ~81% at 15°, then ~62% at 30°)
5. **Trade-off Confirmation**: Higher message weight prioritizes decoding accuracy over image quality (visible watermark pattern in qualitative results)
6. **Blur vs Rotation**: Rotation attacks are significantly better tolerated than blur at equivalent severity levels
7. **Watermark Visibility**: Qualitative images show prominent noise pattern in watermarked images, confirming PSNR trade-off

<img width="915" height="312" alt="metrics_plot_4" src="https://github.com/user-attachments/assets/45554b2a-a4c0-4047-8010-43e37a259e7c" />

<img width="899" height="239" alt="qualitative_plot_4" src="https://github.com/user-attachments/assets/d9e0970c-ee11-4999-b639-08dd8f8c829f" />



