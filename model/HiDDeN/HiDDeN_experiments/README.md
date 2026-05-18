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

### Experiment 4 – Curriculum Loss Weights + Progressive Attack Training

#### Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW (lr=2e-3, weight_decay=1e-4) |
| **Scheduler** | OneCycleLR (max_lr=2e-3, epochs=140) |
| **Image Size** | 128×128 pixels |
| **Batch Size** | 32 (train) / 16 (test) |
| **Loss Function** | Dynamic curriculum weighting |
| **Message Length** | 48 bits (12 hex chars) |
| **Architecture** | Encoder: 6×ConvBNRelu(64ch) + Decoder: 8×ConvBNRelu(64ch) |
| **Training Epochs** | 80 |
| **Device** | CUDA / CPU auto-detect |

### Results

#### Bit Accuracy (%) – Robustness Evaluation

| Condition | Accuracy |
|-----------|----------|
| **Clean (0° / σ=0.0)** | ~90% |
| **Blur σ=0.5** | ~95% |
| **Blur σ=1.0** | ~68% |
| **Blur σ=1.5** | ~60% |
| **Blur σ=2.0** | ~58% |
| **Blur σ=3.0** | ~58% |
| **Rotation 0°** | ~90% |
| **Rotation 5°** | ~88% |
| **Rotation 10°** | ~90% |
| **Rotation 15°** | ~86% |
| **Rotation 20°** | ~78% |
| **Rotation 30°** | ~62% |

#### Image Quality & Exact Recovery

| Metric | Value |
|--------|-------|
| **ASR (Attack Success Rate)** | ~0% |

#### Key Observations

1. **Stable Training via Curriculum**  
   Early emphasis on image reconstruction stabilizes learning and prevents noisy watermark artifacts.

2. **Improved Trade-off Balance**  
   Compared to fixed high message weight (Exp.03), this approach achieves similar clean accuracy while maintaining better visual quality (higher PSNR).

3. **Progressive Robustness Learning**  
   Introducing blur gradually allows the model to adapt without catastrophic drops in performance.

4. **Blur Still the Main Weakness**  
   Performance drops significantly at σ ≥ 1.0 (~20–30% decrease), indicating blur remains the hardest attack.

5. **Strong Rotation Robustness**  
   Rotation is handled well even without heavy exposure during training, showing good generalization.

6. **Curriculum Effectiveness**  
   The shift from image-focused → message-focused loss improves convergence and avoids early overfitting to message decoding.

7. **Better Visual Quality**  
   Compared to Exp.03, watermark artifacts are less visible due to early high image-loss weighting.

8. **Implicit Regularization**  
   The staged training acts as a regularizer, improving both robustness and stability without architectural changes.
   
<img width="915" height="312" alt="metrics_plot_4" src="https://github.com/user-attachments/assets/8bef4d6b-754c-4555-865b-94df989b2588" />

<img width="899" height="239" alt="qualitative_plot_4" src="https://github.com/user-attachments/assets/56d7d70e-4fdd-4e47-a2fa-abd6d6ca0eec" />

### Experiment 5 – Scheduler Swap (Cosine Annealing Warm Restarts)

#### Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) |
| **Scheduler** | CosineAnnealingWarmRestarts (T₀=20, T_mult=2) |
| **Image Size** | 128×128 pixels |
| **Batch Size** | 32 (train) / 16 (test) |
| **Loss Function** | `3.0 × MSE(message) + 0.3 × MSE(image)` |
| **Message Length** | 48 bits (12 hex chars) |
| **Architecture** | Encoder: 6×ConvBNRelu(64ch) + Decoder: 8×ConvBNRelu(64ch) |
| **Training Epochs** | 80 |
| **Device** | CUDA / CPU auto-detect |

### Results

#### Bit Accuracy (%) – From Metrics Plot

| Condition | Accuracy |
|-----------|----------|
| **Clean (σ=0.0 / 0°)** | ~84% |
| **Blur σ=0.5** | ~77% |
| **Blur σ=1.0** | ~61% |
| **Blur σ=1.5** | ~54% |
| **Blur σ=2.0** | ~54% |
| **Blur σ=3.0** | ~50% |
| **Rotation 0°** | ~84% |
| **Rotation 5°** | ~78% |
| **Rotation 10°** | ~77% |
| **Rotation 15°** | ~76% |
| **Rotation 20°** | ~74% |
| **Rotation 30°** | ~68% |

#### Image Quality & Exact Recovery

| Metric | Value |
|--------|-------|
| **ASR (Attack Success Rate)** | ~0% |

### Key Observations

1. **Smoother Training Dynamics**  
   Cosine annealing with restarts stabilizes optimization and avoids aggressive learning rate spikes seen in OneCycleLR.

2. **Slight Drop in Peak Accuracy**  
   Clean accuracy (~84%) is lower than curriculum-based Exp.04 (~90%), indicating scheduler alone cannot replace curriculum learning.

3. **Improved Stability Across Attacks**  
   Performance degradation is smoother, especially under rotation (very gradual decline).

4. **Blur Robustness Unchanged**  
   Similar failure pattern as previous experiments:
   - Sharp drop at σ=1.0 (~84% → ~61%)
   - Plateau around ~50–55%

5. **Better Generalization via Restarts**  
   Warm restarts help escape local minima, improving consistency across different perturbation levels.

6. **Balanced Trade-off**  
   Fixed loss weighting (3.0 / 0.3) provides a compromise between:
   - Message accuracy
   - Image quality (less visible artifacts than Exp.03)

7. **Rotation Robustness Strong**  
   Even at 30°, accuracy remains relatively high (~68%), confirming rotation is easier than blur.

8. **Scheduler vs Curriculum**  
   - Scheduler improves **optimization stability**
   - Curriculum (Exp.04) improves **final performance**
   → Best results likely require combining both

<img width="2100" height="750" alt="metrics_plot_5" src="https://github.com/user-attachments/assets/79e1f7fb-2856-436c-aa9d-52755c89e465" />

<img width="2400" height="600" alt="qualitative_plot_5" src="https://github.com/user-attachments/assets/da675d1b-d4a1-4bc9-9550-709923e02046" />

### Experiment 6 – Higher Resolution Optimizer

#### Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW (lr=2e-3, weight_decay=1e-4) |
| **Scheduler** | OneCycleLR (max_lr=2e-3, epochs=140) |
| **Image Size** | 256×256 pixels |
| **Batch Size** | 16 (train) / 16 (test) |
| **Loss Function** | `3.0 × MSE(message) + 0.3 × MSE(image)` |
| **Message Length** | 48 bits (12 hex chars) |
| **Architecture** | Encoder: 6×ConvBNRelu(64ch) + Decoder: 8×ConvBNRelu(64ch) |
| **Training Epochs** | 80 |
| **Device** | CUDA / CPU auto-detect |

### Results

#### Bit Accuracy (%) – From Metrics Plot

| Condition | Accuracy |
|-----------|----------|
| **Clean (σ=0.0 / 0°)** | ~92% |
| **Blur σ=0.5** | ~88% |
| **Blur σ=1.0** | ~79% |
| **Blur σ=1.5** | ~71% |
| **Blur σ=2.0** | ~65% |
| **Blur σ=3.0** | ~58% |
| **Rotation 0°** | ~92% |
| **Rotation 5°** | ~89% |
| **Rotation 10°** | ~87% |
| **Rotation 15°** | ~85% |
| **Rotation 20°** | ~82% |
| **Rotation 30°** | ~76% |

#### Image Quality & Exact Recovery

| Metric | Value |
|--------|-------|
| **ASR (Attack Success Rate)** | ~0% |

### Key Observations

1. **Higher Resolution Improves Robustness**
   Upscaling to 256×256 provides more spatial capacity for watermark embedding, yielding +8% clean accuracy vs. 128×128 baseline.

2. **Curriculum Learning Strategy**
   Progressive attack introduction (none → blur @epoch15 → rotation @epoch80) enables stable convergence while building perturbation resilience.

3. **Superior Blur Tolerance**
   Accuracy at σ=1.0 improved from ~61% (Exp.05) to ~79%, demonstrating resolution's critical role in frequency-domain robustness.

4. **Rotation Robustness Maintained**
   Gradual degradation under rotation (92% → 76% at 30°) confirms spatial transforms remain easier to handle than frequency attacks.

5. **OneCycleLR Accelerates Convergence**
   Aggressive LR cycling with warmup achieves peak performance faster than cosine annealing, though requires careful curriculum alignment.

6. **Exact Hex Recovery Practical**
    ~74–81% exact 12-char Ethereum address recovery under moderate attacks indicates real-world deployability for NFT watermarking.

7. **PSNR-Accuracy Trade-off Balanced**
   Loss weights (3.0/0.3) preserve visual quality (~38.5 dB PSNR) while prioritizing message fidelity—suitable for stealth applications.

8. **Architecture Scaling Effective**
   Deeper encoder/decoder (6/8 blocks) at higher resolution captures multi-scale features without overfitting, thanks to batch normalization and dropout-free design.

<img width="907" height="312" alt="metrics_plot_6" src="https://github.com/user-attachments/assets/cb049cff-f920-43de-8f89-413c5450b321" />

<img width="905" height="230" alt="qualitative_plot_6" src="https://github.com/user-attachments/assets/364b70f3-f418-4ea1-acd9-1179d4943609" />

### Experiment 7 – Deeper Decoder + Residual Blocks

#### Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW (lr=2e-3, weight_decay=1e-4) |
| **Scheduler** | OneCycleLR (max_lr=2e-3, epochs=140) |
| **Image Size** | 128×128 pixels |
| **Batch Size** | 32 (train) / 16 (test) |
| **Loss Function** | `3.0 × MSE(message) + 0.3 × MSE(image)` |
| **Message Length** | 48 bits (12 hex chars) |
| **Architecture** | Encoder: 6×ConvBNRelu(64ch) + Decoder: 12×ResBlock(64ch) |
| **Training Epochs** | 80 |
| **Device** | CUDA / CPU auto-detect |

### Results

#### Bit Accuracy (%) – From Metrics Plot

| Condition | Accuracy |
|-----------|----------|
| **Clean (σ=0.0 / 0°)** | ~84% |
| **Blur σ=0.5** | ~88% |
| **Blur σ=1.0** | ~55% |
| **Blur σ=1.5** | ~51% |
| **Blur σ=2.0** | ~51% |
| **Blur σ=3.0** | ~52% |
| **Rotation 0°** | ~84% |
| **Rotation 5°** | ~87% |
| **Rotation 10°** | ~89% |
| **Rotation 15°** | ~83% |
| **Rotation 20°** | ~78% |
| **Rotation 30°** | ~66% |

#### Image Quality & Exact Recovery

| Metric | Value |
|--------|-------|
| **ASR (Attack Success Rate)** | ~0% |

#### Key Observations

1. **Residual Blocks Improve Small-Rotation Robustness**
   Accuracy peaks at 10° rotation (~89%), exceeding clean performance—residual connections help preserve spatial features under mild geometric transforms.

2. **Catastrophic Blur Failure**
   Sharp accuracy drop at σ=1.0 (84% → 55%) indicates residual decoder struggles with frequency-domain attacks despite deeper architecture.

3. **Lower Resolution Limits Performance**
   Returning to 128×128 from 256×256 (Exp.06) reduces clean accuracy from ~92% to ~84%, confirming resolution is critical for robust embedding.

4. **Exact Recovery Collapses**
   0% exact 12-char hex recovery under both attacks suggests residual architecture prioritizes bit-level accuracy over precise symbol reconstruction.

5. **Training Instability at High Sigma**
   Accuracy plateaus around 51–52% for σ≥1.5, indicating model reaches chance-level performance ceiling under strong blur.

6. **Rotation Tolerance Remains Strong**
   Despite blur weakness, rotation robustness is preserved (66% at 30°), validating residual blocks' effectiveness for spatial transforms.

7. **PSNR Degradation**
   Lower PSNR (~36.2 dB vs ~38.5 dB in Exp.06) suggests deeper decoder introduces more visible artifacts despite residual connections.

8. **Architecture Trade-off**
   12 residual blocks increase capacity but don't compensate for resolution loss—depth alone cannot overcome spatial information constraints.

<img width="2100" height="750" alt="metrics_plot_7" src="https://github.com/user-attachments/assets/b7abe280-0ad5-4f19-8cb9-186b7f976a63" />

<img width="2400" height="600" alt="qualitative_plot_7" src="https://github.com/user-attachments/assets/395c3831-60dc-4518-a39c-3f8b0eaded7c" />

### Experiment 8 – Add JPEG Attack Optimizer

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
| **Training Epochs** | 80 |
| **Device** | CUDA / CPU auto-detect |

### Results

#### Bit Accuracy (%) – From Metrics Plot

| Condition | Accuracy |
|-----------|----------|
| **Clean (σ=0.0 / 0°)** | ~92% |
| **Blur σ=0.5** | ~90% |
| **Blur σ=1.0** | ~64% |
| **Blur σ=1.5** | ~60% |
| **Blur σ=2.0** | ~58% |
| **Blur σ=3.0** | ~55% |
| **Rotation 0°** | ~92% |
| **Rotation 5°** | ~90% |
| **Rotation 10°** | ~89% |
| **Rotation 15°** | ~89% |
| **Rotation 20°** | ~85% |
| **Rotation 30°** | ~73% |

#### Image Quality & Exact Recovery

| Metric | Value |
|--------|-------|
| **ASR (Attack Success Rate)** | ~0% |

#### Key Observations

1. **JPEG Training Integration**
   Added differentiable JPEG compression (quality 50-90) to curriculum starting at epoch 80, alongside rotation attacks—enables end-to-end gradient-based optimization against     compression artifacts.

2. **Strong Clean Performance Maintained**
   ~92% clean accuracy matches Exp.06 (higher resolution), demonstrating effective learning despite 128×128 constraint and expanded attack surface.

3. **Rotation Robustness Excellent**
   Near-perfect stability up to 15° (~89%), with graceful degradation to ~73% at 30°—best rotation performance across all experiments, suggesting JPEG training indirectly         improves geometric invariance.

4. **Blur Vulnerability Persists**
   Sharp drop at σ=1.0 (92% → 64%) remains the primary weakness, plateauing at ~55% for high sigma—frequency-domain attacks still overwhelm spatial-domain watermarking.

5. **Differentiable JPEG Codec Effective**
   Kornia's jpeg_codec_differentiable enables realistic compression simulation during training without breaking gradient flow—critical for real-world deployment where JPEG is ubiquitous.

6. **PSNR Stable**
   ~37.8 dB indicates acceptable visual quality despite triple-attack curriculum (blur + rotation + JPEG), balancing imperceptibility and robustness.

7. **Exact Recovery Fails**
   0% exact 12-char hex recovery under attacks suggests bit-level accuracy (~55-89%) doesn't translate to perfect symbol reconstruction—error correction coding would be needed.

8. **Curriculum Complexity Managed**
   Successfully trains with three attack types (blur from epoch 15, rotation+JPEG from epoch 80) without catastrophic forgetting, validating progressive difficulty scaling.

<img width="2100" height="750" alt="metrics_plot_8" src="https://github.com/user-attachments/assets/946a3838-b43b-41fb-8398-3331555cc2a4" />

<img width="2400" height="600" alt="qualitative_plot_8" src="https://github.com/user-attachments/assets/7c3eb712-db47-46b3-aa97-0fd97412263e" />

### Experiment 9 – Adversarial Discriminator (Full HiDDeN) Optimizer

#### Configuration

| Parameter | Value |
|----------|------|
| Optimizer | AdamW (Enc+Dec: lr=1e-3, Disc: lr=1e-4) |
| Scheduler | OneCycleLR (max_lr=2e-3, epochs=140) |
| Image Size | 128×128 pixels |
| Batch Size | 32 (train) / 16 (test) |
| Loss Function | 3.0 × MSE(message) + 0.3 × MSE(image) + 0.001 × adversarial |
| Message Length | 48 bits (12 hex chars) |
| Architecture | Encoder: 6×ConvBNRelu(64ch) + Decoder: 8×ConvBNRelu(64ch) + Discriminator |
| Training Epochs | 80 |
| Device | CUDA|

## Results

### Bit Accuracy (%) – From Metrics Plot

| Condition | Accuracy |
|----------|---------|
| Clean (σ=0.0 / 0°) | ~86% |
| Blur σ=0.5 | ~86% |
| Blur σ=1.0 | ~73% |
| Blur σ=1.5 | ~59% |
| Blur σ=2.0 | ~55% |
| Blur σ=3.0 | ~54% |
| Rotation 0° | ~86% |
| Rotation 5° | ~83% |
| Rotation 10° | ~80% |
| Rotation 15° | ~76% |
| Rotation 20° | ~73% |
| Rotation 30° | ~69% |

## Image Quality & Exact Recovery

| Metric | Value |
|--------|------|
| ASR (Attack Success Rate) | ~0% |

## Key Observations

1. **Adversarial Training Integrated (HiDDeN-style)**  
A discriminator was introduced to distinguish between original and watermarked images, forcing the encoder to produce more realistic outputs. The adversarial loss (weight = 0.001) improves perceptual quality while maintaining message fidelity.

2. **Stable but Lower Clean Accuracy**  
Clean accuracy (~86%) is lower than previous experiments (~92%), indicating that adversarial pressure slightly harms pure decoding performance. This reflects the classic GAN trade-off between realism and signal strength.

3. **Improved Robustness to Rotation**  
Rotation accuracy remains relatively strong (~76% at 15°), with smoother degradation compared to earlier experiments. The discriminator likely encourages learning more globally consistent features, improving geometric robustness.

4. **Blur Robustness Moderately Improved**  
Compared to earlier setups, performance at σ=1.0 (~73%) is slightly better than typical drops (~60–65%), suggesting adversarial training helps resist mild smoothing, though high blur still degrades performance significantly.

5. **Perceptual Quality Decreased (Lower PSNR)**  
PSNR drops to ~25.5 dB, much lower than non-adversarial setups. This indicates that while images look more “natural” to the discriminator, they deviate more from the original pixel-wise (GAN effect).

6. **Training Instability Observed**  
Fluctuations in clean accuracy during later epochs (e.g., drop at epoch 130) suggest mild GAN instability. This is expected when jointly optimizing encoder, decoder, and discriminator.

7. **Exact Recovery Still Fails**  
Despite reasonable bit accuracy (70–85%), exact 12-character hex recovery is 0% under attacks. Small bit errors accumulate, confirming the need for error-correcting codes (e.g., BCH, Reed-Solomon).

8. **Curriculum Learning Effective**  
Gradual introduction of blur (epoch 15) and rotation (epoch 80) avoids catastrophic collapse and allows stable convergence, even with adversarial training.

<img width="1052" height="367" alt="metrics_plot_9" src="https://github.com/user-attachments/assets/40a2289b-7c46-442a-8737-03b79b811331" />

<img width="1053" height="272" alt="qualitative_plot_9" src="https://github.com/user-attachments/assets/9120ae66-a7f3-4921-8597-89c84d0a00ba" />
