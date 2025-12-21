## Описание созданных моделей StegaStamp
1. StegaStamp_var0.1
   
   **Model:** Custom PyTorch reimplementation of StegaStamp, consisting of a convolutional encoder that embeds a 100-bit secret as a residual into a 256×256 image, and a convolutional decoder that predicts the secret logits from a possibly distorted image.
   
   **Training:** End-to-end training using synthetic distortions that mimic the print-and-scan pipeline—specifically: JPEG compression (60–95 quality), random resizing + cropping, additive Gaussian noise, and brightness/contrast adjustments. Loss combines binary cross-entropy on the secret and MSE on image reconstruction (weighted 1:0.01).
   
   **Dataset:** No real dataset used—training images are random Gaussian noise tensors clamped to [–1, 1], i.e., synthetic images. This is a simplification compared to the original paper, which used natural images (e.g., ImageNet).
   
   **Robustness:** Designed to be robust to the simulated print-photo distortions applied during training (JPEG, resize, noise, color shifts), but not tested on real printed/photographed images or other attacks (e.g., blur, rotation, cropping beyond center).
   
2. StegaStamp_var0.2
   **Model:** Custom U-Net-like encoder with downsampling → bottleneck (secret injection) → upsampling via transposed convolutions; decoder is a small CNN with global average pooling. Uses 100-bit secret + Reed–Solomon ECC → 180 bits for robust decoding.

   **Training:** Only 1-step adaptation on the user-uploaded image (not full training). Loss is binary cross-entropy on ECC-encoded bits after simulated distortion.

   **Dataset:** No external dataset—uses a single user-provided image (or a red-square demo if none uploaded). Training is image-specific and minimal (for demo only).

   **Robustness:** Simulates print-and-photo pipeline via JPEG compression (65–90 quality), random resize + center crop, and brightness/contrast jitter during distortion. Relies on ECC (Reed–Solomon) to correct bit errors after distortion.

3. StegaStamp_var0.3

   **Model:** Modified StegaStamp architecture for 416-bit payload (52 bytes): encoder uses dense secret projection → 64-channel feature map injected at ¼ resolution; decoder is a deeper CNN with 512-dim global feature and MLP head to output 416 logits.

   **Training:** Only 3-step warmup on the single user-provided image using random 416-bit messages and photorealistic distortion simulation; not full training—intended for demo/adaptation only.

   **Dataset:** No external dataset—relies entirely on one uploaded image (or a text-based demo if none). Training is per-image and minimal.

   **Robustness:** Targets print-and-photograph pipeline via JPEG compression (65–90), scale+crop, and brightness/contrast jitter. Uses Reed–Solomon ECC (RS(255,223)) to protect a 20-byte Ethereum address, embedding 52 bytes (416 bits) total to allow error correction after distortion.

4. StegaStamp_var0.4

   **Model:** Custom StegaStamp-based architecture adapted for 416-bit secrets (52 bytes = 20-byte Ethereum address + 32-byte Reed–Solomon ECC). Encoder uses dense secret projection → spatial broadcast at ¼ resolution → U-Net-like residual generation. Decoder is a 5-layer CNN with global average pooling and MLP head outputting 416 logits.

   **Training:** Trained for 200 steps on a diverse synthetic dataset of 100 procedurally generated images (gradients, noise, text, blocks, grids, blends) with print-photo distortions (JPEG 65–90, scale/crop, brightness/contrast). Uses fixed Ethereum address as secret (not random bits) and gradient clipping for stability. Best checkpoint saved by bit accuracy.

   **Dataset:** No real-world images—uses a custom on-the-fly synthetic dataset with rich visual variation to simulate natural image statistics without external data.

   **Robustness:** Explicitly targets physical-world robustness via non-differentiable print→photo simulation during training and evaluation. Leverages Reed–Solomon (RS(255,223)) error correction to recover the 20-byte Ethereum address even if up to ~16 bytes are corrupted. Recovery validated with bit accuracy and address string match.

5. StegaStamp_var0.5

   **Model:** Custom StegaStamp-style encoder-decoder adapted for 336-bit Ethereum addresses (42 UTF-8 chars). Encoder uses dense secret projection → spatial broadcast at 1/8 resolution → U-Net residual generation. Decoder is a 4-layer CNN with global average pooling predicting 496-bit ECC-extended logits (336 data + 160 parity bits).

   **Training:** Trained for 600 steps on COCO-100—the first 100 images from COCO train2017, downloaded on-the-fly. Uses random secrets (336 bits) during training with Reed–Solomon ECC (20 parity bytes) and aggressive print-photo distortions (JPEG 50–95, scale 0.7–1.3x, brightness/contrast ±30%).

   **Dataset:** Real-world natural images from MS COCO (Common Objects in Context)—a standard vision dataset with diverse scenes, objects, and textures, ensuring better generalization than synthetic data.

   **Robustness:** Designed for physical-world resilience: during inference, simulates realistic print→photo capture (JPEG 65–90, mild scale/brightness shifts). Reed–Solomon error correction enables recovery of the full 42-char Ethereum address even if up to ~20 bytes are corrupted. Success measured by bit accuracy and exact string match.

6. StegaStamp_var0.6

  **Model:** Enhanced StegaStamp architecture designed to embed 496-bit ECC-protected Ethereum addresses (42 UTF-8 chars + 20-byte Reed–Solomon parity). Encoder uses larger secret projection (64×64×8), deeper bottleneck, and scaled residual (×0.1) for stronger yet imperceptible watermarking. Decoder features a 5-layer CNN with 512-dim features, dropout, and higher capacity head for robust extraction.

  **Training:** Trained for 2000 steps on COCO-500 (first 500 COCO train2017 images, with synthetic fallback if download fails). Uses a pool of 51 diverse secrets (50 random + target Ethereum) to prevent overfitting. Employs curriculum learning: easy → medium → hard distortions (JPEG 75–40, scale 0.9–1.4, brightness/contrast shifts). Separate optimizers (decoder lr = 2× encoder) and gradient clipping.

  **Dataset:** Primarily real-world natural images from MS COCO, ensuring rich texture and semantic diversity. Falls back to 200 synthetic solid-color images only if COCO download fails (unlikely in most environments).

  **Robustness:** Explicitly evaluated under three attack levels (clean, easy, medium). Leverages Reed–Solomon ECC (RS(62,42)) to correct byte-level errors. Achieves high bit accuracy and exact address recovery even after JPEG compression, resizing, and color distortion simulating print→photo pipeline. Performance validated via ECC success flag and string match.

7. StegaStamp_var0.7

  **Model:** Simple, lightweight StegaStamp variant designed for 100-bit messages (~12 ASCII characters). Encoder uses dense secret expansion → spatial broadcast at ¼ resolution → shallow U-Net residual generation. Decoder is a 4-layer CNN with global average pooling and dropout—optimized for fast convergence and robustness with limited data.

  **Training:** Trained for 3000 steps on a hybrid dataset: uses COCO images if available, otherwise 200 procedurally generated synthetic images with random colors and rectangles. Employs a pool of 31 diverse 12-character secrets (30 random + target Ethereum prefix) to avoid overfitting. Uses JPEG compression (70–90) as the sole distortion during training.

  **Dataset:** Adaptive—prefers real COCO images for realism, falls back to structured synthetic data to ensure training feasibility in restricted environments (e.g., no internet). All images are 256×256.

  **Robustness:** Focused on practical, achievable robustness for short payloads. Evaluated using PSNR (>30 dB = imperceptible) and True Positive Rate (TPR). Successfully recovers the Ethereum address prefix ("0xBC4CA0EdA7") with >90% bit accuracy even after JPEG compression, demonstrating feasibility for real-world invisible tagging.

# To 25.12:
   
8. StegaStamp_var0.8 -> 58.13%
   
   **Model:** Custom PyTorch StegaStamp with a U-Net-like encoder embedding a 160-bit Ethereum address into a 128×128 RGB image (via residual +        sigmoid), and a CNN decoder predicting secret logits.

   **Training:** End-to-end with JPEG (Q=50) + Gaussian blur applied during training; loss = MSE (image) + 15×BCE (secret).

   **Dataset:** Primarily COCO val2017 (128×128); falls back to synthetic JPEG images if download fails.

   **Robustness:** Evaluated against same JPEG+blur pipeline; 58.3% bit accuracy achieved — below 70% success threshold. Not tested on cropping,       rotation, or real print-scan.
   
9. StegaStamp_var0.9 -> 51.88%

   **Model:** Same StegaStamp encoder/decoder for 160-bit Ethereum addresses in 128×128 RGB images.

   **Training:** Extended to 20 epochs, 1500 COCO images, and mixed-precision (FP16) training; loss reweighted to 0.5×MSE + 20×BCE to prioritize       secret recovery over visual fidelity.

   **Dataset:** Larger COCO subset (1.5k vs 800)

   **Robustness:** Same JPEG (Q=50) + Gaussian blur evaluation; 51.88% bit accuracy — lower than prior run, likely due to stronger secret-loss         weighting increasing distortion sensitivity or overfitting.

10. StegaStamp_var0.10 -> 57%

    **Model:** Lightweight encoder-decoder pair — encoder injects a 100-bit truncated Ethereum address as a residual into 256×256 images               (normalized to [–1,1]); decoder uses 4× downsampling + global pooling.

    **Training:** Trained for 3k steps on 200 synthetic or 500 COCO images, using mixed secret pool (30 random + 1 target address); loss = BCE +       0.5×MSE.

    **Attacks:** Evaluated under JPEG (Q=50) + Gaussian blur (σ=1.0) — same as training.

    **Robustness:** Achieves 51.88% bit accuracy on the target address after attacks — below 70% reliability threshold; clean accuracy is higher,      indicating vulnerability to distortions.

11. StegaStamp_var0.11 -> 
