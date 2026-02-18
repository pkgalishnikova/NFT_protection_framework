# Description of the graphs

1. Accuracy vs. beat position + Accuracy boxplot

The graph shows zero variance, both clean and attacked show exactly 52.0%
This happenes, because the model predicts 52 specific bits with 100% accuracy, but it predicts the other 48 bits with 0% accuracy (always wrong). This creates a degenerate solution where accuracy is deterministic.

Root cause: The decoder learned a trivial constant prediction for 48 bits instead of extracting them from the watermark.

<img width="1482" height="1033" alt="accuracy_boxplot-2" src="https://github.com/user-attachments/assets/e72abb29-5916-415c-88fb-0c8fb4bcf5ef" />

2. Accuracy vs. blur
Shows that 52 correctly predicted bits are being predicted independently of the image. The decoder memorized: "bit 1 = always 1, bit 4 = always 0, ..." etc.

<img width="1782" height="1029" alt="accuracy_vs_blur-2" src="https://github.com/user-attachments/assets/d54cbb9c-d5f4-4604-afcf-c993d21652ae" />

3. Accuracy and PSNR vs. blur strength

Accuracy (blue): Flat at ~52% across all blur strengths (σ=0 to σ=4.0), confirming the model is bad at watermark decoding

PSNR (red): Decreases from 100dB (no blur) to ~26dB (σ=4.0), showing image quality degrades with stronger blur

PSNR degrades normally, but accuracy stays almost at random baseline regardless of attack strength.

<img width="1779" height="1030" alt="accuracy_psnr_vs_blur-2" src="https://github.com/user-attachments/assets/01bebde6-3eda-46ad-a98e-72007994b7f6" />

4. Bit position recovery

It shows perfect 100%/0% split for most bits. The decoder found it easier to memorize the bit pattern than extract from images. The watermark signal embedded by encoder is too subtle.

<img width="2082" height="1030" alt="accuracy_vs_bit_position" src="https://github.com/user-attachments/assets/00439a13-24e6-4aed-9b63-c71b054ea04f" />

5. Image quality metrics

The encoder works perfectly, it embeds a subtle watermark, howvever struggles to decode it.

| Metric | Min | Max | Median | Mean | Std |
|--------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|
| psnr  | 48.50980758666992 | 50.59191131591797 | 48.56714630126953 | 48.60850241088867 | 0.15401925420521684 |
| mse   | 3.4903507184935734e-05 | 5.6374032283201814e-05 | 5.563464401348028e-05 | 5.5139403193606996e-05 | 1.7426648023370696e-06 |
| ssim  | 0.9650952219963074 | 0.9999462962150574 | 0.9996871948242188 | 0.9985095262527466 | 0.003372018923982978 |
| lpips | 0.00011722467024810612 | 0.003146975301206112 | 0.0005070037441328168 | 0.0005772631998115685 | 0.00035311532598154035 |


<img width="2085" height="1477" alt="image_quality_distributions-2" src="https://github.com/user-attachments/assets/f000a1b1-41f8-451e-a40f-6ce0d5dddbe2" />

6. Loss scatter (Image loss vs secret loss)

Shows negative correlation (-0.2474) between image loss and secret loss. Points cluster tightly, suggesting the training found a consistent trade-off. When the model tries harder to preserve image quality (lower image loss), it performs worse at embedding the secret (higher secret loss), and vice versa.

<img width="1782" height="1480" alt="loss_scatter" src="https://github.com/user-attachments/assets/99653c83-e63b-4d31-9dab-ce6fed331702" />

7. Perfect recovery vs blur

<img width="1782" height="1030" alt="perfect_recovery_vs_blur" src="https://github.com/user-attachments/assets/a8153490-620a-4889-9f18-7f8d47107f68" />

8. ROC-AUC curve

AUC = 0.4907 (below 0.5, which is worse than random). The curve barely rises above the diagonal random classifier line, and optimal threshold is 0.501. Conclusion: The classifier has no discriminative ability - it cannot distinguish between watermarked and non-watermarked bits.

<img width="1482" height="1480" alt="roc_curve-2" src="https://github.com/user-attachments/assets/ecc6bdb7-61f1-4a3d-b09f-97b774593902" />


Добавить:

1. Разные атаки и их результаты
2. Image Quality vs Decoding Accuracy Scatter

