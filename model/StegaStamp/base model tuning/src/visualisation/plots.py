def to_pil(t):
    return T.ToPILImage()(((t[0]+1)/2).clamp(0,1).cpu())

raw_residual, scaled_residual = extract_residual(enc, img_tensor, target_tensor, alpha_scale=15)

print("\nAnalyzing bit accuracy vs attack strength...")

blur_sigmas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
bit_accuracies = []

enc.eval()
dec.eval()

for sigma in blur_sigmas:
    if sigma == 0.0:
        attacked = watermarked
    else:
        B, C, H, W = watermarked.shape
        attacked_imgs = []
        for i in range(B):
            pil = T.ToPILImage()(((watermarked[i] + 1) / 2).clamp(0, 1).cpu())
            img_tensor_blur = T.ToTensor()(pil)
            blur = T.GaussianBlur(kernel_size=5, sigma=sigma)
            img_tensor_blur = blur(img_tensor_blur)
            attacked_imgs.append(img_tensor_blur)
        attacked = torch.stack(attacked_imgs).to(watermarked.device) * 2 - 1

    with torch.no_grad():
        logits = dec(attacked)
        pred_bits = (torch.sigmoid(logits) > 0.5).float()
        acc = pred_bits.eq(target_tensor).float().mean().item()
        bit_accuracies.append(acc * 100)

    print(f"   σ={sigma:.1f}: {acc*100:.1f}% accuracy")

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.2)

ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(orig_pil)
ax1.set_title("Original", fontsize=12, fontweight='bold')
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(to_pil(tests['Clean']))
ax2.set_title(f"Watermarked\n'{SECRET_SHORT}'\nPSNR: {psnr_wm:.2f} dB",
             fontsize=11, fontweight='bold')
ax2.axis('off')

ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(to_pil(scaled_residual))
ax3.set_title(f"Residual (15x)\nAlpha: {enc.alpha.item():.3f}",
             fontsize=11, fontweight='bold')
ax3.axis('off')

ax4 = fig.add_subplot(gs[0, 3])
ax4.imshow(to_pil(tests['Blur']))
ax4.set_title(f"After Blur Attack\n'{results['Blur'][2]}'\nAcc: {results['Blur'][0]:.1%}",
             fontsize=11, fontweight='bold')
ax4.axis('off')

ax_graph = fig.add_subplot(gs[1, :3])
ax_graph.plot(blur_sigmas, bit_accuracies, 'o-', linewidth=2, markersize=8, color='#2E86AB')
ax_graph.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random guess (50%)')
ax_graph.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Good (90%)')
ax_graph.grid(True, alpha=0.3)
ax_graph.set_xlabel('Blur Strength (σ)', fontsize=11, fontweight='bold')
ax_graph.set_ylabel('Bit Accuracy (%)', fontsize=11, fontweight='bold')
ax_graph.set_title('Watermark Robustness vs Gaussian Blur Attack', fontsize=12, fontweight='bold')
ax_graph.set_ylim([0, 105])
ax_graph.legend(loc='best')

for sigma, acc in zip(blur_sigmas, bit_accuracies):
    ax_graph.annotate(f'{acc:.1f}%',
                     xy=(sigma, acc),
                     xytext=(0, 10),
                     textcoords='offset points',
                     ha='center',
                     fontsize=8,
                     alpha=0.7)

ax_text = fig.add_subplot(gs[1, 3])
ax_text.axis('off')

stats_text = f"""RESIDUAL STATISTICS

Raw Values:
  Mean: {raw_residual.mean().item():.6f}
  Std:  {raw_residual.std().item():.6f}
  Min:  {raw_residual.min().item():.3f}
  Max:  {raw_residual.max().item():.3f}

Alpha: {enc.alpha.item():.4f}
After scaling (α×tanh):
  α = {torch.tanh(enc.alpha).item():.4f}

Image Impact:
  Mean change: {(watermarked - img_tensor).abs().mean().item():.4f}
  Max change:  {(watermarked - img_tensor).abs().max().item():.4f}

Pixels changed:
  >1%:  {((watermarked - img_tensor).abs() > 0.01).float().mean().item()*100:.1f}%
  >10%: {((watermarked - img_tensor).abs() > 0.1).float().mean().item()*100:.1f}%
"""

ax_text.text(0.05, 0.95, stats_text,
            transform=ax_text.transAxes,
            fontsize=9,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle("StegaStamp Watermarking Evaluation",
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('watermark_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('watermark_analysis.pdf', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nVisualization saved as: watermark_analysis.png")

print(f"\nResidual Statistics:")
print(f"   - Mean: {raw_residual.mean().item():.6f}")
print(f"   - Std: {raw_residual.std().item():.6f}")
print(f"   - Min: {raw_residual.min().item():.6f}")
print(f"   - Max: {raw_residual.max().item():.6f}")

with torch.no_grad():
    B, _, H, W = img_tensor.shape
    sec_feat = enc.secret_preproc(target_tensor).view(B, 8, 16, 16)
    sec_up = F.interpolate(sec_feat, size=(H//4, W//4), mode='bilinear', align_corners=False)
    img_feat = enc.encoder(img_tensor)
    fused = torch.cat([img_feat, sec_up], dim=1)
    fused = enc.fusion(fused)
    refined = fused + enc.refine(fused)

    residual_raw_decoder = enc.decoder(refined)
    residual_after_tanh = torch.tanh(residual_raw_decoder)
    alpha_val = torch.tanh(enc.alpha).item()
    residual_final = alpha_val * residual_after_tanh

    watermark_diff = (watermarked - img_tensor).abs()

    print(f"\nResidual Pipeline:")
    print(f"   Raw decoder output:")
    print(f"     Mean: {residual_raw_decoder.mean().item():.6f}, Std: {residual_raw_decoder.std().item():.6f}")
    print(f"   After tanh:")
    print(f"     Mean: {residual_after_tanh.mean().item():.6f}, Std: {residual_after_tanh.std().item():.6f}")
    print(f"   After alpha scaling (α={alpha_val:.4f}):")
    print(f"     Mean: {residual_final.mean().item():.6f}, Std: {residual_final.std().item():.6f}")
    print(f"   Actual image change:")
    print(f"     Mean: {watermark_diff.mean().item():.6f}, Max: {watermark_diff.max().item():.6f}")
    print(f"     Pixels changed >0.01: {(watermark_diff > 0.01).float().mean().item()*100:.2f}%")
    print(f"     Pixels changed >0.1: {(watermark_diff > 0.1).float().mean().item()*100:.2f}%")
