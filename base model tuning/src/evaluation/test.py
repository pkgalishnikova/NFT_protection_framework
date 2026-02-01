print("\n" + "="*70)
print("TESTING")
print("="*70)

print("\n📤 Using demo image...")
img_path = "/content/0.jpg"

orig_pil = Image.open(img_path).convert("RGB")
test_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
img_tensor = test_transform(orig_pil).unsqueeze(0).to(DEVICE)

target_tensor = target_secret.unsqueeze(0).to(DEVICE)

enc.eval()
dec.eval()

with torch.no_grad():
    watermarked = enc(img_tensor, target_tensor)

    mse_val = calculate_mse(img_tensor, watermarked)
    psnr_wm = calculate_psnr(img_tensor, watermarked)

    clip_metrics = calculate_clip_metrics(img_tensor, watermarked, SECRET_SHORT)

    asr, ear, recovered_list = calculate_asr_and_ear(dec, watermarked, target_tensor, num_trials=20)

    # Clean
    clean_logits = dec(watermarked)
    clean_pred = (torch.sigmoid(clean_logits) > 0.5).float()
    clean_acc = clean_pred.eq(target_tensor).float().mean().item()
    clean_recovered = bits_to_ethereum(clean_pred[0], MESSAGE_LEN)

    # Attacked
    attacked_img = simple_attack(watermarked)
    attacked_logits = dec(attacked_img)
    attacked_pred = (torch.sigmoid(attacked_logits) > 0.5).float()
    attacked_acc = attacked_pred.eq(target_tensor).float().mean().item()
    attacked_recovered = bits_to_ethereum(attacked_pred[0], MESSAGE_LEN)

tests = {
    'Clean': watermarked,
    'Blur': simple_attack(watermarked),
}

print(f"\n🔐 Target Secret: '{SECRET_SHORT}'")
print(f"📊 Watermark Quality:")
print(f"   - PSNR: {psnr_wm:.2f} dB")
print(f"   - MSE: {mse_val:.12f}")
print(f"\n🎨 CLIP-Based Utility Metrics:")
print(f"   - CLIPimg (image similarity): {clip_metrics['CLIPimg']:.4f}")
print(f"   - CLIPout (text-image alignment): {clip_metrics['CLIPout']:.4f}")
print(f"   - CLIPdir (direction similarity): {clip_metrics['CLIPdir']:.4f}")
print(f"\n🎯 Specificity Metrics:")
print(f"   - Clean Accuracy: {clean_acc:.1%}")
print(f"   - Attack Success Rate (ASR): {asr:.1%}")
print(f"   - Error Attack Rate (EAR): {ear:.1%}")
print(f"   - Sample Recovered: '{clean_recovered}'")

# Show attack robustness
print(f"\n🔍 Attack Robustness (20 trials):")
unique_recovered = list(set(recovered_list))
for i, addr in enumerate(unique_recovered[:5]):
    count = recovered_list.count(addr)
    status = "✅ TARGET" if addr == SECRET_SHORT else "❌ WRONG"
    print(f"   {addr} ({count}/20) {status}")

results = {
    'Clean': (clean_acc, calculate_tpr(clean_logits.sigmoid(), target_tensor), clean_recovered),
    'Blur': (attacked_acc, calculate_tpr(attacked_logits.sigmoid(), target_tensor), attacked_recovered)
}
