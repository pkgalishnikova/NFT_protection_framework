enc = ImprovedEncoder(MESSAGE_LEN).to(DEVICE)
dec = ImprovedDecoder(MESSAGE_LEN).to(DEVICE)

optimizer = torch.optim.Adam(
    list(enc.parameters()) + list(dec.parameters()),
    lr=1e-4,
    betas=(0.9, 0.999)
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000, eta_min=1e-5)

target_secret = ethereum_to_bits(SECRET_SHORT, MESSAGE_LEN)

train_secrets = []
train_secrets.append(target_secret)

for i in range(50):
    rand_hex = '0x' + ''.join([format(random.randint(0, 15), 'X') for _ in range(MESSAGE_LEN // 4)])
    train_secrets.append(ethereum_to_bits(rand_hex, MESSAGE_LEN))

print(f"Secret pool: {len(train_secrets)} diverse Ethereum addresses")

all_bits = torch.stack(train_secrets).float()
ones_ratio = all_bits.mean().item()
print(f"Bit balance in pool: {ones_ratio:.1%} ones, {1-ones_ratio:.1%} zeros")

enc.train()
dec.train()
training_mode = True

step = 0
max_steps = 3000
best_target_acc = 0

while step < max_steps:
    for images in loader:
        if step >= max_steps:
            break

        images = images.to(DEVICE)
        B = images.size(0)

        secrets = torch.stack([
            train_secrets[random.randint(0, len(train_secrets)-1)]
            for _ in range(B)
        ]).to(DEVICE)

        watermarked = enc(images, secrets)

        attacked = differentiable_blur(watermarked)

        logits = dec(attacked)

        secret_loss = F.binary_cross_entropy_with_logits(logits, secrets)
        image_loss = F.mse_loss(watermarked, images)

        loss = 2.0 * secret_loss + 0.3 * image_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            # Check encoder gradients
            enc_grads = []
            for name, param in enc.named_parameters():
                if param.grad is not None:
                    enc_grads.append(param.grad.abs().mean().item())

            dec_grads = []
            for name, param in dec.named_parameters():
                if param.grad is not None:
                    dec_grads.append(param.grad.abs().mean().item())

            enc_grad_mean = np.mean(enc_grads) if enc_grads else 0.0
            dec_grad_mean = np.mean(dec_grads) if dec_grads else 0.0

            print(f"\nGradient Check at Step {step}:")
            print(f"   Encoder avg grad: {enc_grad_mean:.6f}")
            print(f"   Decoder avg grad: {dec_grad_mean:.6f}")
            print(f"   Ratio (enc/dec): {enc_grad_mean/(dec_grad_mean+1e-10):.4f}")

            if enc.decoder[0].weight.grad is not None:
                print(f"   Encoder decoder[0] grad: {enc.decoder[0].weight.grad.abs().mean().item():.6f}")
            if enc.alpha.grad is not None:
                print(f"   Alpha grad: {enc.alpha.grad.item():.6f}")

        if step % 100 == 0:
            with torch.no_grad():
                training_mode = False  # Use fixed sigma for eval
                target_batch = target_secret.unsqueeze(0).repeat(2, 1).to(DEVICE)
                test_wm = enc(images[:2], target_batch)
                test_att = differentiable_blur(test_wm)
                test_logits = dec(test_att)

                pred_bits = (test_logits.sigmoid() > 0.5).float()
                target_acc = pred_bits.eq(target_batch).float().mean().item()

                psnr = calculate_psnr(images[:2], test_wm)
                tpr = calculate_tpr(test_logits.sigmoid(), target_batch)

                recovered = bits_to_ethereum(pred_bits[0], MESSAGE_LEN)

                train_pred = (logits.sigmoid() > 0.5).float()
                train_acc = train_pred.eq(secrets).float().mean().item()

                lr = optimizer.param_groups[0]['lr']

                print(f"Step {step:4d} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Train Acc: {train_acc:5.1%} | "
                      f"Target Acc: {target_acc:5.1%} | "
                      f"PSNR: {psnr:5.2f}dB | "
                      f"TPR: {tpr:5.1%} | "
                      f"LR: {lr:.2e} | "
                      f"Recovered: '{recovered}'")

                if target_acc > best_target_acc:
                    best_target_acc = target_acc
                    if target_acc > 0.70 and step > 500:
                        print(f"New best: {best_target_acc:.1%}")

                training_mode = True

        step += 1

training_mode = False
print(f"   Best target accuracy: {best_target_acc:.1%}")
