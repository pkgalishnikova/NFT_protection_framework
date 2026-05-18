# experiment 3 - heavier message weight optimizer

import os, sys, random, warnings, math
warnings.filterwarnings("ignore")

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

import kornia
import kornia.geometry.transform as KGT
import kornia.filters as KF

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

SECRET_STR = "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D"
SECRET_SHORT = SECRET_STR[:12]
MESSAGE_LEN = 48
IMG_SIZE = 128

print(f"Hide 12 symbols: '{SECRET_SHORT}' ({MESSAGE_LEN} bits)")


# METRICS

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse < 1e-10:
        return 100.0
    return (20 * torch.log10(torch.tensor(2.0) / torch.sqrt(mse))).item()


# ARCHITECTURE

class ConvBNRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)

class Encoder(nn.Module):
    def __init__(self, message_len=48, channels=64, num_blocks=4):
        super().__init__()
        layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))
        self.image_layers = nn.Sequential(*layers)

        self.after_concat = ConvBNRelu(channels + 3 + message_len, channels)

        self.final = nn.Conv2d(channels, 3, kernel_size=1)

    def forward(self, image, message):
        expanded = message.unsqueeze(-1).unsqueeze(-1)
        expanded = expanded.expand(-1, -1, image.shape[2], image.shape[3])

        features = self.image_layers(image)

        concat = torch.cat([features, image, expanded], dim=1)

        out = self.after_concat(concat)
        out = self.final(out)
        return out

class Decoder(nn.Module):
    def __init__(self, message_len=48, channels=64, num_blocks=7):
        super().__init__()
        layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(ConvBNRelu(channels, message_len))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.conv_layers = nn.Sequential(*layers)

        self.linear = nn.Linear(message_len, message_len)

    def forward(self, image):
        x = self.conv_layers(image)
        x = x.squeeze(3).squeeze(2)
        x = self.linear(x)
        return x

# DIFF. ATTACKS

def diff_blur(img, sigma=None):
    if sigma is None:
        sigma = random.uniform(0.5, 2.0)
    ks = max(3, 2 * int(3 * sigma) + 1)
    if ks % 2 == 0: ks += 1
    return KF.gaussian_blur2d(img, (ks, ks), (sigma, sigma))

def diff_rotate(img, angle=None):
    if angle is None:
        angle = random.uniform(-30, 30)
    B = img.shape[0]
    angle_t = torch.tensor([angle]*B, device=img.device, dtype=img.dtype)
    return KGT.rotate(img, angle_t)

def apply_attack(img, attack_type='none'):
    if attack_type == 'blur':
        return diff_blur(img)
    elif attack_type == 'rotation':
        return diff_rotate(img)
    elif attack_type == 'combined':
        return diff_rotate(diff_blur(img))
    return img


# HEX <-> BITS

def ethereum_to_bits(address, num_bits=48):
    if address.startswith('0x') or address.startswith('0X'):
        address = address[2:]
    num_hex_chars = num_bits // 4
    address_part = address[:num_hex_chars].upper()
    binary_str = bin(int(address_part, 16))[2:].zfill(num_bits)
    return torch.tensor([int(b) for b in binary_str], dtype=torch.float32)

def bits_to_ethereum(bits, num_bits=48, original_secret=SECRET_SHORT):
    bits_np = (bits[:num_bits] > 0.5).cpu().numpy().astype(np.uint8)
    binary_str = ''.join([str(int(b)) for b in bits_np])
    try:
        hex_value = hex(int(binary_str, 2))[2:].upper()
        num_hex_chars = num_bits // 4
        hex_value = hex_value.zfill(num_hex_chars)
        return '0x' + hex_value[:len(original_secret)-2]
    except:
        return "0x" + "?"*(len(original_secret)-2)


# DATASET

print("\nLoading data from COCO (val2017)...")
coco_path = "val2017"
if os.path.exists(coco_path):
    import glob
    img_paths = glob.glob(os.path.join(coco_path, "*.jpg"))[:5000]
    print(f"Found {len(img_paths)} iamges")
else:
    print("Data not found")
    os.makedirs("synthetic", exist_ok=True)
    img_paths = []
    for i in range(500):
        img = Image.new('RGB', (128, 128), (random.randint(50,255), random.randint(50,255), random.randint(50,255)))
        draw = ImageDraw.Draw(img)
        for _ in range(5):
            x1, y1 = random.randint(0,80), random.randint(0,80)
            draw.rectangle([x1, y1, x1+random.randint(20,60), y1+random.randint(20,60)],
                           fill=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        path = f"synthetic/img_{i:04d}.jpg"
        img.save(path)
        img_paths.append(path)

class SimpleDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img)
        except:
            return torch.randn(3, IMG_SIZE, IMG_SIZE)

split = int(len(img_paths) * 0.8)
train_paths = img_paths[:split]
test_paths = img_paths[split:]

train_loader = DataLoader(SimpleDataset(train_paths), batch_size=32, shuffle=True, num_workers=2, drop_last=True)
test_loader = DataLoader(SimpleDataset(test_paths), batch_size=16, shuffle=False)
print(f"Train: {len(train_paths)}, Test: {len(test_paths)} images. Batch: 32.")


# TRAINING

print("\n" + "="*70)
print("Start training...")
print("="*70)

enc = Encoder(MESSAGE_LEN, channels=64, num_blocks=6).to(DEVICE)
dec = Decoder(MESSAGE_LEN, channels=64, num_blocks=8).to(DEVICE)


optimizer = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=2e-3, weight_decay=1e-4)

steps_per_epoch = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, epochs=140, steps_per_epoch=steps_per_epoch)

target_secret = ethereum_to_bits(SECRET_SHORT, MESSAGE_LEN)

NUM_EPOCHS = 80
best_acc = 0

print(f"Training: {NUM_EPOCHS} epochs")

for epoch in range(NUM_EPOCHS):
    enc.train()
    dec.train()

    epoch_loss = 0
    epoch_acc = 0
    batches = 0

    for images in train_loader:
        images = images.to(DEVICE)
        B = images.size(0)

        messages = torch.randint(0, 2, (B, MESSAGE_LEN)).float().to(DEVICE)
        encoded_images = enc(images, messages)

        if epoch < 15:
            noised = encoded_images
        else:
            blur_prob = min(0.6, (epoch - 15) / 100)
            rot_prob = min(0.4, max(0, (epoch - 80) / 100))

            r = random.random()
            if r < blur_prob:
                max_sigma = min(2.0, 0.5 + (epoch - 15) / 60)
                noised = diff_blur(encoded_images, sigma=random.uniform(0.3, max_sigma))
            elif r < blur_prob + rot_prob:
                max_angle = min(25, 5 + (epoch - 80) / 5)
                noised = diff_rotate(encoded_images, angle=random.uniform(-max_angle, max_angle))
            else:
                noised = encoded_images

        decoded = dec(noised)

        loss_msg = F.mse_loss(decoded, messages)
        loss_img = F.mse_loss(encoded_images, images)

        loss = 5.0 * loss_msg + 0.3 * loss_img

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            pred_bits = (decoded > 0.5).float()
            acc = pred_bits.eq(messages).float().mean().item()
            epoch_acc += acc
            epoch_loss += loss.item()
            batches += 1

    avg_acc = epoch_acc / batches
    avg_loss = epoch_loss / batches

    if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
        enc.eval()
        dec.eval()
        all_clean, all_rot = [], []
        psnr_sum, psnr_cnt = 0, 0
        with torch.no_grad():
            for i, test_imgs in enumerate(test_loader):
                if i >= 5: break
                test_imgs = test_imgs.to(DEVICE)
                B_test = test_imgs.size(0)
                sec_batch = target_secret.unsqueeze(0).repeat(B_test, 1).to(DEVICE)
                wm = enc(test_imgs, sec_batch)
                psnr_sum += calculate_psnr(test_imgs, wm)
                psnr_cnt += 1

                dec_clean = dec(wm)
                all_clean.extend((dec_clean > 0.5).float().eq(sec_batch).float().mean(dim=1).tolist())

                wm_rot = diff_rotate(wm, 15)
                dec_rot = dec(wm_rot)
                all_rot.extend((dec_rot > 0.5).float().eq(sec_batch).float().mean(dim=1).tolist())

            clean_acc = np.mean(all_clean)
            rot_acc = np.mean(all_rot)
            psnr = psnr_sum / max(psnr_cnt, 1)

            sample_imgs = next(iter(test_loader)).to(DEVICE)
            sample_wm = enc(sample_imgs[:1], target_secret.unsqueeze(0).to(DEVICE))
            recovered = bits_to_ethereum((dec(sample_wm)[0] > 0.5).float(), MESSAGE_LEN)

        atk_info = ""
        if epoch >= 80: atk_info = " +rot"
        elif epoch >= 15: atk_info = " +blur"
        print(f"Ep. {epoch:3d} | Loss: {avg_loss:.4f} | Train: {avg_acc:.1%} | "
              f"Clean: {clean_acc:.1%} | Rot15: {rot_acc:.1%} | PSNR: {psnr:.1f}{atk_info} | {recovered}")

        if clean_acc > best_acc:
            best_acc = clean_acc
            torch.save({'enc': enc.state_dict(), 'dec': dec.state_dict()}, 'best_model.pth')

print(f"\nTraining finished. Best clean accuracy: {best_acc:.1%}")


# TESTING

enc.eval()
dec.eval()

ckpt = torch.load('best_model.pth')
enc.load_state_dict(ckpt['enc'])
dec.load_state_dict(ckpt['dec'])

blur_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
rot_levels = [0, 5, 10, 15, 20, 30]

print("\nTesting...")

blur_accs = {l: [] for l in blur_levels}
rot_accs = {l: [] for l in rot_levels}
exact_blur1 = 0
exact_rot15 = 0
total = 0

with torch.no_grad():
    for imgs in test_loader:
        imgs = imgs.to(DEVICE)
        B = imgs.size(0)
        sec = target_secret.unsqueeze(0).repeat(B, 1).to(DEVICE)
        wm = enc(imgs, sec)

        for lvl in blur_levels:
            att = diff_blur(wm, sigma=lvl) if lvl > 0 else wm
            pred = (dec(att) > 0.5).float()
            acc = pred.eq(sec).float().mean(dim=1).tolist()
            blur_accs[lvl].extend(acc)
            if lvl == 1.0:
                for i in range(B):
                    if bits_to_ethereum(pred[i], MESSAGE_LEN) == SECRET_SHORT:
                        exact_blur1 += 1

        for lvl in rot_levels:
            att = diff_rotate(wm, lvl) if lvl > 0 else wm
            pred = (dec(att) > 0.5).float()
            acc = pred.eq(sec).float().mean(dim=1).tolist()
            rot_accs[lvl].extend(acc)
            if lvl == 15:
                for i in range(B):
                    if bits_to_ethereum(pred[i], MESSAGE_LEN) == SECRET_SHORT:
                        exact_rot15 += 1

        total += B

print(f"\nResults on {total} test images:")
print(f"   Exact recovery with Blur (sigma=1.0): {exact_blur1}/{total}")
print(f"   Exact recovery with Rotation (15°):   {exact_rot15}/{total}")

for lvl in blur_levels:
    print(f"   Blur σ={lvl:.1f}: {np.mean(blur_accs[lvl])*100:.1f}%")
for lvl in rot_levels:
    print(f"   Rot {lvl}°: {np.mean(rot_accs[lvl])*100:.1f}%")


fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(blur_levels, [np.mean(blur_accs[l])*100 for l in blur_levels], 'o-', lw=2)
axs[0].set_title('Accuracy vs Blur', fontweight='bold')
axs[0].set_xlabel('Sigma'); axs[0].set_ylabel('Bit Accuracy (%)')
axs[0].grid(True, alpha=0.3); axs[0].set_ylim(0, 105)

axs[1].plot(rot_levels, [np.mean(rot_accs[l])*100 for l in rot_levels], 's-', color='r', lw=2)
axs[1].set_title('Accuracy vs Rotation', fontweight='bold')
axs[1].set_xlabel('Angle')
axs[1].grid(True, alpha=0.3); axs[1].set_ylim(0, 105)

plt.tight_layout()
plt.savefig('metrics_plot.png', dpi=150)
plt.show()

with torch.no_grad():
    sample = next(iter(test_loader)).to(DEVICE)
    sec = target_secret.unsqueeze(0).to(DEVICE)
    img_t = sample[0:1]
    wm_t = enc(img_t, sec)
    att_blur = diff_blur(wm_t, 1.0)
    att_rot = diff_rotate(wm_t, 15)

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
def show(ax, t, title):
    ax.imshow(((t[0].detach().cpu() + 1)/2).clamp(0,1).permute(1,2,0).numpy())
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.axis('off')

show(axs[0], img_t, "Original")
show(axs[1], wm_t, "Watermarked")
show(axs[2], att_blur, "After blur (σ=1)")
show(axs[3], att_rot, "After rotation (15°)")
plt.tight_layout()
plt.savefig('qualitative_plot.png', dpi=150)
plt.show()
print("qualitative_plot.png saved")
