import os, sys, random, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ Device: {DEVICE}")

# ==============================
# CONFIG
# ==============================
SECRET_STR = "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D"
print(f"🔐 Full address: {SECRET_STR}")

SECRET_SHORT = SECRET_STR[:12]  # "0xBC4CA0EdA7"
MESSAGE_LEN = 100  # bits

print(f"📊 Using SHORT version: '{SECRET_SHORT}' ({MESSAGE_LEN} bits)")

# ==============================
# METRICS FUNCTIONS
# ==============================

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images (range [-1, 1])"""
    mse = F.mse_loss(img1, img2)
    if mse < 1e-10:
        return 100.0
    max_pixel = 2.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


def calculate_tpr(predictions, targets, threshold=0.5):
    """Calculate True Positive Rate"""
    pred_bits = (predictions > threshold).float()
    target_bits = targets.float()
    tp = ((pred_bits == 1) & (target_bits == 1)).float().sum()
    fn = ((pred_bits == 0) & (target_bits == 1)).float().sum()
    tpr = tp / (tp + fn + 1e-10)
    return tpr.item()


# ==============================
# ARCHITECTURE
# ==============================

class SimpleEncoder(nn.Module):
    def __init__(self, secret_len=100):
        super().__init__()
        self.secret_len = secret_len
        
        self.secret_fc = nn.Sequential(
            nn.Linear(secret_len, 4096),
            nn.ReLU()
        )
        
        self.img_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(128 + 16, 128, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(),
        )
        
        self.img_dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, img, secret):
        B, _, H, W = img.shape
        sec = self.secret_fc(secret).view(B, 16, 16, 16)
        sec = F.interpolate(sec, size=(H//4, W//4), mode='bilinear', align_corners=False)
        feat = self.img_conv(img)
        fused = self.fusion(torch.cat([feat, sec], dim=1))
        residual = torch.tanh(self.img_dec(fused))
        return torch.clamp(img + 0.15 * residual, -1, 1)


class SimpleDecoder(nn.Module):
    def __init__(self, secret_len=100):
        super().__init__()
        self.secret_len = secret_len
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, secret_len)
        )

    def forward(self, img):
        feat = self.conv(img).flatten(1)
        return self.fc(feat)


# ==============================
# FIXED UTILS
# ==============================

def ethereum_to_bits(address, num_bits=100):
    """Convert Ethereum address to binary (hex-based)"""
    if address.startswith('0x') or address.startswith('0X'):
        address = address[2:]
    
    num_hex_chars = num_bits // 4
    address_part = address[:num_hex_chars].upper()
    
    binary_str = bin(int(address_part, 16))[2:].zfill(num_bits)
    bits = torch.tensor([int(b) for b in binary_str], dtype=torch.float32)
    
    return bits


def bits_to_ethereum(bits, num_bits=100):
    """Convert binary back to Ethereum address format"""
    bits_np = (bits[:num_bits] > 0.5).cpu().numpy().astype(np.uint8)
    binary_str = ''.join([str(int(b)) for b in bits_np])
    
    try:
        hex_value = hex(int(binary_str, 2))[2:].upper()
        num_hex_chars = num_bits // 4
        hex_value = hex_value.zfill(num_hex_chars)
        return '0x' + hex_value
    except:
        return "0x" + "?"*(num_bits // 4)


def simple_attack(img):
    """JPEG compression (Q=50) + Gaussian blur (σ=1.0)"""
    from io import BytesIO
    B, C, H, W = img.shape
    imgs_out = []

    for i in range(B):
        pil = T.ToPILImage()(((img[i] + 1) / 2).clamp(0, 1).cpu())
        
        # JPEG compression
        buf = BytesIO()
        pil.save(buf, "JPEG", quality=50)
        buf.seek(0)
        pil = Image.open(buf).convert("RGB")
        
        img_tensor = T.ToTensor()(pil)
        
        # Gaussian blur
        blur = T.GaussianBlur(kernel_size=5, sigma=1.0)
        img_tensor = blur(img_tensor)
        
        imgs_out.append(img_tensor)

    imgs_out = torch.stack(imgs_out).to(img.device)
    return imgs_out * 2 - 1


# ==============================
# DATASET
# ==============================

print("\n📥 Preparing dataset...")

coco_path = "./coco/train2017"
if os.path.exists(coco_path):
    import glob
    img_paths = glob.glob(os.path.join(coco_path, "*.jpg"))[:500]
    print(f"✅ Found {len(img_paths)} COCO images")
else:
    print("⚠️ COCO not found. Creating 200 synthetic images...")
    os.makedirs("synthetic", exist_ok=True)
    img_paths = []
    
    for i in range(200):
        img = Image.new('RGB', (256, 256),
                       (random.randint(50, 255),
                        random.randint(50, 255),
                        random.randint(50, 255)))
        
        draw = ImageDraw.Draw(img)
        for _ in range(5):
            x1, y1 = random.randint(0, 200), random.randint(0, 200)
            x2, y2 = x1 + random.randint(20, 80), y1 + random.randint(20, 80)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.rectangle([x1, y1, x2, y2], fill=color)
        
        path = f"synthetic/img_{i:04d}.jpg"
        img.save(path)
        img_paths.append(path)
    
    print(f"✅ Created {len(img_paths)} synthetic images")


class SimpleDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.transform = T.Compose([
            T.Resize(256),
            T.RandomCrop(256),
            T.RandomHorizontalFlip(),
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
            return torch.randn(3, 256, 256)


dataset = SimpleDataset(img_paths)
loader = DataLoader(dataset, batch_size=16, shuffle=True,
                   num_workers=2, pin_memory=True, drop_last=True)

print(f"✅ Dataset ready: {len(dataset)} images, batch size 16")

# ==============================
# TRAINING
# ==============================

print("\n" + "="*70)
print("TRAINING")
print("="*70)

enc = SimpleEncoder(MESSAGE_LEN).to(DEVICE)
dec = SimpleDecoder(MESSAGE_LEN).to(DEVICE)

optimizer = torch.optim.Adam(
    list(enc.parameters()) + list(dec.parameters()),
    lr=2e-4
)

# Target secret (FIXED)
target_secret = ethereum_to_bits(SECRET_SHORT, MESSAGE_LEN)

# Create diverse training secrets (FIXED)
print("\n🎲 Creating training secret pool...")
train_secrets = []
for i in range(30):
    rand_hex = '0x' + ''.join([format(random.randint(0, 15), 'X') for _ in range(MESSAGE_LEN // 4)])
    train_secrets.append(ethereum_to_bits(rand_hex, MESSAGE_LEN))

train_secrets.append(target_secret)
print(f"✅ Secret pool: {len(train_secrets)} diverse Ethereum addresses")

print("\n🚀 Starting training...")
print(f"   Target: '{SECRET_SHORT}'")
print(f"   Expected time: ~20-30 minutes")
print(f"   Goal: >70% accuracy\n")

enc.train()
dec.train()

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
        attacked = simple_attack(watermarked)
        logits = dec(attacked)

        secret_loss = F.binary_cross_entropy_with_logits(logits, secrets)
        image_loss = F.mse_loss(watermarked, images)
        loss = secret_loss + 0.5 * image_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            with torch.no_grad():
                target_batch = target_secret.unsqueeze(0).repeat(2, 1).to(DEVICE)
                test_wm = enc(images[:2], target_batch)
                test_att = simple_attack(test_wm)
                test_logits = dec(test_att)

                pred_bits = (test_logits.sigmoid() > 0.5).float()
                target_acc = pred_bits.eq(target_batch).float().mean().item()

                psnr = calculate_psnr(images[:2], test_wm)
                tpr = calculate_tpr(test_logits.sigmoid(), target_batch)

                recovered = bits_to_ethereum(pred_bits[0], MESSAGE_LEN)

                train_pred = (logits.sigmoid() > 0.5).float()
                train_acc = train_pred.eq(secrets).float().mean().item()

                print(f"Step {step:4d} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Train Acc: {train_acc:5.1%} | "
                      f"Target Acc: {target_acc:5.1%} | "
                      f"PSNR: {psnr:5.2f}dB | "
                      f"TPR: {tpr:5.1%} | "
                      f"Recovered: '{recovered}'")

                if target_acc > best_target_acc:
                    best_target_acc = target_acc
                    if target_acc > 0.70:
                        print(f"  ⭐ New best: {best_target_acc:.1%}")

        step += 1

print(f"\n✅ Training complete!")
print(f"   Best target accuracy: {best_target_acc:.1%}")

# ==============================
# TESTING
# ==============================

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
    psnr_wm = calculate_psnr(img_tensor, watermarked)

    tests = {
        'Clean': watermarked,
        'JPEG+Blur': simple_attack(watermarked),
    }

    results = {}
    for name, test_img in tests.items():
        logits = dec(test_img)
        pred_bits = (logits.sigmoid() > 0.5).float()
        acc = pred_bits.eq(target_tensor).float().mean().item()
        tpr = calculate_tpr(logits.sigmoid(), target_tensor)
        recovered = bits_to_ethereum(pred_bits[0], MESSAGE_LEN)
        results[name] = (acc, tpr, recovered)

print(f"\n🔐 Target: '{SECRET_SHORT}'")
print(f"📊 Watermark PSNR: {psnr_wm:.2f} dB")
print(f"\n📊 Results:")
for name, (acc, tpr, text) in results.items():
    status = "✅" if acc > 0.70 else "❌"
    print(f"  {name:12s}: Acc={acc:5.1%} | TPR={tpr:5.1%} | '{text}' {status}")

# Visualize
def to_pil(t):
    return T.ToPILImage()(((t[0]+1)/2).clamp(0,1).cpu())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(orig_pil)
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(to_pil(watermarked))
axes[1].set_title(f"Watermarked\n'{SECRET_SHORT}'\nPSNR: {psnr_wm:.2f} dB")
axes[1].axis('off')

axes[2].imshow(to_pil(tests['JPEG+Blur']))
axes[2].set_title(f"After JPEG+Blur\nRecovered: '{results['JPEG+Blur'][2]}'\nAcc: {results['JPEG+Blur'][0]:.1%}")
axes[2].axis('off')

plt.suptitle(f"StegaStamp (Best: {best_target_acc:.1%})", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\n✅ DONE!")
print(f"\n📈 Final Metrics:")
print(f"   - Best Accuracy: {best_target_acc:.1%}")
print(f"   - Watermark PSNR: {psnr_wm:.2f} dB")
print(f"   - Clean TPR: {results['Clean'][1]:.1%}")
print(f"   - Attacked TPR: {results['JPEG+Blur'][1]:.1%}")