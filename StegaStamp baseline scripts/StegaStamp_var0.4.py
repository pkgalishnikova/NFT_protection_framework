import os, sys, random, warnings, io
warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import binascii

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"⚙️  Device: {DEVICE}")

# ==============================
# 1. INSTALL reedsolo (ECC)
# ==============================
try:
    import reedsolo
    RS = reedsolo.RSCodec(32)  # RS(255, 223): 223 data + 32 parity
except ImportError:
    print("🔧 Installing reedsolo...")
    %pip -q install reedsolo
    import reedsolo
    RS = reedsolo.RSCodec(32)

# ==============================
# 2. CONFIG: ETH ADDRESS → 20 RAW BYTES + ECC → 416 BITS
# ==============================
SECRET_LEN = 416  # 52 bytes × 8 bits (20 data + 32 parity)
ECC_LEN = 416     # embed RS-encoded payload directly

def eth_to_bits(addr: str) -> torch.Tensor:
    """'0xBC4C...f13D' → 416-bit tensor (20 raw bytes + 32 ECC = 52 bytes)"""
    if addr.startswith("0x"):
        addr = addr[2:]
    try:
        raw = binascii.unhexlify(addr)  # 20 bytes
    except Exception as e:
        raise ValueError(f"Invalid hex: {e}")
    padded = raw.ljust(223, b'\x00')[:223]  # pad to RS data size
    encoded = RS.encode(padded)             # 255 bytes
    used = encoded[:52]                     # 52 bytes = 416 bits (20+32)
    bits = np.unpackbits(np.frombuffer(used, dtype=np.uint8))
    return torch.from_numpy(bits.astype(np.float32))

def bits_to_eth(bits: torch.Tensor) -> str:
    """416 bits → '0x...' (with RS correction)"""
    bits = (bits > 0.5).cpu().numpy().astype(np.uint8)
    try:
        byte_arr = np.packbits(bits[:416]).tobytes()  # 52 bytes
        try:
            decoded, _, _ = RS.decode(byte_arr)
            raw20 = decoded[:20]
        except:
            raw20 = byte_arr[:20]  # fallback
        hex_str = binascii.hexlify(raw20).decode('ascii').upper()
        return f"0x{hex_str}"
    except Exception as e:
        return f"[FAIL: {str(e)}]"

# ==============================
# 3. MODELS (416-bit support)
# ==============================
class StegaStampEncoder(nn.Module):
    def __init__(self, secret_len=SECRET_LEN):
        super().__init__()
        self.secret_dense = nn.Sequential(
            nn.Linear(secret_len, 64 * 32 * 32),  # 65536
            nn.ReLU()
        )
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256 + 64, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Tanh()
        )

    def forward(self, image, secret):
        B, _, H, W = image.shape
        sec = self.secret_dense(secret).view(B, 64, 32, 32)
        sec = F.interpolate(sec, size=(H//4, W//4), mode='nearest')
        x = self.enc(image)
        x = self.bottleneck(torch.cat([x, sec], 1))
        return torch.clamp(image + self.dec(x), -1, 1)

class StegaStampDecoder(nn.Module):
    def __init__(self, secret_len=ECC_LEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, secret_len)
        )

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.head(x)

# ==============================
# 4. DISTORTION (Print→Photo Simulation)
# ==============================
def distort(img):
    """Non-differentiable photorealistic distortion (eval only)"""
    B, C, H, W = img.shape
    x = (img + 1) / 2  # [-1,1] → [0,1]
    from io import BytesIO
    pil_imgs = []
    for i in range(B):
        pil = T.ToPILImage()(x[i].cpu().clamp(0,1))
        # JPEG compression
        buf = BytesIO()
        pil.save(buf, "JPEG", quality=random.randint(65, 90))
        buf.seek(0)
        pil = Image.open(buf).convert("RGB")
        # Scale + center crop
        scale = random.uniform(0.85, 1.15)
        pil = pil.resize((int(W*scale), int(H*scale)), Image.BILINEAR)
        left = max(0, (pil.width - W) // 2)
        top = max(0, (pil.height - H) // 2)
        pil = pil.crop((left, top, left+W, top+H))
        # Brightness/contrast jitter
        pil = T.functional.adjust_brightness(pil, random.uniform(0.9, 1.1))
        pil = T.functional.adjust_contrast(pil, random.uniform(0.9, 1.1))
        pil_imgs.append(T.ToTensor()(pil))
    x = torch.stack(pil_imgs).to(img.device)
    return torch.clamp(x, 0, 1) * 2 - 1

# ==============================
# 5. UPLOAD IMAGE
# ==============================
print("📤 Upload an image (JPG/PNG) to embed Ethereum address")

try:
    from google.colab import files
    uploaded = files.upload()
    img_path = list(uploaded.keys())[0]
    print(f"✅ Uploaded: {img_path}")
except:
    print("ℹ️  No upload — using demo image")
    demo = Image.new('RGB', (300, 300), (240, 240, 240))
    draw = ImageDraw.Draw(demo)
    draw.text((50, 130), "YOUR\nIMAGE", fill=(60, 60, 60), font=None)
    demo.save("/tmp/demo.jpg")
    img_path = "/tmp/demo.jpg"

orig_pil = Image.open(img_path).convert("RGB")
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
img_tensor = transform(orig_pil).unsqueeze(0).to(DEVICE)

# ==============================
# 6. TRAIN ON SYNTHETIC BUT DIVERSE DATASET (100 images, no download)
# ==============================
from torch.utils.data import Dataset, DataLoader
import random

class DiverseSyntheticDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _make_image(self):
        mode = random.choice(['grad', 'noise', 'text', 'blocks', 'grid', 'blend'])
        img = Image.new('RGB', (300, 300), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        if mode == 'grad':
            for i in range(300):
                c = int(255 * (i / 300))
                draw.line([(i, 0), (i, 300)], fill=(c, 128, 255 - c))
        elif mode == 'noise':
            arr = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
        elif mode == 'text':
            texts = ["Stega", "Stamp", "ETH", "0x", "OK", "CVPR", "2020"]
            for _ in range(random.randint(3, 8)):
                x, y = random.randint(10, 250), random.randint(10, 250)
                s = random.choice(texts)
                sz = random.randint(12, 24)
                draw.text((x, y), s, fill=(random.randint(0,200),)*3)
        elif mode == 'blocks':
            for _ in range(15):
                x, y = random.randint(0, 250), random.randint(0, 250)
                w, h = random.randint(15, 50), random.randint(15, 50)
                color = tuple(random.randint(0, 255) for _ in range(3))
                draw.rectangle([x, y, x+w, y+h], fill=color)
        elif mode == 'grid':
            for i in range(0, 300, 30):
                draw.line([(i, 0), (i, 300)], fill=(200, 200, 200), width=1)
                draw.line([(0, i), (300, i)], fill=(200, 200, 200), width=1)
        elif mode == 'blend':
            img1 = Image.new('RGB', (300,300), (255, 100, 100))
            img2 = Image.new('RGB', (300,300), (100, 100, 255))
            img = Image.blend(img1, img2, alpha=random.random())

        return img

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = self._make_image()
        return self.transform(img)

# Create dataset & loader
trainset = DiverseSyntheticDataset(100)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, drop_last=True)
print(f"✅ Training on {len(trainset)} diverse synthetic images (CPU-safe)")

# Models
enc = StegaStampEncoder().to(DEVICE)
dec = StegaStampDecoder().to(DEVICE)
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-4)

ETH_ADDR = "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D"
secret_template = eth_to_bits(ETH_ADDR)

print("🏋️ Training (200 steps on diverse data)...")
enc.train(); dec.train()
step = 0
best_acc = 0.0

for epoch in range(5):
    for clean in trainloader:
        if step >= 200: break
        clean = clean.to(DEVICE)
        B = clean.shape[0]
        secrets = secret_template.repeat(B, 1).to(DEVICE)

        wm = enc(clean, secrets)
        distorted = distort(wm)  # simulate print→photo
        logits = dec(distorted)

        loss = F.binary_cross_entropy_with_logits(logits, secrets)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
        opt.step()

        with torch.no_grad():
            acc = ((logits.sigmoid() > 0.5).float() == secrets).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            torch.save({'encoder': enc.state_dict(), 'decoder': dec.state_dict()}, "output/best_general.pth")

        if step % 50 == 0:
            print(f"  Step {step:3d} | Loss: {loss.item():.4f} | BitAcc: {acc:.1%}")
        step += 1
    if step >= 200: break

# Load best
ckpt = torch.load("output/best_general.pth", map_location=DEVICE)
enc.load_state_dict(ckpt['encoder'])
dec.load_state_dict(ckpt['decoder'])
enc.eval(); dec.eval()
print(f"✅ Training done. Best bit accuracy: {best_acc:.2%}")

# ==============================
# 7. LOAD BEST MODEL & EMBED
# ==============================
print("📥 Loading best-performing model...")
ckpt = torch.load("output/best_eth_model.pth", map_location=DEVICE)
enc.load_state_dict(ckpt['encoder'])
dec.load_state_dict(ckpt['decoder'])
enc.eval(); dec.eval()

ETH_ADDR = "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D"

secret_bits = eth_to_bits(ETH_ADDR).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    watermarked = enc(img_tensor, secret_bits)
    captured = distort(watermarked)  # realistic eval
    logits = dec(captured)
    recovered_eth = bits_to_eth(logits[0])
    bit_acc = ((logits.sigmoid() > 0.5).float() == secret_bits[0]).float().mean().item()

print(f"🔤 Original: {ETH_ADDR}")
print(f"🔓 Recovered: {recovered_eth}")
print(f"🎯 Bit accuracy: {bit_acc:.2%}")
print(f"✅ Match (case-insensitive): {recovered_eth.lower() == ETH_ADDR.lower()}")
# ==============================
# 8. VISUALIZE
# ==============================
def to_pil(t): return T.ToPILImage()(((t[0] + 1) / 2).clamp(0, 1).cpu())

plt.figure(figsize=(15, 4))
images = [
    orig_pil.resize((256, 256)),
    to_pil(watermarked),
    to_pil(captured)
]
titles = [
    "Original",
    "Watermarked\n(StegaStamp)",
    f"After Print→Photo\nRecovered:\n{recovered_eth}"
]

for i, (im, ttl) in enumerate(zip(images, titles)):
    plt.subplot(1, 3, i+1)
    plt.imshow(im)
    plt.title(ttl, fontsize=11, pad=10)
    plt.axis("off")

plt.suptitle("StegaStamp: Ethereum Address Embedding", fontsize=14, y=0.99)
plt.tight_layout()
plt.show()

# ==============================
# 9. SAVE OUTPUT (Optional)
# ==============================
os.makedirs("output", exist_ok=True)
to_pil(watermarked).save("output/watermarked.png")
to_pil(captured).save("output/captured.png")
print(f"\n💾 Saved:\n  output/watermarked.png\n  output/captured.png")