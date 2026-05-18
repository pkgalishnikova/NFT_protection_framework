import os, sys, random, warnings, io, base64, zipfile, shutil, requests
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

# ==============================
# 1. INSTALL reedsolo
# ==============================
try:
    import reedsolo
except ImportError:
    print("🔧 Installing reedsolo...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "reedsolo"])
    import reedsolo

# ==============================
# 2. CONFIG: Full Ethereum address (42 chars = 336 bits)
# ==============================
SECRET_STR = "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D"
assert len(SECRET_STR) == 42, f"Expected 42-char Ethereum address, got {len(SECRET_STR)}"

SECRET_BYTES = 42
SECRET_LEN = SECRET_BYTES * 8  # 336 bits
PARITY_BYTES = 20              # Reed-Solomon parity
ECC_LEN = (SECRET_BYTES + PARITY_BYTES) * 8  # (42+20)*8 = 496 bits

RS = reedsolo.RSCodec(PARITY_BYTES)

print(f"🔐 Secret: {SECRET_STR}")
print(f"📊 Secret bits: {SECRET_LEN}, ECC bits: {ECC_LEN}")

# ==============================
# 3. MODELS (scaled for 336-bit secret)
# ==============================
class StegaStampEncoder(nn.Module):
    def __init__(self, secret_len=SECRET_LEN):
        super().__init__()
        self.secret_dense = nn.Sequential(
            nn.Linear(secret_len, 50*50*16),
            nn.ReLU()
        )
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256+16, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1), nn.Tanh()
        )

    def forward(self, image, secret):
        B, _, H, W = image.shape
        sec = self.secret_dense(secret).view(B, 16, 50, 50)
        sec = F.interpolate(sec, size=(H//8, W//8), mode='nearest')
        x = self.enc(image)
        x = self.bottleneck(torch.cat([x, sec], dim=1))
        return torch.clamp(image + self.dec(x), -1, 1)

class StegaStampDecoder(nn.Module):
    def __init__(self, secret_len=ECC_LEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Linear(256, secret_len)

    def forward(self, x):
        return self.head(self.net(x).flatten(1))

# ==============================
# 4. UTILS: Bit conversion & ECC
# ==============================
def text_to_bits(s, secret_bytes=SECRET_BYTES):
    s = s.encode('utf-8')
    if len(s) > secret_bytes:
        s = s[:secret_bytes]
    else:
        s = s.ljust(secret_bytes, b'\x00')
    bits = np.unpackbits(np.frombuffer(s, dtype=np.uint8))
    return torch.from_numpy(bits.astype(np.float32))

def bits_to_text(b, secret_bytes=SECRET_BYTES):
    b = (b > 0.5).cpu().numpy().astype(np.uint8)
    try:
        recovered_bytes = np.packbits(b[:secret_bytes * 8])
        return bytes(recovered_bytes).decode('utf-8', errors='replace').rstrip('\x00')
    except Exception as e:
        return f"[DECODE ERROR: {e}]"

def encode_ecc(b):
    out = []
    for row in b.cpu().numpy().astype(np.uint8):
        packed = np.packbits(row).tobytes()
        encoded = RS.encode(packed)
        bits = np.unpackbits(np.frombuffer(encoded, np.uint8))[:ECC_LEN]
        out.append(torch.from_numpy(bits.astype(np.float32)))
    return torch.stack(out).to(b.device)

def decode_ecc(logits):
    out = []
    for row in logits.cpu().numpy():
        hard_bits = (torch.sigmoid(torch.tensor(row)) > 0.5).numpy().astype(np.uint8)
        try:
            packed = np.packbits(hard_bits).tobytes()
            corrected, _, _ = RS.decode(packed)
            bits = np.unpackbits(np.frombuffer(corrected, np.uint8))[:SECRET_LEN]
        except:
            bits = np.zeros(SECRET_LEN)
        out.append(torch.from_numpy(bits.astype(np.float32)))
    return torch.stack(out).to(logits.device)

def distort(img, training=True):
    B, C, H, W = img.shape
    x = (img + 1) / 2  # [-1,1] -> [0,1]
    from io import BytesIO
    pil_imgs = []
    for i in range(B):
        pil = T.ToPILImage()(x[i].cpu().clamp(0, 1))
        buf = BytesIO()
        quality = random.randint(50, 95) if training else random.randint(65, 90)
        pil.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        pil = Image.open(buf).convert("RGB")
        scale = random.uniform(0.7, 1.3) if training else random.uniform(0.85, 1.15)
        pil = pil.resize((int(W * scale), int(H * scale)), Image.BILINEAR)
        left = max(0, (pil.width - W) // 2)
        top = max(0, (pil.height - H) // 2)
        pil = pil.crop((left, top, left + W, top + H))
        if training:
            pil = T.functional.adjust_brightness(pil, random.uniform(0.7, 1.3))
            pil = T.functional.adjust_contrast(pil, random.uniform(0.7, 1.3))
        else:
            pil = T.functional.adjust_brightness(pil, random.uniform(0.9, 1.1))
            pil = T.functional.adjust_contrast(pil, random.uniform(0.9, 1.1))
        pil_imgs.append(T.ToTensor()(pil))
    x = torch.stack(pil_imgs).to(img.device)
    return torch.clamp(x, 0, 1) * 2 - 1

# ==============================
# 5. DOWNLOAD FIRST 100 COCO IMAGES
# ==============================
def download_coco_100():
    data_dir = "./coco"
    img_dir = os.path.join(data_dir, "train2017")
    ann_file = os.path.join(data_dir, "annotations/instances_train2017.json")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(ann_file), exist_ok=True)

    # Download annotations if missing
    if not os.path.exists(ann_file):
        print("⬇️  Downloading COCO annotations...")
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        r = requests.get(url, stream=True)
        zip_path = os.path.join(data_dir, "annotations.zip")
        with open(zip_path, 'wb') as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)

    # Get first 100 image URLs
    from pycocotools.coco import COCO
    coco = COCO(ann_file)
    img_ids = list(coco.imgs.keys())[:100]
    img_info = [coco.imgs[i] for i in img_ids]

    print("⬇️  Downloading first 100 COCO train images...")
    img_paths = []
    for i, info in enumerate(img_info):
        path = os.path.join(img_dir, info['file_name'])
        if not os.path.exists(path):
            try:
                r = requests.get(info['coco_url'], stream=True)
                r.raise_for_status()
                with open(path, 'wb') as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
            except Exception as e:
                print(f"⚠️  Skip {info['file_name']}: {e}")
                continue
        img_paths.append(path)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/100 downloaded")

    return img_paths

print("🔍 Setting up COCO-100 dataset...")
try:
    from pycocotools.coco import COCO
except ImportError:
    print("📦 Installing pycocotools...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pycocotools"])
    from pycocotools.coco import COCO

img_paths = download_coco_100()
print(f"✅ Loaded {len(img_paths)} COCO images")

# Dataset class
class ImagePathDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

transform = T.Compose([
    T.Resize(280),
    T.RandomCrop(256),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = ImagePathDataset(img_paths, transform=transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

# ==============================
# 6. TRAINING
# ==============================
enc = StegaStampEncoder().to(DEVICE)
dec = StegaStampDecoder().to(DEVICE)
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-4)

print("\n🏋️ Training on COCO-100 for Ethereum address embedding...")
enc.train(); dec.train()
total_steps = 0
max_steps = 600  # ~75 epochs over 100 images

while total_steps < max_steps:
    for images in train_loader:
        images = images.to(DEVICE)
        B = images.size(0)
        secrets_raw = torch.randint(0, 2, (B, SECRET_LEN), dtype=torch.float32, device=DEVICE)
        secrets_ecc = encode_ecc(secrets_raw)

        wm = enc(images, secrets_raw)
        distorted = distort(wm, training=True)
        logits = dec(distorted)
        loss = F.binary_cross_entropy_with_logits(logits, secrets_ecc)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if total_steps % 60 == 0:
            acc = (logits.sigmoid() > 0.5).float().eq(secrets_ecc).float().mean().item()
            print(f"Step {total_steps}: Loss={loss.item():.4f}, BitAcc={acc:.2%}")
        total_steps += 1
        if total_steps >= max_steps:
            break

print("✅ Training complete.")

# ==============================
# 7. INFERENCE ON USER IMAGE
# ==============================
print("\n📤 Please upload an image (JPG/PNG)")

try:
    from google.colab import files
    uploaded = files.upload()
    img_path = list(uploaded.keys())[0]
    print(f"✅ Uploaded: {img_path}")
except:
    print("ℹ️  No upload — using demo image")
    demo_img = Image.new('RGB', (400, 400), (240, 240, 240))
    draw = ImageDraw.Draw(demo_img)
    draw.rectangle([80, 80, 320, 320], fill=(60, 120, 220))
    demo_img.save("/tmp/demo.jpg")
    img_path = "/tmp/demo.jpg"

# Preprocess
orig_pil = Image.open(img_path).convert("RGB")
transform_inf = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
img_tensor = transform_inf(orig_pil).unsqueeze(0).to(DEVICE)

# Encode full Ethereum address
secret_bits = text_to_bits(SECRET_STR).unsqueeze(0).to(DEVICE)

enc.eval(); dec.eval()
with torch.no_grad():
    watermarked = enc(img_tensor, secret_bits)
    captured = distort(watermarked, training=False)
    logits = dec(captured)
    recovered_bits = decode_ecc(logits)
    recovered_text = bits_to_text(recovered_bits[0])
    bit_acc = (recovered_bits[0] == secret_bits[0]).float().mean().item()

print(f"\n🔤 Original: {SECRET_STR}")
print(f"🔓 Recovered: {recovered_text}")
print(f"✅ Bit accuracy: {bit_acc:.2%}")

# ==============================
# 8. VISUALIZE
# ==============================
def to_pil(t):
    return T.ToPILImage()(((t[0] + 1) / 2).clamp(0, 1).cpu())

plt.figure(figsize=(15, 4))
images = [orig_pil.resize((256, 256)), to_pil(watermarked), to_pil(captured)]
titles = ["Original", "Watermarked", f"After Distortion\nRecovered: {recovered_text[:20]}..."]

for i, (im, ttl) in enumerate(zip(images, titles)):
    plt.subplot(1, 3, i + 1)
    plt.imshow(im)
    plt.title(ttl, fontsize=11)
    plt.axis("off")

plt.suptitle("StegaStamp: Ethereum Address Embedding (Trained on COCO-100)", fontsize=13)
plt.tight_layout()
plt.show()

# ==============================
# 9. SAVE OUTPUT
# ==============================
save_dir = "stegastamp_eth"
os.makedirs(save_dir, exist_ok=True)
to_pil(watermarked).save(f"{save_dir}/watermarked.png")
to_pil(captured).save(f"{save_dir}/captured.png")
torch.save(enc.state_dict(), f"{save_dir}/encoder.pth")
torch.save(dec.state_dict(), f"{save_dir}/decoder.pth")
print(f"\n💾 Saved to '{save_dir}/'")