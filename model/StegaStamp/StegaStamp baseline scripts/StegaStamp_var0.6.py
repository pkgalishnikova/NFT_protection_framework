import os, sys, random, warnings, io, zipfile, shutil, requests
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

try:
    import reedsolo
except ImportError:
    print("Installing reedsolo...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "reedsolo"])
    import reedsolo

SECRET_STR = "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D"
assert len(SECRET_STR) == 42, f"Expected 42-char Ethereum address, got {len(SECRET_STR)}"

SECRET_BYTES = 42
PARITY_BYTES = 20
MESSAGE_LEN = (SECRET_BYTES + PARITY_BYTES) * 8

RS = reedsolo.RSCodec(PARITY_BYTES)

print(f"Secret: {SECRET_STR}")
print(f"Message length with ECC: {MESSAGE_LEN} bits")

class StegaStampEncoder(nn.Module):
    def __init__(self, secret_len=MESSAGE_LEN):
        super().__init__()
        self.secret_len = secret_len

        self.secret_dense = nn.Sequential(
            nn.Linear(secret_len, 64*64*8),
            nn.ReLU()
        )

        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256+8, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
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
        sec = self.secret_dense(secret).view(B, 8, 64, 64)
        sec = F.interpolate(sec, size=(H//8, W//8), mode='bilinear', align_corners=False)

        x = self.enc(image)
        x = self.bottleneck(torch.cat([x, sec], dim=1))
        residual = self.dec(x)

        return torch.clamp(image + 0.1 * residual, -1, 1)

class StegaStampDecoder(nn.Module):
    def __init__(self, secret_len=MESSAGE_LEN):
        super().__init__()
        self.secret_len = secret_len

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, secret_len)
        )

    def forward(self, x):
        features = self.net(x).flatten(1)
        return self.head(features)

def text_to_ecc_bits(text):
    text_bytes = text.encode('utf-8')
    if len(text_bytes) > SECRET_BYTES:
        text_bytes = text_bytes[:SECRET_BYTES]
    else:
        text_bytes = text_bytes.ljust(SECRET_BYTES, b'\x00')

    ecc_bytes = RS.encode(text_bytes)

    bits = np.unpackbits(np.frombuffer(ecc_bytes, dtype=np.uint8))

    return torch.from_numpy(bits.astype(np.float32))

def ecc_bits_to_text(bits):
    bits_np = (bits > 0.5).cpu().numpy().astype(np.uint8)
    ecc_bytes = np.packbits(bits_np).tobytes()

    try:
        decoded_bytes = RS.decode(ecc_bytes)[0]
        text = decoded_bytes.decode('utf-8', errors='replace').rstrip('\x00')
        return text, True
    except reedsolo.ReedSolomonError as e:
        return f"[ECC FAILED: {e}]", False

def distort(img, severity='medium'):
    B, C, H, W = img.shape
    x = (img + 1) / 2

    from io import BytesIO
    pil_imgs = []

    for i in range(B):
        pil = T.ToPILImage()(x[i].cpu().clamp(0, 1))

        buf = BytesIO()
        if severity == 'hard':
            quality = random.randint(40, 70)
        elif severity == 'medium':
            quality = random.randint(60, 85)
        else:
            quality = random.randint(75, 95)

        pil.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        pil = Image.open(buf).convert("RGB")

        if severity == 'hard':
            scale = random.uniform(0.6, 1.4)
        elif severity == 'medium':
            scale = random.uniform(0.8, 1.2)
        else:
            scale = random.uniform(0.9, 1.1)

        new_size = (int(W * scale), int(H * scale))
        pil = pil.resize(new_size, Image.BILINEAR)

        left = max(0, (pil.width - W) // 2)
        top = max(0, (pil.height - H) // 2)
        pil = pil.crop((left, top, left + W, top + H))

        if pil.size != (W, H):
            pil = pil.resize((W, H), Image.BILINEAR)

        if severity == 'hard':
            pil = T.functional.adjust_brightness(pil, random.uniform(0.6, 1.4))
            pil = T.functional.adjust_contrast(pil, random.uniform(0.6, 1.4))
        elif severity == 'medium':
            pil = T.functional.adjust_brightness(pil, random.uniform(0.8, 1.2))
            pil = T.functional.adjust_contrast(pil, random.uniform(0.8, 1.2))

        pil_imgs.append(T.ToTensor()(pil))

    x = torch.stack(pil_imgs).to(img.device)
    return torch.clamp(x, 0, 1) * 2 - 1

def download_coco_500():
    data_dir = "./coco"
    img_dir = os.path.join(data_dir, "train2017")
    ann_file = os.path.join(data_dir, "annotations/instances_train2017.json")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(ann_file), exist_ok=True)

    if not os.path.exists(ann_file):
        print("Downloading COCO annotations...")
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        try:
            r = requests.get(url, stream=True, timeout=60)
            zip_path = os.path.join(data_dir, "annotations.zip")
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(1024*1024):
                    f.write(chunk)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            os.remove(zip_path)
        except Exception as e:
            print(f"Could not download annotations: {e}")
            return []

    try:
        from pycocotools.coco import COCO
    except ImportError:
        print("Installing pycocotools...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pycocotools"])
        from pycocotools.coco import COCO

    coco = COCO(ann_file)
    img_ids = list(coco.imgs.keys())[:500]
    img_info = [coco.imgs[i] for i in img_ids]

    print(f"Downloading first 500 COCO images (this may take 5-10 minutes)...")
    img_paths = []

    for i, info in enumerate(img_info):
        path = os.path.join(img_dir, info['file_name'])

        if not os.path.exists(path):
            try:
                r = requests.get(info['coco_url'], stream=True, timeout=10)
                r.raise_for_status()
                with open(path, 'wb') as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
            except Exception as e:
                if i < 10:
                    print(f"Skip {info['file_name']}: {e}")
                continue

        img_paths.append(path)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/500 downloaded/found")

    return img_paths

print("Setting up COCO dataset...")
img_paths = download_coco_500()
print(f"Loaded {len(img_paths)} COCO images")

if len(img_paths) < 50:
    print("Not enough images downloaded. Using synthetic data...")
    img_paths = []
    os.makedirs("synthetic_data", exist_ok=True)
    for i in range(200):
        img = Image.new('RGB', (400, 400),
                       (random.randint(100, 255),
                        random.randint(100, 255),
                        random.randint(100, 255)))
        path = f"synthetic_data/img_{i:04d}.jpg"
        img.save(path)
        img_paths.append(path)
    print(f"Created {len(img_paths)} synthetic images")

class ImagePathDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img
        except:
            return torch.randn(3, 256, 256)

transform = T.Compose([
    T.Resize(280),
    T.RandomCrop(256),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImagePathDataset(img_paths, transform=transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)

enc = StegaStampEncoder().to(DEVICE)
dec = StegaStampDecoder().to(DEVICE)

opt_enc = torch.optim.Adam(enc.parameters(), lr=1e-4)
opt_dec = torch.optim.Adam(dec.parameters(), lr=2e-4)

print("\nTraining with improved ECC embedding...")
print(f"   Dataset size: {len(img_paths)} images")
print(f"   Training steps: 2000 ({2000//len(train_loader)} epochs)")

enc.train()
dec.train()

total_steps = 0
max_steps = 2000
best_acc = 0

print("\nGenerating random secret pool...")
NUM_RANDOM_SECRETS = 50
random_secrets = []
for i in range(NUM_RANDOM_SECRETS):
    random_text = ''.join([chr(random.randint(65, 90)) for _ in range(42)])
    random_ecc = text_to_ecc_bits(random_text)
    random_secrets.append(random_ecc)

target_secret_ecc = text_to_ecc_bits(SECRET_STR)
random_secrets.append(target_secret_ecc)

print(f"Created pool of {len(random_secrets)} secrets (including target)")

while total_steps < max_steps:
    for images in train_loader:
        images = images.to(DEVICE)
        B = images.size(0)

        secrets_batch = []
        for _ in range(B):
            secret_idx = random.randint(0, len(random_secrets) - 1)
            secrets_batch.append(random_secrets[secret_idx])
        secrets_batch = torch.stack(secrets_batch).to(DEVICE)

        watermarked = enc(images, secrets_batch)

        if total_steps < 500:
            distorted = distort(watermarked, severity='easy')
        elif total_steps < 1200:
            distorted = distort(watermarked, severity='medium')
        else:
            distorted = distort(watermarked, severity='hard')

        logits = dec(distorted)

        secret_loss = F.binary_cross_entropy_with_logits(logits, secrets_batch)
        image_loss = F.mse_loss(watermarked, images)

        loss = secret_loss + 0.3 * image_loss

        opt_enc.zero_grad()
        opt_dec.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)

        opt_enc.step()
        opt_dec.step()

        if total_steps % 100 == 0:
            with torch.no_grad():
                pred_bits = (logits.sigmoid() > 0.5).float()
                bit_acc = pred_bits.eq(secrets_batch).float().mean().item()

                target_batch = target_secret_ecc.unsqueeze(0).to(DEVICE)
                test_wm = enc(images[:1], target_batch)
                test_dist = distort(test_wm, severity='medium')
                test_logits = dec(test_dist)
                target_pred = (test_logits.sigmoid() > 0.5).float()
                target_acc = target_pred.eq(target_batch).float().mean().item()

                recovered_text, success = ecc_bits_to_text(target_pred[0])

                print(f"\nStep {total_steps}:")
                print(f"  Training Loss: {loss.item():.4f} (Secret: {secret_loss.item():.4f}, Image: {image_loss.item():.6f})")
                print(f"  Training Bit Acc (random secrets): {bit_acc:.2%}")
                print(f"  TARGET Bit Acc: {target_acc:.2%}")
                print(f"  TARGET Recovered: {recovered_text[:30]}{'...' if len(recovered_text) > 30 else ''}")
                print(f"  ECC Success: {'Yes' if success else 'No'}")

                if target_acc > best_acc:
                    best_acc = target_acc
                    if target_acc > 0.85:
                        print(f"  New best TARGET accuracy: {best_acc:.2%}")

        total_steps += 1
        if total_steps >= max_steps:
            break

print(f"\nTraining complete. Best TARGET accuracy: {best_acc:.2%}")

print("\nUpload an image for watermarking (or press Enter to use demo)")

try:
    from google.colab import files
    uploaded = files.upload()
    if uploaded:
        img_path = list(uploaded.keys())[0]
        print(f"Uploaded: {img_path}")
    else:
        raise Exception("No upload")
except:
    print("Using demo image")
    demo_img = Image.new('RGB', (400, 400), (240, 240, 240))
    draw = ImageDraw.Draw(demo_img)
    draw.rectangle([80, 80, 320, 320], fill=(60, 120, 220))
    draw.text((150, 180), "DEMO", fill=(255, 255, 255))
    demo_img.save("/tmp/demo.jpg")
    img_path = "/tmp/demo.jpg"

orig_pil = Image.open(img_path).convert("RGB")
transform_inf = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
img_tensor = transform_inf(orig_pil).unsqueeze(0).to(DEVICE)

secret_bits = text_to_ecc_bits(SECRET_STR).unsqueeze(0).to(DEVICE)

enc.eval()
dec.eval()

with torch.no_grad():
    watermarked = enc(img_tensor, secret_bits)

    print("\nTesting robustness...")

    logits_clean = dec(watermarked)
    recovered_clean, success_clean = ecc_bits_to_text(logits_clean[0])
    bit_acc_clean = (logits_clean.sigmoid() > 0.5).float().eq(secret_bits).float().mean().item()

    distorted_easy = distort(watermarked, severity='easy')
    logits_easy = dec(distorted_easy)
    recovered_easy, success_easy = ecc_bits_to_text(logits_easy[0])
    bit_acc_easy = (logits_easy.sigmoid() > 0.5).float().eq(secret_bits).float().mean().item()

    distorted_medium = distort(watermarked, severity='medium')
    logits_medium = dec(distorted_medium)
    recovered_medium, success_medium = ecc_bits_to_text(logits_medium[0])
    bit_acc_medium = (logits_medium.sigmoid() > 0.5).float().eq(secret_bits).float().mean().item()

print(f"\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Original Secret: {SECRET_STR}")
print(f"\nTest Results:")
print(f"  Clean watermark:")
print(f"    Bit Accuracy: {bit_acc_clean:.2%}")
print(f"    Recovered: {recovered_clean}")
print(f"    ECC Success: {'Yes' if success_clean else 'No'}")
print(f"\n  After easy attack (JPEG 75-95, scale 0.9-1.1):")
print(f"    Bit Accuracy: {bit_acc_easy:.2%}")
print(f"    Recovered: {recovered_easy}")
print(f"    ECC Success: {'Yes' if success_easy else 'No'}")
print(f"\n  After medium attack (JPEG 60-85, scale 0.8-1.2):")
print(f"    Bit Accuracy: {bit_acc_medium:.2%}")
print(f"    Recovered: {recovered_medium}")
print(f"    ECC Success: {'Yes' if success_medium else 'No'}")
print("="*70)

def to_pil(t):
    return T.ToPILImage()(((t[0] + 1) / 2).clamp(0, 1).cpu())

plt.figure(figsize=(18, 5))
images = [
    orig_pil.resize((256, 256)),
    to_pil(watermarked),
    to_pil(distorted_easy),
    to_pil(distorted_medium)
]
titles = [
    "Original",
    f"Watermarked\n(Clean: {recovered_clean[:20]}...)",
    f"Easy Attack\n(Recovered: {recovered_easy[:20]}...)",
    f"Medium Attack\n(Recovered: {recovered_medium[:20]}...)"
]

for i, (im, ttl) in enumerate(zip(images, titles)):
    plt.subplot(1, 4, i + 1)
    plt.imshow(im)
    plt.title(ttl, fontsize=10)
    plt.axis("off")

plt.suptitle(f"StegaStamp with ECC: Ethereum Address Embedding (Best: {best_acc:.1%})",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

save_dir = "stegastamp_eth_fixed"
os.makedirs(save_dir, exist_ok=True)

to_pil(watermarked).save(f"{save_dir}/watermarked.png")
to_pil(distorted_easy).save(f"{save_dir}/attacked_easy.png")
to_pil(distorted_medium).save(f"{save_dir}/attacked_medium.png")

torch.save({
    'encoder': enc.state_dict(),
    'decoder': dec.state_dict(),
    'secret': SECRET_STR,
    'best_acc': best_acc
}, f"{save_dir}/model.pth")

with open(f"{save_dir}/results.txt", 'w') as f:
    f.write(f"Original: {SECRET_STR}\n")
    f.write(f"Clean: {recovered_clean} ({bit_acc_clean:.2%})\n")
    f.write(f"Easy: {recovered_easy} ({bit_acc_easy:.2%})\n")
    f.write(f"Medium: {recovered_medium} ({bit_acc_medium:.2%})\n")
    f.write(f"Best training accuracy: {best_acc:.2%}\n")

print(f"\nSaved to '{save_dir}/'")
print(f"\nDONE! Check the visualization and results above.")
