# %%
import os, sys, random, warnings, io, base64
warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================
# 1. INSTALL reedsolo (for ECC)
# ==============================
try:
    import reedsolo
    RS = reedsolo.RSCodec(20)
except ImportError:
    print("🔧 Installing reedsolo...")
    %pip -q install reedsolo
    import reedsolo
    RS = reedsolo.RSCodec(20)

# ==============================
# 2. MODELS
# ==============================
SECRET_LEN, ECC_LEN = 100, 180

class StegaStampEncoder(nn.Module):
    def __init__(self, secret_len=SECRET_LEN):
        super().__init__()
        self.secret_dense = nn.Sequential(nn.Linear(secret_len, 50*50*16), nn.ReLU())
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
        sec = self.secret_dense(secret).view(B,16,50,50)
        sec = F.interpolate(sec, size=(H//8, W//8), mode='nearest')
        x = self.enc(image)
        x = self.bottleneck(torch.cat([x, sec], 1))
        return torch.clamp(image + self.dec(x), -1, 1)

class StegaStampDecoder(nn.Module):
    def __init__(self, secret_len=ECC_LEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,2,1), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(),
            nn.Conv2d(64,128,3,2,1), nn.ReLU(),
            nn.Conv2d(128,256,3,2,1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Linear(256, secret_len)
    def forward(self, x): return self.head(self.net(x).flatten(1))

# ==============================
# 3. UTILS: ECC + DISTORTION
# ==============================
def text_to_bits(s):  # str → 100-bit tensor
    s = s.ljust(7)[:7].encode()
    bits = np.unpackbits(np.frombuffer(s, np.uint8))
    return torch.from_numpy(np.pad(bits, (0,100-len(bits)))[:100].astype(np.float32))

def bits_to_text(b):  # bits → str
    b = (b > 0.5).cpu().numpy().astype(np.uint8)
    try: return bytes(np.packbits(b[:56])).decode().rstrip('\x00')
    except: return "[FAIL]"

def encode_ecc(b):    # [B,100] → [B,180]
    out = []
    for row in b.cpu().numpy().astype(np.uint8):
        enc = RS.encode(np.packbits(row).tobytes())
        bits = np.unpackbits(np.frombuffer(enc, np.uint8))[:180]
        out.append(torch.from_numpy(bits.astype(np.float32)))
    return torch.stack(out).to(b.device)

def decode_ecc(log):  # [B,180] → [B,100]
    out = []
    for row in log.cpu().numpy():
        hard = (torch.sigmoid(torch.tensor(row)) > 0.5).numpy().astype(np.uint8)
        try:
            corr,_,_ = RS.decode(np.packbits(hard).tobytes())
            bits = np.unpackbits(np.frombuffer(corr, np.uint8))[:100]
        except:
            bits = np.zeros(100)
        out.append(torch.from_numpy(bits.astype(np.float32)))
    return torch.stack(out).to(log.device)

def distort(img, training=False):
    B, C, H, W = img.shape
    x = (img + 1)/2  # [-1,1] → [0,1]
    from io import BytesIO
    pil_imgs = []
    for i in range(B):
        pil = T.ToPILImage()(x[i].cpu().clamp(0,1))
        buf = BytesIO()
        pil.save(buf, "JPEG", quality=random.randint(65, 90))
        buf.seek(0)
        pil = Image.open(buf).convert("RGB")
        # Resize + center crop
        scale = random.uniform(0.85, 1.15)
        pil = pil.resize((int(W*scale), int(H*scale)), Image.BILINEAR)
        left = max(0, (pil.width - W)//2)
        top = max(0, (pil.height - H)//2)
        pil = pil.crop((left, top, left+W, top+H))
        # Brightness/contrast jitters
        pil = T.functional.adjust_brightness(pil, random.uniform(0.9, 1.1))
        pil = T.functional.adjust_contrast(pil, random.uniform(0.9, 1.1))
        pil_imgs.append(T.ToTensor()(pil))
    x = torch.stack(pil_imgs).to(img.device)
    return torch.clamp(x, 0, 1)*2 - 1

print("📤 Please upload an image (JPG/PNG, any size)")

# Colab upload
try:
    from google.colab import files
    uploaded = files.upload()
    img_path = list(uploaded.keys())[0]
    print(f"✅ Uploaded: {img_path}")
except:
    # Local/Jupyter fallback: use base64 demo image if no upload
    print("ℹ️  No upload interface — using demo image")
    # Create a simple test image (red square on white)
    demo_img = Image.new('RGB', (300, 300), (255, 255, 255))
    draw = ImageDraw.Draw(demo_img)
    draw.rectangle([50, 50, 250, 250], fill=(220, 60, 60))
    demo_img.save("/tmp/demo.jpg")
    img_path = "/tmp/demo.jpg"

# Load & preprocess image
orig_pil = Image.open(img_path).convert("RGB")
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
img_tensor = transform(orig_pil).unsqueeze(0).to(DEVICE)

# ==============================
# 5. INITIALIZE & TRAIN (1 STEP)
# ==============================
enc = StegaStampEncoder().to(DEVICE)
dec = StegaStampDecoder().to(DEVICE)
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=5e-4)

print("🏋️ Quick adaptation (1 step)...")
enc.train(); dec.train()
for _ in range(1):
    secrets_raw = torch.randint(0,2,(1,SECRET_LEN), dtype=torch.float32, device=DEVICE)
    secrets_ecc = encode_ecc(secrets_raw)
    wm = enc(img_tensor, secrets_raw)
    distorted = distort(wm, training=False)
    logits = dec(distorted)
    loss = F.binary_cross_entropy_with_logits(logits, secrets_ecc)
    opt.zero_grad(); loss.backward(); opt.step()

enc.eval(); dec.eval()
print("✅ Model warmed up")

# ==============================
# 6. ENCODE YOUR SECRET
# ==============================
SECRET = "Hi!"  # ← CHANGE THIS TO YOUR MESSAGE (≤7 chars)
print(f"\n🔐 Encoding secret: '{SECRET}'")

secret_bits = text_to_bits(SECRET).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    watermarked = enc(img_tensor, secret_bits)
    captured = distort(watermarked, training=False)  # simulate print→photo
    decoded_logits = dec(captured)
    recovered_bits = decode_ecc(decoded_logits)
    recovered_text = bits_to_text(recovered_bits[0])
    bit_acc = (recovered_bits[0] == secret_bits[0]).float().mean().item()

print(f"🔤 Original: '{SECRET}' → 🔓 Recovered: '{recovered_text}' (bit acc: {bit_acc:.1%})")

# ==============================
# 7. VISUALIZE RESULTS
# ==============================
def to_pil(t): return T.ToPILImage()(((t[0]+1)/2).clamp(0,1).cpu())

plt.figure(figsize=(15, 4))
images = [orig_pil.resize((256,256)), to_pil(watermarked), to_pil(captured)]
titles = ["Original", f"Watermarked\n(Δ = subtle)", f"After Print→Photo\nRecovered: '{recovered_text}'"]

for i, (im, ttl) in enumerate(zip(images, titles)):
    plt.subplot(1, 3, i+1)
    plt.imshow(im)
    plt.title(ttl, fontsize=12)
    plt.axis("off")

plt.suptitle("StegaStamp Demo with Your Image", fontsize=14, y=0.98)
plt.tight_layout()
plt.show()

# ==============================
# 8. SAVE RESULTS (Optional)
# ==============================
save_dir = "stegastamp_output"
os.makedirs(save_dir, exist_ok=True)
to_pil(watermarked).save(f"{save_dir}/watermarked.png")
to_pil(captured).save(f"{save_dir}/captured.png")
print(f"\n💾 Saved:\n  - watermarked.png\n  - captured.png\nin folder '{save_dir}/'")