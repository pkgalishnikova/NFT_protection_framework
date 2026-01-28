# base model
import os, sys, random, warnings
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

def load_clip_model(device='cuda'):
    """Load CLIP model (only once)"""
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

# Global CLIP model (load once)
CLIP_MODEL, CLIP_PREPROCESS = None, None

def init_clip():
    global CLIP_MODEL, CLIP_PREPROCESS
    if CLIP_MODEL is None:
        CLIP_MODEL, CLIP_PREPROCESS = load_clip_model(DEVICE)

def tensor_to_pil(img_tensor):
    """Convert [-1,1] tensor to PIL Image"""
    img = (img_tensor + 1) / 2  # [-1,1] → [0,1]
    img = torch.clamp(img, 0, 1)
    return T.ToPILImage()(img.cpu())

def calculate_clip_metrics(original_img, watermarked_img, secret_text):
    """
    Calculate all 3 CLIP-based metrics
    Args:
        original_img: Tensor [B, C, H, W] in [-1, 1]
        watermarked_img: Tensor [B, C, H, W] in [-1, 1]
        secret_text: str (e.g., "0xBC4CA0EdA7")
    Returns:
        dict with CLIPdir, CLIPimg, CLIPout
    """
    init_clip()

    # Convert tensors to PIL images
    orig_pil = tensor_to_pil(original_img[0])
    wm_pil = tensor_to_pil(watermarked_img[0])

    # Preprocess images for CLIP
    orig_clip = CLIP_PREPROCESS(orig_pil).unsqueeze(0).to(DEVICE)
    wm_clip = CLIP_PREPROCESS(wm_pil).unsqueeze(0).to(DEVICE)

    # Encode images
    with torch.no_grad():
        orig_features = CLIP_MODEL.encode_image(orig_clip)
        wm_features = CLIP_MODEL.encode_image(wm_clip)

        # Text encoding
        text_tokens = clip.tokenize([secret_text]).to(DEVICE)
        text_features = CLIP_MODEL.encode_text(text_tokens)

    # Normalize features
    orig_features = orig_features / orig_features.norm(dim=-1, keepdim=True)
    wm_features = wm_features / wm_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # (1) CLIP image similarity (original vs watermarked)
    clip_img = (orig_features @ wm_features.T).item()

    # (2) CLIP output similarity (watermarked vs text)
    clip_out = (wm_features @ text_features.T).item()

    # (3) CLIP direction similarity
    # Direction vectors
    orig_to_wm = wm_features - orig_features
    orig_to_text = text_features - orig_features

    # Normalize direction vectors
    orig_to_wm = orig_to_wm / orig_to_wm.norm(dim=-1, keepdim=True)
    orig_to_text = orig_to_text / orig_to_text.norm(dim=-1, keepdim=True)

    clip_dir = (orig_to_wm @ orig_to_text.T).item()

    return {
        'CLIPimg': clip_img,
        'CLIPout': clip_out,
        'CLIPdir': clip_dir
    }

def calculate_mse(original_img, watermarked_img):
    """Calculate MSE between images"""
    return F.mse_loss(original_img, watermarked_img).item()

def calculate_asr_and_ear(decoder, watermarked_img, target_secret, threshold=0.5, num_trials=10):
    """
    Calculate Attack Success Rate (ASR) and Error Attack Rate (EAR)
    ASR: % of times we recover EXACT target secret
    EAR: % of times we recover WRONG but valid secret
    """
    recovered_secrets = []

    for _ in range(num_trials):
        attacked_img = simple_attack(watermarked_img)

        with torch.no_grad():
            logits = decoder(attacked_img)
            pred_bits = (torch.sigmoid(logits) > threshold).float()

        recovered_addr = bits_to_ethereum(pred_bits[0], MESSAGE_LEN, SECRET_SHORT)
        recovered_secrets.append(recovered_addr)

    target_str = SECRET_SHORT

    asr = sum(1 for addr in recovered_secrets if addr == target_str) / num_trials

    valid_but_wrong = 0
    for addr in recovered_secrets:
        if (addr.startswith('0x') and
            len(addr) == len(target_str) and
            addr != target_str):
            valid_but_wrong += 1

    ear = valid_but_wrong / num_trials

    return asr, ear, recovered_secrets


# ==============================
# ARCHITECTURE - MINIMAL CHANGES
# ==============================

class ImprovedEncoder(nn.Module):
    def __init__(self, secret_len=100):
        super().__init__()
        self.secret_len = secret_len

        # Expand secret to spatial map with positional awareness
        self.secret_preproc = nn.Sequential(
            nn.Linear(secret_len, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8 * 16 * 16),  # 8 channels, 16x16 spatial
        )

        # Image encoder (feature extractor)
        self.encoder = nn.Sequential(
            # Stage 1: 256x256 → 128x128
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # Stage 2: 128x128 → 64x64
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        # Fusion & refinement with residual blocks
        self.fusion = nn.Conv2d(128 + 8, 128, 1)
        self.refine = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )

        # Residual decoder (bottleneck skip connection)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, 3, 1, 1),
        )

        # Learnable scaling (instead of fixed 0.15)
        self.alpha = nn.Parameter(torch.tensor(0.15))  # init same as before, but trainable

    def forward(self, img, secret):
        B, _, H, W = img.shape

        # Process secret → spatial tensor
        sec_feat = self.secret_preproc(secret).view(B, 8, 16, 16)
        sec_up = F.interpolate(sec_feat, size=(H//4, W//4), mode='bilinear', align_corners=False)

        # Encode image
        img_feat = self.encoder(img)

        # Fuse
        fused = torch.cat([img_feat, sec_up], dim=1)
        fused = self.fusion(fused)
        refined = fused + self.refine(fused)  # residual connection

        # Decode residual
        residual = torch.tanh(self.decoder(refined))
        watermark = img + torch.tanh(self.alpha) * residual  # clamp via tanh(alpha) in [-1,1]

        return torch.clamp(watermark, -1, 1)


class ImprovedDecoder(nn.Module):
    def __init__(self, secret_len=100):
        super().__init__()
        self.secret_len = secret_len

        # Pyramid feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3),  # 256→128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),  # 128→64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),  # 64→32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 2, 1),  # 32→16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        # Pyramid pooling (global context)
        self.psp_pool = nn.AdaptiveAvgPool2d(1)
        self.psp_conv = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.ReLU(),
        )

        # Attention to enhance watermark regions
        self.attn = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 + 64, 512),  # 256 global + 64 pooled
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, secret_len),
        )

    def forward(self, img):
        x = self.stem(img)        # B x 32 x 128 x 128
        x = self.stage1(x)       # B x 64 x 64 x 64
        feat = self.stage2(x)    # B x 256 x 16 x 16

        # Global context
        global_feat = self.psp_pool(feat)        # B x 256 x 1 x 1
        pooled = self.psp_conv(global_feat).squeeze(-1).squeeze(-1)  # B x 64

        # Attention-modulated global average
        attn_map = self.attn(feat)               # B x 1 x 16 x 16
        attended = (feat * attn_map).mean(dim=[2, 3])  # B x 256

        # Fuse
        combined = torch.cat([attended, pooled], dim=1)  # B x (256+64)

        return self.classifier(combined)


# ==============================
# IMPROVED ATTACK - DIFFERENTIABLE
# ==============================

# CHANGE 2: Differentiable blur (no PIL, no detach!)
def differentiable_blur(img, sigma_range=(0.0, 2.0)):
    """Differentiable Gaussian blur using torch operations"""
    B, C, H, W = img.shape

    # Random sigma for each image in batch
    if training_mode:
        sigmas = [random.uniform(*sigma_range) for _ in range(B)]
        kernel_size = random.choice([3, 5])
    else:
        sigmas = [1.0] * B
        kernel_size = 5

    blurred = []
    for i in range(B):
        blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigmas[i])
        blurred.append(blur(img[i:i+1]))

    return torch.cat(blurred, dim=0)

# Keep old version for evaluation
def simple_attack(img):
    """Non-differentiable blur for evaluation only"""
    B, C, H, W = img.shape
    imgs_out = []

    for i in range(B):
        pil = T.ToPILImage()(((img[i] + 1) / 2).clamp(0, 1).cpu())
        img_tensor = T.ToTensor()(pil)
        blur = T.GaussianBlur(kernel_size=5, sigma=1.0)
        img_tensor = blur(img_tensor)
        imgs_out.append(img_tensor)

    imgs_out = torch.stack(imgs_out).to(img.device)
    return imgs_out * 2 - 1

# Global flag for training mode
training_mode = False

# ==============================
# UTILS
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


def bits_to_ethereum(bits, num_bits=100, original_secret=SECRET_SHORT):
    """Convert binary back to Ethereum address format"""
    bits_np = (bits[:num_bits] > 0.5).cpu().numpy().astype(np.uint8)
    binary_str = ''.join([str(int(b)) for b in bits_np])

    try:
        hex_value = hex(int(binary_str, 2))[2:].upper()
        num_hex_chars = num_bits // 4  # 100 bits = 25 hex chars
        hex_value = hex_value.zfill(num_hex_chars)

        original_hex_len = len(original_secret) - 2
        return '0x' + hex_value[:original_hex_len]
    except:
        original_hex_len = len(original_secret) - 2
        return "0x" + "?"*original_hex_len

def extract_residual(encoder, img, secret, alpha_scale=10):
    """
    Extract the residual pattern from the encoder
    Args:
        encoder: The encoder model
        img: Original image tensor
        secret: Secret bits tensor
        alpha_scale: Amplification factor for visualization (default 10)
    Returns:
        residual_tensor: The residual pattern [-1, 1]
    """
    with torch.no_grad():
        B, _, H, W = img.shape

        # Process secret → spatial tensor
        sec_feat = encoder.secret_preproc(secret).view(B, 8, 16, 16)
        sec_up = F.interpolate(sec_feat, size=(H//4, W//4), mode='bilinear', align_corners=False)

        # Encode image
        img_feat = encoder.encoder(img)

        # Fuse
        fused = torch.cat([img_feat, sec_up], dim=1)
        fused = encoder.fusion(fused)
        refined = fused + encoder.refine(fused)

        # Decode residual (this is the key part)
        residual = torch.tanh(encoder.decoder(refined))

        # Scale for visualization - make it more visible
        residual_scaled = residual * alpha_scale

        residual = residual.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        residual_scaled = residual_scaled.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        return residual, residual_scaled


# ==============================
# DATASET
# ==============================

print("\n📥 Preparing dataset...")

coco_path = "val2017"
if os.path.exists(coco_path):
    import glob
    img_paths = glob.glob(os.path.join(coco_path, "*.jpg"))[:1500]
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
print("TRAINING WITH IMPROVEMENTS")
print("="*70)
print("🔧 Changes:")
print("   1. Alpha: 0.15 → 0.5 (stronger residuals)")
print("   2. Differentiable blur (gradients flow to encoder)")
print("   3. Secret loss weight: 1.0 → 2.0")
print("="*70)

enc = ImprovedEncoder(MESSAGE_LEN).to(DEVICE)
dec = ImprovedDecoder(MESSAGE_LEN).to(DEVICE)

optimizer = torch.optim.Adam(
    list(enc.parameters()) + list(dec.parameters()),
    lr=1e-4,
    betas=(0.9, 0.999)
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000, eta_min=1e-5)

# Target secret (FIXED)
target_secret = ethereum_to_bits(SECRET_SHORT, MESSAGE_LEN)

# Create diverse training secrets
print("\n🎲 Creating training secret pool...")
train_secrets = []
train_secrets.append(target_secret)

for i in range(50):
    rand_hex = '0x' + ''.join([format(random.randint(0, 15), 'X') for _ in range(MESSAGE_LEN // 4)])
    train_secrets.append(ethereum_to_bits(rand_hex, MESSAGE_LEN))

print(f"✅ Secret pool: {len(train_secrets)} diverse Ethereum addresses")

all_bits = torch.stack(train_secrets).float()
ones_ratio = all_bits.mean().item()
print(f"📊 Bit balance in pool: {ones_ratio:.1%} ones, {1-ones_ratio:.1%} zeros")

print("\n🚀 Starting training with differentiable blur...")
print(f"   Target: '{SECRET_SHORT}'")
print(f"   Differentiable blur: σ=0.5-1.5")
print(f"   Expected time: ~25-35 minutes")
print(f"   Goal: >75% accuracy\n")

enc.train()
dec.train()
training_mode = True  # Enable random blur

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

        # CHANGE 3: Use differentiable blur (gradients flow!)
        attacked = differentiable_blur(watermarked)

        logits = dec(attacked)

        secret_loss = F.binary_cross_entropy_with_logits(logits, secrets)
        image_loss = F.mse_loss(watermarked, images)

        # CHANGE 4: Increased secret loss weight from 1.0 to 2.0
        loss = 2.0 * secret_loss + 0.3 * image_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # ===== GRADIENT DIAGNOSTIC (every 100 steps) =====
        if step % 100 == 0:
            # Check encoder gradients
            enc_grads = []
            for name, param in enc.named_parameters():
                if param.grad is not None:
                    enc_grads.append(param.grad.abs().mean().item())

            # Check decoder gradients
            dec_grads = []
            for name, param in dec.named_parameters():
                if param.grad is not None:
                    dec_grads.append(param.grad.abs().mean().item())

            enc_grad_mean = np.mean(enc_grads) if enc_grads else 0.0
            dec_grad_mean = np.mean(dec_grads) if dec_grads else 0.0

            print(f"\n🔍 Gradient Check at Step {step}:")
            print(f"   Encoder avg grad: {enc_grad_mean:.6f}")
            print(f"   Decoder avg grad: {dec_grad_mean:.6f}")
            print(f"   Ratio (enc/dec): {enc_grad_mean/(dec_grad_mean+1e-10):.4f}")

            # Check specific critical layers
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
                        print(f"  ⭐ New best: {best_target_acc:.1%}")

                training_mode = True  # Back to random blur

        step += 1

training_mode = False  # Done training
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

# ==============================
# ENHANCED VISUALIZATION
# Replace the old visualization section with this
# ==============================

# ==============================
# ENHANCED VISUALIZATION
# Replace the old visualization section with this
# ==============================

def to_pil(t):
    return T.ToPILImage()(((t[0]+1)/2).clamp(0,1).cpu())

# Extract residual for visualization
raw_residual, scaled_residual = extract_residual(enc, img_tensor, target_tensor, alpha_scale=15)

# ==============================
# ENHANCED VISUALIZATION
# Replace the old visualization section with this
# ==============================

def to_pil(t):
    return T.ToPILImage()(((t[0]+1)/2).clamp(0,1).cpu())

# Extract residual for visualization
raw_residual, scaled_residual = extract_residual(enc, img_tensor, target_tensor, alpha_scale=15)

# ==============================
# 1. ATTACK STRENGTH ANALYSIS
# ==============================
print("\n📊 Analyzing bit accuracy vs attack strength...")

# Test different blur strengths
blur_sigmas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
bit_accuracies = []

enc.eval()
dec.eval()

for sigma in blur_sigmas:
    if sigma == 0.0:
        # No attack
        attacked = watermarked
    else:
        # Apply blur with specific sigma
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

# ==============================
# 2. MAIN VISUALIZATION
# ==============================

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.2)

# Row 1: Main images (4 images)
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

# Row 2: Attack strength analysis graph (spanning 3 columns) + residual stats (1 column)

# Plot: Bit Accuracy vs Attack Strength
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

# Add value labels on points
for sigma, acc in zip(blur_sigmas, bit_accuracies):
    ax_graph.annotate(f'{acc:.1f}%',
                     xy=(sigma, acc),
                     xytext=(0, 10),
                     textcoords='offset points',
                     ha='center',
                     fontsize=8,
                     alpha=0.7)

# Residual statistics text
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

print(f"\n💾 Visualization saved as: watermark_analysis.png")

# ==============================
# 3. DETAILED CONSOLE OUTPUT
# ==============================

print(f"\n🎨 Residual Statistics:")
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

    print(f"\n📊 Residual Pipeline:")
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

print(f"\n✅ DONE!")
