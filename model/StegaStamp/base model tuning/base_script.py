import os
import sys
import random
import warnings

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
print(f"Device: {DEVICE}")

SECRET_STR = "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D"
print(f"Full address: {SECRET_STR}")

SECRET_SHORT = SECRET_STR[:12]
MESSAGE_LEN = 100

print(f"Using SHORT version: '{SECRET_SHORT}' ({MESSAGE_LEN} bits)")

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse < 1e-10:
        return 100.0
    max_pixel = 2.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def calculate_tpr(predictions, targets, threshold=0.5):
    pred_bits = (predictions > threshold).float()
    target_bits = targets.float()
    tp = ((pred_bits == 1) & (target_bits == 1)).float().sum()
    fn = ((pred_bits == 0) & (target_bits == 1)).float().sum()
    tpr = tp / (tp + fn + 1e-10)
    return tpr.item()

def load_clip_model(device='cuda'):
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

CLIP_MODEL, CLIP_PREPROCESS = None, None

def init_clip():
    global CLIP_MODEL, CLIP_PREPROCESS
    if CLIP_MODEL is None:
        CLIP_MODEL, CLIP_PREPROCESS = load_clip_model(DEVICE)

def tensor_to_pil(img_tensor):
    img = (img_tensor + 1) / 2
    img = torch.clamp(img, 0, 1)
    return T.ToPILImage()(img.cpu())

def calculate_clip_metrics(original_img, watermarked_img, secret_text):
    init_clip()

    orig_pil = tensor_to_pil(original_img[0])
    wm_pil = tensor_to_pil(watermarked_img[0])

    orig_clip = CLIP_PREPROCESS(orig_pil).unsqueeze(0).to(DEVICE)
    wm_clip = CLIP_PREPROCESS(wm_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        orig_features = CLIP_MODEL.encode_image(orig_clip)
        wm_features = CLIP_MODEL.encode_image(wm_clip)

        text_tokens = clip.tokenize([secret_text]).to(DEVICE)
        text_features = CLIP_MODEL.encode_text(text_tokens)

    orig_features = orig_features / orig_features.norm(dim=-1, keepdim=True)
    wm_features = wm_features / wm_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    clip_img = (orig_features @ wm_features.T).item()
    clip_out = (wm_features @ text_features.T).item()

    orig_to_wm = wm_features - orig_features
    orig_to_text = text_features - orig_features

    orig_to_wm = orig_to_wm / orig_to_wm.norm(dim=-1, keepdim=True)
    orig_to_text = orig_to_text / orig_to_text.norm(dim=-1, keepdim=True)

    clip_dir = (orig_to_wm @ orig_to_text.T).item()

    return {
        'CLIPimg': clip_img,
        'CLIPout': clip_out,
        'CLIPdir': clip_dir
    }

def calculate_mse(original_img, watermarked_img):
    return F.mse_loss(original_img, watermarked_img).item()

def calculate_asr_and_ear(decoder, watermarked_img, target_secret, threshold=0.5, num_trials=10):
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
        if (
            addr.startswith('0x') and
            len(addr) == len(target_str) and
            addr != target_str
        ):
            valid_but_wrong += 1

    ear = valid_but_wrong / num_trials

    return asr, ear, recovered_secrets

class ImprovedEncoder(nn.Module):
    def __init__(self, secret_len=100):
        super().__init__()
        self.secret_len = secret_len

        self.secret_preproc = nn.Sequential(
            nn.Linear(secret_len, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8 * 16 * 16),
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

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

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 3, 3, 1, 1),
        )

        self.alpha = nn.Parameter(torch.tensor(0.15))

    def forward(self, img, secret):
        B, _, H, W = img.shape

        sec_feat = self.secret_preproc(secret).view(B, 8, 16, 16)

        sec_up = F.interpolate(
            sec_feat,
            size=(H // 4, W // 4),
            mode='bilinear',
            align_corners=False
        )

        img_feat = self.encoder(img)

        fused = torch.cat([img_feat, sec_up], dim=1)
        fused = self.fusion(fused)

        refined = fused + self.refine(fused)

        residual = torch.tanh(self.decoder(refined))

        watermark = img + torch.tanh(self.alpha) * residual

        return torch.clamp(watermark, -1, 1)

class ImprovedDecoder(nn.Module):
    def __init__(self, secret_len=100):
        super().__init__()
        self.secret_len = secret_len

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.psp_pool = nn.AdaptiveAvgPool2d(1)

        self.psp_conv = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.ReLU(),
        )

        self.attn = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.ReLU(),

            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),

            nn.Linear(256 + 64, 512),
            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, secret_len),
        )

    def forward(self, img):
        x = self.stem(img)
        x = self.stage1(x)
        feat = self.stage2(x)

        global_feat = self.psp_pool(feat)

        pooled = self.psp_conv(global_feat).squeeze(-1).squeeze(-1)

        attn_map = self.attn(feat)

        attended = (feat * attn_map).mean(dim=[2, 3])

        combined = torch.cat([attended, pooled], dim=1)

        return self.classifier(combined)

def differentiable_blur(img, sigma_range=(0.0, 2.0)):
    B, C, H, W = img.shape

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

def simple_attack(img):
    B, C, H, W = img.shape

    imgs_out = []

    for i in range(B):
        pil = T.ToPILImage()(
            ((img[i] + 1) / 2).clamp(0, 1).cpu()
        )

        img_tensor = T.ToTensor()(pil)

        blur = T.GaussianBlur(kernel_size=5, sigma=1.0)

        img_tensor = blur(img_tensor)

        imgs_out.append(img_tensor)

    imgs_out = torch.stack(imgs_out).to(img.device)

    return imgs_out * 2 - 1

training_mode = False

def ethereum_to_bits(address, num_bits=100):
    if address.startswith('0x') or address.startswith('0X'):
        address = address[2:]

    num_hex_chars = num_bits // 4

    address_part = address[:num_hex_chars].upper()

    binary_str = bin(int(address_part, 16))[2:].zfill(num_bits)

    bits = torch.tensor(
        [int(b) for b in binary_str],
        dtype=torch.float32
    )

    return bits

def bits_to_ethereum(bits, num_bits=100, original_secret=SECRET_SHORT):
    bits_np = (bits[:num_bits] > 0.5).cpu().numpy().astype(np.uint8)

    binary_str = ''.join([str(int(b)) for b in bits_np])

    try:
        hex_value = hex(int(binary_str, 2))[2:].upper()

        num_hex_chars = num_bits // 4

        hex_value = hex_value.zfill(num_hex_chars)

        original_hex_len = len(original_secret) - 2

        return '0x' + hex_value[:original_hex_len]

    except:
        original_hex_len = len(original_secret) - 2
        return "0x" + "?" * original_hex_len

def extract_residual(encoder, img, secret, alpha_scale=10):
    with torch.no_grad():
        B, _, H, W = img.shape

        sec_feat = encoder.secret_preproc(secret).view(B, 8, 16, 16)

        sec_up = F.interpolate(
            sec_feat,
            size=(H // 4, W // 4),
            mode='bilinear',
            align_corners=False
        )

        img_feat = encoder.encoder(img)

        fused = torch.cat([img_feat, sec_up], dim=1)

        fused = encoder.fusion(fused)

        refined = fused + encoder.refine(fused)

        residual = torch.tanh(encoder.decoder(refined))

        residual_scaled = residual * alpha_scale

        residual = residual.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        residual_scaled = residual_scaled.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        return residual, residual_scaled
