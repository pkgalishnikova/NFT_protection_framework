!nvidia-smi

!pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
!pip install diffusers accelerate transformers pillow opencv-python-headless scikit-image
!pip install -q lpips

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

!mkdir -p checkpoints/StegaStamp
!mkdir -p example/input
!mkdir -p example/output

def reconstruction_loss(watermarked, original):
    return F.mse_loss(watermarked, original)

def secret_loss(pred_secret, true_secret):
    return F.binary_cross_entropy_with_logits(pred_secret, true_secret)

import torchvision.transforms.functional as TF
import random
from PIL import Image
from io import BytesIO
import torch
import torch.nn.functional as F

def distort_image(img):
    """
    Simulate print-photo pipeline corruptions.
    Input: img [B, 3, H, W] in [-1, 1]
    Output: distorted img in same range
    Device-agnostic: uses img.device
    """
    device = img.device
    B, C, H, W = img.shape

    img = (img + 1) / 2
    img = torch.clamp(img, 0.0, 1.0)

    if random.random() < 0.5:
        quality = random.randint(60, 95)
        pil_imgs = []
        for i in range(B):
            pil_img = TF.to_pil_image(img[i].cpu())  # move to CPU for PIL
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            pil_img = Image.open(buffer).convert("RGB")
            pil_imgs.append(pil_img)

        img = torch.stack([TF.to_tensor(p) for p in pil_imgs]).to(device)

    if random.random() < 0.7:
        scale = random.uniform(0.8, 1.2)
        new_h, new_w = int(H * scale), int(W * scale)
        img = TF.resize(img, [new_h, new_w], antialias=True)
        h_offset = max(0, (new_h - H) // 2)
        w_offset = max(0, (new_w - W) // 2)
        img = TF.crop(img, h_offset, w_offset, H, W)

    if random.random() < 0.6:
        noise_std = random.uniform(0.005, 0.02)
        img = img + torch.randn_like(img, device=device) * noise_std
        img = torch.clamp(img, 0.0, 1.0)

    if random.random() < 0.5:
        brightness_factor = random.uniform(0.9, 1.1)
        contrast_factor = random.uniform(0.9, 1.1)
        img = TF.adjust_brightness(img, brightness_factor)
        img = TF.adjust_contrast(img, contrast_factor)
        img = torch.clamp(img, 0.0, 1.0)

    img = img * 2 - 1
    return img

def train_step(encoder, decoder, optimizer, batch_size=8, secret_len=100):
    encoder.train()
    decoder.train()

    device = next(encoder.parameters()).device

    secrets = generate_random_message(batch_size, secret_len).to(device)
    clean_images = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    clean_images = torch.clamp(clean_images, -1.0, 1.0)

    watermarked = encoder(clean_images, secrets)

    distorted = distort_image(watermarked.detach())

    decoded_secrets = decoder(distorted)

    loss_secret = F.binary_cross_entropy_with_logits(decoded_secrets, secrets)
    loss_recon = F.mse_loss(watermarked, clean_images)
    total_loss = loss_secret + 0.01 * loss_recon

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    pred_bits = (torch.sigmoid(decoded_secrets) > 0.5).float()
    acc = (pred_bits == secrets).float().mean().item()

    return {
        'loss_total': total_loss.item(),
        'loss_secret': loss_secret.item(),
        'loss_recon': loss_recon.item(),
        'bit_acc': acc
    }

class StegaStampEncoder(nn.Module):
    """
    StegaStamp Encoder: Embeds secret message into images
    Based on: https://github.com/tancik/StegaStamp
    """
    def __init__(self, secret_len=100, height=256, width=256):
        super().__init__()
        self.secret_len = secret_len
        self.height = height
        self.width = width

        # Secret processing layers
        self.secret_dense = nn.Sequential(
            nn.Linear(secret_len, 7500),
            nn.ReLU(inplace=True)
        )

        # Image encoder - all with padding=1 to maintain dimensions
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # After concatenation with secret
        self.conv6 = nn.Conv2d(64 + 30, 64, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, image, secret):
        """
        Args:
            image: [B, 3, H, W] input image normalized to [-1, 1]
            secret: [B, secret_len] binary message {0, 1}
        Returns:
            watermarked_image: [B, 3, H, W] watermarked image
        """
        # Get actual image dimensions
        B, C, H, W = image.shape

        # Process secret
        secret_enlarged = self.secret_dense(secret)  # [B, 7500]
        secret_enlarged = secret_enlarged.view(B, 30, 50, 5)  # Use B instead of -1
        secret_enlarged = F.interpolate(
            secret_enlarged,
            size=(H, W),
            mode='nearest'
        )

        # Encode image
        x = self.relu(self.conv1(image))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        # Concatenate with secret
        x = torch.cat([x, secret_enlarged], dim=1)

        # Generate residual
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        residual = self.tanh(self.conv8(x))

        # Add residual to original image
        watermarked = image + residual

        # Clamp to valid range
        watermarked = torch.clamp(watermarked, -1.0, 1.0)

        return watermarked


class StegaStampDecoder(nn.Module):
    """
    StegaStamp Decoder: Extracts secret message from images
    """
    def __init__(self, secret_len=100, height=256, width=256):
        super().__init__()
        self.secret_len = secret_len
        self.height = height
        self.width = width

        self.decoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # 128x128
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64x64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 8x8
            nn.ReLU(inplace=True),
        )

        self.dense = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, secret_len)
        )

    def forward(self, image):
        """
        Args:
            image: [B, 3, H, W] watermarked image
        Returns:
            secret_pred: [B, secret_len] predicted secret (logits)
        """
        x = self.decoder(image)
        x = x.view(x.size(0), -1)
        secret_pred = self.dense(x)
        return secret_pred
    
def generate_random_message(batch_size, secret_len):
    return torch.randint(0, 2, (batch_size, secret_len), dtype=torch.float32)

def tensor_to_pil(tensor):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)

    np_img = tensor.cpu().permute(1, 2, 0).numpy()
    np_img = (np_img * 255).astype(np.uint8)
    return Image.fromarray(np_img)

def pil_to_tensor(pil_img, device='cuda'):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(pil_img).unsqueeze(0).to(device)

def calculate_bit_accuracy(pred_bits, true_bits):
    return (pred_bits == true_bits).float().mean().item()

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(2.0 / torch.sqrt(mse))

def visualize_results(original, watermarked, message, title="Results"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(watermarked)
    axes[1].set_title(f"Watermarked Image")
    axes[1].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SECRET_LEN = 100
IMAGE_SIZE = 256

encoder = StegaStampEncoder(
    secret_len=SECRET_LEN,
    height=IMAGE_SIZE,
    width=IMAGE_SIZE
).to(device)

decoder = StegaStampDecoder(
    secret_len=SECRET_LEN,
    height=IMAGE_SIZE,
    width=IMAGE_SIZE
).to(device)

checkpoint_path = "./checkpoints/StegaStamp/stegastamp_checkpoint.pth"
if os.path.exists(checkpoint_path):
    print("Loading pretrained StegaStamp models...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    print("Models loaded successfully!")
else:
    print("No pretrained models found. Using random initialization.")

encoder.eval()
decoder.eval()

print(f"\nModel Statistics:")
print(f"   Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
print(f"   Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")

params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=1e-4)

num_epochs = 5
steps_per_epoch = 50

print("Starting training...")

for epoch in range(num_epochs):
    encoder.train(); decoder.train()
    pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{num_epochs}")
    epoch_losses = []
    epoch_accs = []

    for _ in pbar:
        res = train_step(encoder, decoder, optimizer, batch_size=8)
        epoch_losses.append(res['loss_total'])
        epoch_accs.append(res['bit_acc'])
        pbar.set_postfix({
            'loss': f"{np.mean(epoch_losses[-10:]):.4f}",
            'acc': f"{np.mean(epoch_accs[-10:]):.3f}"
        })

    avg_acc = np.mean(epoch_accs)
    print(f"Epoch {epoch+1}: Avg Bit Acc = {avg_acc:.3%}")

    if (epoch + 1) % 10 == 0 or avg_acc > 0.95:
        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, f"./checkpoints/StegaStamp/stegastamp_epoch{epoch+1}_acc{avg_acc:.3f}.pth")

encoder.eval()
decoder.eval()

input_path = "example/input/0.jpg"
if os.path.exists(input_path):
    input_img_pil = Image.open(input_path).convert("RGB")
    input_img_pil = input_img_pil.resize((256, 256))
    img_tensor = pil_to_tensor(input_img_pil, device=device)  # reuses your helper

    secret = generate_random_message(1, SECRET_LEN).to(device)

    with torch.no_grad():
        wm = encoder(img_tensor, secret)
        wm_dist = distort_image(wm)  # ← crucial for fair eval!
        decoded = decoder(wm_dist)
        pred = (torch.sigmoid(decoded) > 0.5).float()
        acc = calculate_bit_accuracy(pred, secret)

    print(f"Decoded bit accuracy (after distortion): {acc:.2%}")

    orig_pil = tensor_to_pil(img_tensor)
    wm_pil = tensor_to_pil(wm)
    visualize_results(orig_pil, wm_pil, secret, title=f"Acc: {acc:.2%}")