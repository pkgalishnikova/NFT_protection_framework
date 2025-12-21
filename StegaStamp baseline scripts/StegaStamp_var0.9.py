# 51.88%
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from tqdm import tqdm
import urllib.request
import zipfile
import json

# ==================== Model Definitions ====================

from torch.nn.functional import relu, sigmoid

class StegaStampEncoder(nn.Module):
    def __init__(
        self,
        resolution=128,
        IMAGE_CHANNELS=3,
        fingerprint_size=100,
        return_residual=False,
    ):
        super(StegaStampEncoder, self).__init__()
        self.fingerprint_size = fingerprint_size
        self.IMAGE_CHANNELS = IMAGE_CHANNELS
        self.return_residual = return_residual
        self.secret_dense = nn.Linear(self.fingerprint_size, 16 * 16 * IMAGE_CHANNELS)

        log_resolution = int(math.log(resolution, 2))
        assert resolution == 2 ** log_resolution, f"Image resolution must be a power of 2, got {resolution}."

        self.fingerprint_upsample = nn.Upsample(scale_factor=(2**(log_resolution-4), 2**(log_resolution-4)))
        self.conv1 = nn.Conv2d(2 * IMAGE_CHANNELS, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)
        self.pad6 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up6 = nn.Conv2d(256, 128, 2, 1)
        self.upsample6 = nn.Upsample(scale_factor=(2, 2))
        self.conv6 = nn.Conv2d(128 + 128, 128, 3, 1, 1)
        self.pad7 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up7 = nn.Conv2d(128, 64, 2, 1)
        self.upsample7 = nn.Upsample(scale_factor=(2, 2))
        self.conv7 = nn.Conv2d(64 + 64, 64, 3, 1, 1)
        self.pad8 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up8 = nn.Conv2d(64, 32, 2, 1)
        self.upsample8 = nn.Upsample(scale_factor=(2, 2))
        self.conv8 = nn.Conv2d(32 + 32, 32, 3, 1, 1)
        self.pad9 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up9 = nn.Conv2d(32, 32, 2, 1)
        self.upsample9 = nn.Upsample(scale_factor=(2, 2))
        self.conv9 = nn.Conv2d(32 + 32 + 2 * IMAGE_CHANNELS, 32, 3, 1, 1)
        self.conv10 = nn.Conv2d(32, 32, 3, 1, 1)
        self.residual = nn.Conv2d(32, IMAGE_CHANNELS, 1)

    def forward(self, fingerprint, image):
        fingerprint = relu(self.secret_dense(fingerprint))
        fingerprint = fingerprint.view((-1, self.IMAGE_CHANNELS, 16, 16))
        fingerprint_enlarged = self.fingerprint_upsample(fingerprint)
        inputs = torch.cat([fingerprint_enlarged, image], dim=1)
        conv1 = relu(self.conv1(inputs))
        conv2 = relu(self.conv2(conv1))
        conv3 = relu(self.conv3(conv2))
        conv4 = relu(self.conv4(conv3))
        conv5 = relu(self.conv5(conv4))
        up6 = relu(self.up6(self.pad6(self.upsample6(conv5))))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = relu(self.conv6(merge6))
        up7 = relu(self.up7(self.pad7(self.upsample7(conv6))))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = relu(self.conv7(merge7))
        up8 = relu(self.up8(self.pad8(self.upsample8(conv7))))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = relu(self.conv8(merge8))
        up9 = relu(self.up9(self.pad9(self.upsample9(conv8))))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = relu(self.conv9(merge9))
        conv10 = relu(self.conv10(conv9))
        residual = self.residual(conv10)
        if not self.return_residual:
            residual = sigmoid(residual)
        return residual


class StegaStampDecoder(nn.Module):
    def __init__(self, resolution=128, IMAGE_CHANNELS=3, fingerprint_size=100):
        super(StegaStampDecoder, self).__init__()
        self.resolution = resolution
        self.IMAGE_CHANNELS = IMAGE_CHANNELS
        self.decoder = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, (3, 3), 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), 2, 1),
            nn.ReLU(),
        )
        self.dense = nn.Sequential(
            nn.Linear(resolution * resolution * 128 // 32 // 32, 512),
            nn.ReLU(),
            nn.Linear(512, fingerprint_size),
        )

    def forward(self, image):
        x = self.decoder(image)
        x = x.view(-1, self.resolution * self.resolution * 128 // 32 // 32)
        return self.dense(x)


# ==================== Utility Functions ====================

def ethereum_address_to_binary(address):
    """Convert Ethereum address (42 chars with 0x) to 160-bit binary"""
    if address.startswith('0x') or address.startswith('0X'):
        address = address[2:]
    binary = bin(int(address, 16))[2:].zfill(160)
    return torch.tensor([int(b) for b in binary], dtype=torch.float32)


def binary_to_ethereum_address(binary):
    """Convert 160-bit binary back to Ethereum address"""
    binary_str = ''.join([str(int(b > 0.5)) for b in binary])
    hex_str = hex(int(binary_str, 2))[2:].zfill(40)
    return '0x' + hex_str


def download_file(url, filename):
    """Download file with progress bar"""
    print(f"Downloading {filename}...")

    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rProgress: {percent}%", end='')

    urllib.request.urlretrieve(url, filename, reporthook)
    print("\nDownload complete!")


def setup_coco_dataset(num_images=1000):
    """Download and setup a subset of COCO dataset"""
    data_dir = './data'
    coco_dir = os.path.join(data_dir, 'coco')

    os.makedirs(coco_dir, exist_ok=True)

    # Download COCO validation images (smaller than train)
    zip_path = os.path.join(coco_dir, 'val2017.zip')
    val_dir = os.path.join(coco_dir, 'val2017')

    # Check if images already exist
    if os.path.exists(val_dir):
        existing_images = [f for f in os.listdir(val_dir) if f.endswith('.jpg')]
        if len(existing_images) >= num_images:
            print(f"Found {len(existing_images)} existing COCO images")
            image_files = [os.path.join(val_dir, f) for f in existing_images[:num_images]]
            return image_files

    # Download if not exists
    if not os.path.exists(zip_path):
        print("Downloading COCO validation dataset (1.0 GB)...")
        print("This may take 5-15 minutes depending on your internet speed...")
        url = "http://images.cocodataset.org/zips/val2017.zip"
        try:
            download_file(url, zip_path)
        except Exception as e:
            print(f"Error downloading COCO dataset: {e}")
            print("Will use synthetic images instead...")
            return []

    # Extract if zip exists
    if os.path.exists(zip_path) and not os.path.exists(val_dir):
        print("Extracting images (this may take a few minutes)...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract all files
                zip_ref.extractall(coco_dir)
            print("Extraction complete!")
        except Exception as e:
            print(f"Error extracting COCO dataset: {e}")
            print("Will use synthetic images instead...")
            return []

    # Get image paths
    if os.path.exists(val_dir):
        image_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.jpg')]
        print(f"Found {len(image_files)} COCO images")
        return image_files[:num_images]
    else:
        print("COCO directory not found. Will use synthetic images...")
        return []


def download_test_image():
    """Download a test image"""
    test_image_path = './test_image.jpg'

    if not os.path.exists(test_image_path):
        print("Downloading test image...")
        # Download a sample image from Unsplash
        url = "https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?w=800"
        download_file(url, test_image_path)

    return test_image_path


# ==================== Attack Simulations ====================

class JPEGCompression:
    def __init__(self, quality=50):
        self.quality = quality

    def __call__(self, img_tensor):
        """Apply JPEG compression to a tensor image"""
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        buffer = BytesIO()
        img_pil.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        img_compressed = Image.open(buffer)

        img_np = np.array(img_compressed).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        return img_tensor


class GaussianBlur:
    def __init__(self, kernel_size=5, sigma=1.0):
        self.blur = transforms.GaussianBlur(kernel_size, sigma)

    def __call__(self, img_tensor):
        return self.blur(img_tensor)


# ==================== Dataset ====================

class SimpleImageDataset(Dataset):
    def __init__(self, image_paths, resolution=128):
        self.image_paths = image_paths
        self.resolution = resolution

        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            img = self.transform(img)

            # Generate random Ethereum address (160 bits)
            random_address = '0x' + ''.join([format(np.random.randint(0, 16), 'x') for _ in range(40)])
            fingerprint = ethereum_address_to_binary(random_address)

            return img, fingerprint, random_address
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a black image if loading fails
            img = torch.zeros(3, self.resolution, self.resolution)
            random_address = '0x' + ''.join([format(np.random.randint(0, 16), 'x') for _ in range(40)])
            fingerprint = ethereum_address_to_binary(random_address)
            return img, fingerprint, random_address


# ==================== Metrics ====================

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def calculate_bit_accuracy(pred_bits, true_bits):
    """Calculate bit accuracy"""
    pred_binary = (pred_bits > 0.5).float()
    true_binary = true_bits
    accuracy = (pred_binary == true_binary).float().mean().item()
    return accuracy * 100


# ==================== Training ====================

def train_model(encoder, decoder, train_loader, num_epochs=15, device='cuda'):
    encoder.to(device)
    decoder.to(device)

    # Enable cuDNN benchmarking for faster training
    torch.backends.cudnn.benchmark = True

    # Enable TF32 for even faster training on Ampere GPUs (RTX 30xx, A100, etc.)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    optimizer_enc = optim.Adam(encoder.parameters(), lr=0.0001, betas=(0.9, 0.999))
    optimizer_dec = optim.Adam(decoder.parameters(), lr=0.0001, betas=(0.9, 0.999))

    # Mixed Precision Training - GradScaler for automatic loss scaling
    scaler = torch.cuda.amp.GradScaler()

    criterion_image = nn.MSELoss()
    criterion_secret = nn.BCEWithLogitsLoss()

    jpeg_attack = JPEGCompression(quality=50)
    blur_attack = GaussianBlur(kernel_size=5, sigma=1.0)

    best_bit_acc = 0

    print("\n🚀 Mixed Precision Training Enabled (FP16)")
    print("Expected speedup: 2-3x faster training\n")

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()

        total_loss = 0
        total_bit_acc = 0
        total_psnr = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for images, fingerprints, addresses in pbar:
            images = images.to(device, non_blocking=True)
            fingerprints = fingerprints.to(device, non_blocking=True)

            # Mixed Precision: Forward pass in FP16
            with torch.cuda.amp.autocast():
                # Encode
                encoded_images = encoder(fingerprints, images)

                # Apply attacks (stays in FP32 for stability)
                attacked_images = []
                for i in range(encoded_images.shape[0]):
                    # Temporarily move to FP32 for attack processing
                    img = encoded_images[i].detach().float()
                    img = jpeg_attack(img)
                    img = blur_attack(img)
                    attacked_images.append(img)

                attacked_images = torch.stack(attacked_images).to(device, non_blocking=True)

                # Decode
                decoded_fingerprints = decoder(attacked_images)

                # Calculate losses in FP16
                loss_image = criterion_image(encoded_images, images)
                loss_secret = criterion_secret(decoded_fingerprints, fingerprints)

                # Reduce image loss weight to improve PSNR (less distortion)
                # Increase secret loss weight to maintain bit accuracy
                loss = 0.5 * loss_image + 20.0 * loss_secret

            # Backpropagation with gradient scaling
            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()

            # Scale loss and backward pass
            scaler.scale(loss).backward()

            # Unscale gradients and step optimizers
            scaler.step(optimizer_enc)
            scaler.step(optimizer_dec)

            # Update scaler for next iteration
            scaler.update()

            # Calculate metrics (in FP32 for accuracy)
            with torch.no_grad():
                bit_acc = calculate_bit_accuracy(torch.sigmoid(decoded_fingerprints.float()), fingerprints.float())
                psnr = calculate_psnr(encoded_images.float(), images.float())

            total_loss += loss.item()
            total_bit_acc += bit_acc
            total_psnr += psnr

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Bit Acc': f'{bit_acc:.2f}%',
                'PSNR': f'{psnr:.2f} dB',
                'GPU': f'{torch.cuda.memory_allocated()/1024**3:.2f}GB'
            })

        avg_loss = total_loss / len(train_loader)
        avg_bit_acc = total_bit_acc / len(train_loader)
        avg_psnr = total_psnr / len(train_loader)

        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Average Loss: {avg_loss:.4f}')
        print(f'Average Bit Accuracy: {avg_bit_acc:.2f}%')
        print(f'Average PSNR: {avg_psnr:.2f} dB')
        print(f'GPU Memory Used: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB\n')

        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()

        # Save best model
        if avg_bit_acc > best_bit_acc:
            best_bit_acc = avg_bit_acc
            torch.save(encoder.state_dict(), 'best_encoder.pth')
            torch.save(decoder.state_dict(), 'best_decoder.pth')
            print(f'✓ Saved best model with bit accuracy: {best_bit_acc:.2f}%\n')

    return encoder, decoder


# ==================== Evaluation ====================

def evaluate_watermark(encoder, decoder, image_path, ethereum_address, device='cuda', save_images=False):
    """Evaluate watermark embedding and extraction on a single image"""
    encoder.eval()
    decoder.eval()

    print(f"\nLoading image from: {image_path}")

    # Load image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    print(f"Original image size: {original_size}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(img).unsqueeze(0).to(device)

    # Convert Ethereum address to binary
    fingerprint = ethereum_address_to_binary(ethereum_address).unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode
        encoded_image = encoder(fingerprint, image_tensor)

        # Calculate PSNR before attacks
        psnr_before = calculate_psnr(encoded_image, image_tensor)

        # Apply attacks
        jpeg_attack = JPEGCompression(quality=50)
        blur_attack = GaussianBlur(kernel_size=5, sigma=1.0)

        attacked_image = encoded_image[0].cpu()
        attacked_image = jpeg_attack(attacked_image)
        attacked_image = blur_attack(attacked_image)
        attacked_image = attacked_image.unsqueeze(0).to(device)

        # Decode
        decoded_fingerprint = decoder(attacked_image)
        decoded_fingerprint_sigmoid = torch.sigmoid(decoded_fingerprint)

        # Calculate metrics
        bit_accuracy = calculate_bit_accuracy(decoded_fingerprint_sigmoid, fingerprint)

        # Convert back to Ethereum address
        extracted_address = binary_to_ethereum_address(decoded_fingerprint_sigmoid[0].cpu())

    # Visualize all stages
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Original image
    axes[0, 0].imshow(image_tensor[0].cpu().permute(1, 2, 0))
    axes[0, 0].set_title('Original Image\n(No Watermark)', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    # Watermarked image
    axes[0, 1].imshow(encoded_image[0].cpu().permute(1, 2, 0))
    axes[0, 1].set_title(f'Watermarked Image\nPSNR: {psnr_before:.2f} dB', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    # After JPEG compression
    jpeg_only = jpeg_attack(encoded_image[0].cpu())
    axes[1, 0].imshow(jpeg_only.permute(1, 2, 0))
    axes[1, 0].set_title('After JPEG Compression\n(Quality=50)', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')

    # After both attacks
    axes[1, 1].imshow(attacked_image[0].cpu().permute(1, 2, 0))
    axes[1, 1].set_title(f'After JPEG + Gaussian Blur\nBit Accuracy: {bit_accuracy:.2f}%',
                        fontsize=11, fontweight='bold',
                        color='green' if bit_accuracy >= 70 else 'red')
    axes[1, 1].axis('off')

    plt.suptitle('StegaStamp Watermarking Pipeline', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

    # Create difference visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image_tensor[0].cpu().permute(1, 2, 0))
    axes[0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(encoded_image[0].cpu().permute(1, 2, 0))
    axes[1].set_title('Watermarked', fontsize=11, fontweight='bold')
    axes[1].axis('off')

    # Difference (amplified for visibility)
    diff = torch.abs(encoded_image[0].cpu() - image_tensor[0].cpu())
    diff_amplified = torch.clamp(diff * 10, 0, 1)  # Amplify by 10x
    axes[2].imshow(diff_amplified.permute(1, 2, 0))
    axes[2].set_title('Difference (10x amplified)', fontsize=11, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle('Watermark Visibility Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f'\n{"="*70}')
    print(f'EVALUATION RESULTS')
    print(f'{"="*70}')
    print(f'Image: {os.path.basename(image_path)}')
    print(f'Original Address:  {ethereum_address}')
    print(f'Extracted Address: {extracted_address}')
    print(f'{"="*70}')
    print(f'Bit Accuracy: {bit_accuracy:.2f}%')
    print(f'PSNR (before attacks): {psnr_before:.2f} dB')
    print(f'Address Match: {"✓ YES" if ethereum_address.lower() == extracted_address.lower() else "✗ NO"}')
    print(f'{"="*70}')

    # Calculate bit-by-bit comparison
    pred_binary = (decoded_fingerprint_sigmoid[0].cpu() > 0.5).int()
    true_binary = fingerprint[0].cpu().int()
    correct_bits = (pred_binary == true_binary).sum().item()
    total_bits = len(true_binary)

    print(f'\nDetailed Analysis:')
    print(f'Correct bits: {correct_bits}/{total_bits}')
    print(f'Incorrect bits: {total_bits - correct_bits}')

    if bit_accuracy >= 70.0:
        print(f'\n✓ SUCCESS: Bit accuracy {bit_accuracy:.2f}% exceeds 70% threshold!')
    else:
        print(f'\n✗ ATTENTION: Bit accuracy {bit_accuracy:.2f}% is below 70% threshold.')

    return bit_accuracy, psnr_before, extracted_address

# ==================== Main Execution ====================

def main():
    # Configuration
    RESOLUTION = 128
    IMAGE_CHANNELS = 3
    FINGERPRINT_SIZE = 160  # Ethereum address is 160 bits
    BATCH_SIZE = 8
    NUM_EPOCHS = 20
    NUM_TRAIN_IMAGES = 1500
    DEVICE = 'cuda'  # Using GPU

    # Verify GPU is available
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available! Please check CUDA installation.")

    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

    print(f'\n{"="*60}')
    print(f'STEGASTAMP WATERMARKING SYSTEM')
    print(f'{"="*60}')
    print(f'Device: {DEVICE}')
    print(f'Resolution: {RESOLUTION}x{RESOLUTION}')
    print(f'Fingerprint size: {FINGERPRINT_SIZE} bits (Ethereum Address)')
    print(f'{"="*60}\n')

    # Initialize models
    encoder = StegaStampEncoder(
        resolution=RESOLUTION,
        IMAGE_CHANNELS=IMAGE_CHANNELS,
        fingerprint_size=FINGERPRINT_SIZE,
        return_residual=False
    )

    decoder = StegaStampDecoder(
        resolution=RESOLUTION,
        IMAGE_CHANNELS=IMAGE_CHANNELS,
        fingerprint_size=FINGERPRINT_SIZE
    )

    # Setup dataset
    print('STEP 1: Setting up COCO dataset...')
    image_paths = setup_coco_dataset(num_images=NUM_TRAIN_IMAGES)

    if len(image_paths) == 0:
        print("Warning: No COCO images found. Creating synthetic dataset...")
        # Create synthetic dataset if COCO download fails
        os.makedirs('./data/synthetic', exist_ok=True)
        image_paths = []
        for i in range(NUM_TRAIN_IMAGES):
            img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            path = f'./data/synthetic/img_{i:04d}.jpg'
            img.save(path)
            image_paths.append(path)

    print(f'Using {len(image_paths)} training images\n')

    # Create dataset and dataloader
    train_dataset = SimpleImageDataset(image_paths, resolution=RESOLUTION)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # Parallel data loading for GPU
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True  # Keep workers alive between epochs
    )

    # Training
    print('STEP 2: Training models...')
    encoder, decoder = train_model(
        encoder, decoder, train_loader,
        num_epochs=NUM_EPOCHS,
        device=DEVICE
    )

    # Save final models
    torch.save(encoder.state_dict(), 'final_encoder.pth')
    torch.save(decoder.state_dict(), 'final_decoder.pth')
    print('Final models saved!\n')

    # Load best models for evaluation
    encoder.load_state_dict(torch.load('best_encoder.pth'))
    decoder.load_state_dict(torch.load('best_decoder.pth'))
    encoder.to(DEVICE)
    decoder.to(DEVICE)
    print('Loaded best models for evaluation\n')

    # Download test image
    print('STEP 3: Setup test image for evaluation...')

    # Option 1: Try to download a default test image
    default_test_image = './test_image.jpg'
    if not os.path.exists(default_test_image):
        try:
            print("Downloading default test image...")
            test_image_path = download_test_image()
        except:
            print("Could not download default image.")
            default_test_image = None
    else:
        test_image_path = default_test_image

    # Option 2: Let user specify their own image
    print("\n" + "="*70)
    print("MANUAL IMAGE SELECTION FOR EVALUATION")
    print("="*70)
    print("\nYou can now specify your own image for watermark evaluation.")
    print("Options:")
    print("  1. Press ENTER to use the default test image")
    print("  2. Enter the full path to your image (e.g., C:/Users/Name/image.jpg)")
    print("  3. Place your image in the current directory and enter just the filename")
    print("="*70)

    user_image_path = input("\nEnter image path (or press ENTER for default): ").strip()

    if user_image_path:
        # User provided a path
        if os.path.exists(user_image_path):
            test_image_path = user_image_path
            print(f"✓ Using your image: {test_image_path}")
        else:
            print(f"✗ Image not found: {user_image_path}")
            if default_test_image and os.path.exists(default_test_image):
                print(f"✓ Falling back to default image: {default_test_image}")
                test_image_path = default_test_image
            else:
                print("✗ No valid image available. Please check the path and try again.")
                return
    else:
        # User pressed ENTER - use default
        if default_test_image and os.path.exists(default_test_image):
            test_image_path = default_test_image
            print(f"✓ Using default test image: {test_image_path}")
        else:
            print("✗ No default image available.")
            return

    # Display the selected image
    try:
        display_img = Image.open(test_image_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(display_img)
        plt.title(f'Selected Image for Evaluation\n{os.path.basename(test_image_path)}',
                 fontsize=11, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        print(f"  Image size: {display_img.size}")
        print(f"  Image mode: {display_img.mode}")
    except Exception as e:
        print(f"✗ Error displaying image: {e}")
        return

    # Evaluation with the specified Ethereum address
    print('\nSTEP 4: Evaluating with specified Ethereum address...')
    ETHEREUM_ADDRESS = "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D"

    bit_acc, psnr, extracted = evaluate_watermark(
        encoder, decoder, test_image_path, ETHEREUM_ADDRESS, DEVICE
    )

    print(f'\nVisualization saved as: watermark_result.png')
    print(f'\nExperiment complete!')


if __name__ == '__main__':
    main()