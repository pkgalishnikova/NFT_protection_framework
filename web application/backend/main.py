import io
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms as T
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from reedsolo import RSCodec
import requests as http_requests
import kornia.filters as KF
import kornia.geometry.transform as KGT

load_dotenv()
PINATA_JWT = os.getenv("PINATA_JWT")

REAL_BYTES   = 3
PARITY_BYTES = 1
MESSAGE_LEN  = (REAL_BYTES + PARITY_BYTES) * 8
rsc = RSCodec(PARITY_BYTES)
IMG_SIZE = 256
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "best_clean.pth"

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
        return self.final(out)

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
        return self.linear(x)

def address_to_bits(wallet_address: str) -> torch.Tensor:
    hex_str = wallet_address[2:2 + REAL_BYTES * 2]
    raw_bytes = bytes.fromhex(hex_str)
    encoded = rsc.encode(raw_bytes)
    bits = []
    for byte in encoded:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return torch.tensor(bits, dtype=torch.float32)

def bits_to_address_prefix(bits_tensor: torch.Tensor) -> str:
    bits = (bits_tensor[:MESSAGE_LEN] > 0.5).int().tolist()
    byte_vals = []
    for i in range(0, MESSAGE_LEN, 8):
        byte_vals.append(int("".join(str(b) for b in bits[i:i+8]), 2))
    try:
        decoded, _, _ = rsc.decode(bytes(byte_vals))
        return "0x" + bytes(decoded).hex().upper()
    except Exception:
        return "0xERR"

def preprocess(image: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    img = (tensor.squeeze(0).detach().cpu() + 1) / 2
    img = img.clamp(0, 1)
    return T.ToPILImage()(img)

print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
enc = Encoder(MESSAGE_LEN, channels=64, num_blocks=6).to(DEVICE)
dec = Decoder(MESSAGE_LEN, channels=64, num_blocks=8).to(DEVICE)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
enc.load_state_dict(ckpt["enc"])
dec.load_state_dict(ckpt["dec"])
enc.eval()
dec.eval()
print("Model loaded successfully.")

def try_decode_variants(img_tensor: torch.Tensor):
    candidates = []

    def confidence(logits):
        return (logits - 0.5).abs().mean().item()

    with torch.no_grad():
        logits = dec(img_tensor)
        candidates.append((logits, confidence(logits)))

        kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ], dtype=torch.float32, device=DEVICE).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        sharpened = F.conv2d(img_tensor, kernel, padding=1, groups=3).clamp(-1, 1)
        logits = dec(sharpened)
        candidates.append((logits, confidence(logits)))

        for angle in [-15, -10, -5, 5, 10, 15]:
            angle_t = torch.tensor([float(angle)], device=DEVICE)
            rotated = KGT.rotate(img_tensor, angle_t)
            logits = dec(rotated)
            candidates.append((logits, confidence(logits)))

        votes = []
        for _ in range(11):
            noise = torch.randn_like(img_tensor) * 0.02
            votes.append(dec(img_tensor + noise))
        voted_logits = torch.stack(votes).mean(dim=0)
        candidates.append((voted_logits, confidence(voted_logits)))

    best_logits, best_conf = max(candidates, key=lambda x: x[1])
    return (best_logits > 0.5).float().squeeze(0), best_conf

app = FastAPI(title="NFT Watermark API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/embed")
async def embed(
    image: UploadFile = File(...),
    wallet_address: str = Form(...),
):
    raw = await image.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    original_size = pil_img.size

    img_tensor = preprocess(pil_img)
    bits = address_to_bits(wallet_address).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        watermarked_tensor = enc(img_tensor, bits)

    watermarked_pil = tensor_to_pil(watermarked_tensor)
    watermarked_pil = watermarked_pil.resize(original_size, Image.LANCZOS)

    buf = io.BytesIO()
    watermarked_pil.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/upload-ipfs")
async def upload_ipfs(
    image: UploadFile = File(...),
    name: str = Form(...),
):
    image_bytes = await image.read()
    headers = {"Authorization": f"Bearer {PINATA_JWT}"}

    img_res = http_requests.post(
        "https://uploads.pinata.cloud/v3/files",
        headers=headers,
        files={"file": ("watermarked.png", image_bytes, "image/png")}
    )
    if not img_res.ok:
        return JSONResponse({"error": f"Image upload failed: {img_res.text}"}, status_code=500)

    image_cid = img_res.json()["data"]["cid"]
    image_uri = f"ipfs://{image_cid}"

    metadata = {
        "name": name,
        "description": "Watermarked NFT minted via VerifyNFT",
        "image": image_uri,
    }
    meta_res = http_requests.post(
        "https://uploads.pinata.cloud/v3/files",
        headers=headers,
        files={"file": ("metadata.json", json.dumps(metadata).encode(), "application/json")}
    )
    if not meta_res.ok:
        return JSONResponse({"error": f"Metadata upload failed: {meta_res.text}"}, status_code=500)

    meta_cid = meta_res.json()["data"]["cid"]
    return JSONResponse({"token_uri": f"ipfs://{meta_cid}"})


@app.post("/verify")
async def verify(
    image: UploadFile = File(...),
    wallet_address: str = Form(...),
):
    raw = await image.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    img_tensor = preprocess(pil_img)

    extracted_bits, confidence = try_decode_variants(img_tensor)

    expected_bits = address_to_bits(wallet_address).to(DEVICE)
    bits_accuracy = extracted_bits.eq(expected_bits).float().mean().item() * 100

    extracted_prefix = bits_to_address_prefix(extracted_bits)
    wallet_prefix = "0x" + wallet_address[2:2 + REAL_BYTES * 2].upper()
    match = bits_accuracy >= 90.0

    return JSONResponse({
        "match": match,
        "extracted": extracted_prefix,
        "wallet": wallet_address,
        "wallet_prefix": wallet_prefix,
        "bits_accuracy": round(bits_accuracy, 2),
        "decoder_confidence": round(confidence, 4),
    })

@app.get("/")
def root():
    return {"status": "ok", "message": "NFT Watermark API is running"}
