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
