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

        # Attention to enhance watermark regions
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
