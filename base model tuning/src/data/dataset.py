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
