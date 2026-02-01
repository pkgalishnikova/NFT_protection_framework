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

training_mode = False
