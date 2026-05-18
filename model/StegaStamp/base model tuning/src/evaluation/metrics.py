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
    img = (img_tensor + 1) / 2  # [-1,1] → [0,1]
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
        if (addr.startswith('0x') and
            len(addr) == len(target_str) and
            addr != target_str):
            valid_but_wrong += 1

    ear = valid_but_wrong / num_trials

    return asr, ear, recovered_secrets
