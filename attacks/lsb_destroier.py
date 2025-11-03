import numpy as np
import cv2
from PIL import Image
import io
import random

# ---------- helper attacks ----------
def add_awgn(img, sigma=1.0, seed=None):
    """img: uint8 ndarray grayscale. sigma: standard deviation."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    noise = rng.normal(0, sigma, img.shape)
    out = img.astype(np.float32) + noise
    return np.uint8(np.clip(np.round(out), 0, 255))

def jpeg_compress(img, quality=90):
    """Return uint8 ndarray after JPEG compression (using PIL)."""
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=int(quality))
    buf.seek(0)
    pil2 = Image.open(buf).convert('L')
    return np.array(pil2, dtype=np.uint8)

def resize_down_up(img, scale=0.95, interp=cv2.INTER_CUBIC):
    h, w = img.shape
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    small = cv2.resize(img, (new_w, new_h), interpolation=interp)
    back = cv2.resize(small, (w, h), interpolation=interp)
    return np.uint8(np.clip(np.round(back), 0, 255))

def median_filter(img, ksize=3):
    return cv2.medianBlur(img, ksize)

def gaussian_blur(img, sigma=0.8, ksize=None):
    if ksize is None:
        # choose odd kernel from sigma
        k = max(3, int(2 * round(3 * sigma) + 1))
    else:
        k = ksize
    return cv2.GaussianBlur(img, (k, k), sigmaX=float(sigma))

def unsharp_mask(img, amount=1.0, sigma=1.0):
    blur = gaussian_blur(img, sigma=sigma)
    res = img.astype(np.float32) + amount * (img.astype(np.float32) - blur.astype(np.float32))
    return np.uint8(np.clip(np.round(res), 0, 255))

# ---------- diagnostics ----------
def lsb_stats(img):
    """Return fraction of pixels with LSB == 1 (odd) and counts."""
    arr = img.flatten()
    odd = np.count_nonzero(arr & 1)
    total = arr.size
    return odd / total, odd, total

# ---------- attack orchestrator ----------
def attack_remove_lsb(img,
                      mode='soft',
                      seed=None,
                      apply_sharpen_after=False,
                      local_var_mask=False,
                      var_win=7, var_thresh=20):
    """
    Mode: 'soft' | 'mid' | 'aggressive' | 'custom'
    If local_var_mask True, stronger operations are applied on high-variance regions.
    Returns attacked_img (uint8 ndarray).
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    rng = np.random.RandomState(seed)

    im = img.copy()
    h, w = im.shape

    # prepare local variance mask optionally
    mask = None
    if local_var_mask:
        mean = cv2.blur(im.astype(np.float32), (var_win, var_win))
        mean_sq = cv2.blur((im.astype(np.float32)**2), (var_win, var_win))
        var = np.maximum(0, mean_sq - mean**2)
        mask = (var > var_thresh).astype(np.uint8)  # 0/1

    if mode == 'soft':
        # AWGN small -> median 3 -> jpeg high quality
        im = add_awgn(im, sigma=1.0, seed=rng.randint(0, 2**31-1))
        im = median_filter(im, ksize=3)
        im = jpeg_compress(im, quality=90)

    elif mode == 'mid':
        # resize slightly -> AWGN medium -> median -> jpeg
        im = resize_down_up(im, scale=0.95, interp=cv2.INTER_CUBIC)
        im = add_awgn(im, sigma=2.0, seed=rng.randint(0, 2**31-1))
        im = median_filter(im, ksize=3)
        im = jpeg_compress(im, quality=85)

    elif mode == 'aggressive':
        # stronger mixing: resize heavier -> median 5 -> AWGN higher -> jpeg lower
        im = resize_down_up(im, scale=0.92, interp=cv2.INTER_CUBIC)
        im = median_filter(im, ksize=5)
        im = add_awgn(im, sigma=4.0, seed=rng.randint(0, 2**31-1))
        im = jpeg_compress(im, quality=78)

    elif mode == 'custom':
        # example custom: local_var_mask percent mixing
        if local_var_mask:
            # apply AWGN stronger only where mask==1
            awgn = add_awgn(im, sigma=3.0, seed=rng.randint(0, 2**31-1))
            im = np.where(mask[...,None]==1, awgn[...,None], im[...,None]).squeeze()
            im = median_filter(im, ksize=3)
            im = jpeg_compress(im, quality=85)
        else:
            # fall back to mid
            im = resize_down_up(im, scale=0.95, interp=cv2.INTER_CUBIC)
            im = add_awgn(im, sigma=2.5, seed=rng.randint(0, 2**31-1))
            im = median_filter(im, ksize=3)
            im = jpeg_compress(im, quality=85)

    else:
        raise ValueError("Unknown mode")

    if apply_sharpen_after:
        # small unsharp to restore crispness, careful with amount
        im = unsharp_mask(im, amount=0.6, sigma=0.8)

    return np.uint8(np.clip(im, 0, 255))

# # ---------- example usage ----------
# if __name__ == "__main__":
#     import imageio
#     img = imageio.imread("sample_wm.png", as_gray=True)  # assicurati uint8
#     if img.dtype != np.uint8:
#         img = (img * 255).astype(np.uint8)
#     print("Before LSB odd ratio:", lsb_stats(img))
#     attacked = attack_remove_lsb(img, mode='soft', seed=42)
#     print("After LSB odd ratio:", lsb_stats(attacked))
#     imageio.imwrite("sample_wm_attacked.png", attacked)
