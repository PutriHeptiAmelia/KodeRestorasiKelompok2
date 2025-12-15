import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import math

# ============================================================
# 1. BACA DUA GAMBAR (POTRAIT & LANDSCAPE)
# ============================================================

path1 = "bunga_potrait.png"      # gambar potrait
path2 = "view_landscape.png"   # gambar landscape

img1 = Image.open(path1).convert("L")
img1 = np.array(img1)

img2 = Image.open(path2).convert("L")
img2 = np.array(img2)

# ============================================================
# 2. INPUT INTENSITAS DARI USER
# ============================================================

print("=== INPUT INTENSITAS GAUSSIAN NOISE ===")
sigma1 = float(input("Masukkan sigma Gaussian 1: "))
sigma2 = float(input("Masukkan sigma Gaussian 2: "))

print("\n=== INPUT PROBABILITAS SALT & PEPPER ===")
pa1 = float(input("Masukkan Pa untuk Salt & Pepper 1: "))
pb1 = float(input("Masukkan Pb untuk Salt & Pepper 1: "))
pa2 = float(input("Masukkan Pa untuk Salt & Pepper 2: "))
pb2 = float(input("Masukkan Pb untuk Salt & Pepper 2: "))

# ============================================================
# 3. GAUSSIAN NOISE MANUAL (Box-Muller)
# ============================================================

def randn():
    u1 = random.random()
    u2 = random.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

def gaussian_noise(img, mu, sigma):
    noisy = img.astype(float)
    h, w = img.shape
    for x in range(h):
        for y in range(w):
            noisy[x, y] += randn() * sigma + mu
    return np.clip(noisy, 0, 255).astype(np.uint8)

# ============================================================
# 4. SALT & PEPPER NOISE MANUAL
# ============================================================

def salt_and_pepper(img, Pa, Pb, a=0, b=255):
    noisy = img.copy()
    h, w = img.shape
    for x in range(h):
        for y in range(w):
            r = random.random()
            if r < Pa:
                noisy[x, y] = a
            elif r < Pa + Pb:
                noisy[x, y] = b
    return noisy

# ============================================================
# 5. WINDOW 3×3
# ============================================================

def get_window_3x3(img, x, y):
    return [img[x+i][y+j] for i in range(-1, 2) for j in range(-1, 2)]

# ============================================================
# 6. FILTER MANUAL
# ============================================================

def mean_filter(img):
    output = np.zeros_like(img)
    h, w = img.shape
    for x in range(1, h-1):
        for y in range(1, w-1):
            output[x, y] = sum(get_window_3x3(img, x, y)) / 9
    return output

def median_filter(img):
    output = np.zeros_like(img)
    h, w = img.shape
    for x in range(1, h-1):
        for y in range(1, w-1):
            output[x, y] = sorted(get_window_3x3(img, x, y))[4]
    return output

def min_filter(img):
    output = np.zeros_like(img)
    h, w = img.shape
    for x in range(1, h-1):
        for y in range(1, w-1):
            output[x, y] = min(get_window_3x3(img, x, y))
    return output

def max_filter(img):
    output = np.zeros_like(img)
    h, w = img.shape
    for x in range(1, h-1):
        for y in range(1, w-1):
            output[x, y] = max(get_window_3x3(img, x, y))
    return output

# ============================================================
# 7. MSE
# ============================================================

def mse(img1, img2):
    return np.mean((img1.astype(float) - img2.astype(float))**2)

# ============================================================
# 8–11. PROSES UNTUK SETIAP GAMBAR
# ============================================================

def process_image(img, label):
    g1 = gaussian_noise(img, 0, sigma1)
    g2 = gaussian_noise(img, 0, sigma2)
    sp1 = salt_and_pepper(img, pa1, pb1)
    sp2 = salt_and_pepper(img, pa2, pb2)

    filters = {
        "Mean": mean_filter,
        "Median": median_filter,
        "Min": min_filter,
        "Max": max_filter
    }

    results = {}
    for name, func in filters.items():
        results[f"{name}_g1"] = func(g1)
        results[f"{name}_g2"] = func(g2)
        results[f"{name}_sp1"] = func(sp1)
        results[f"{name}_sp2"] = func(sp2)

    images = [
        (f"{label} - Citra Asli", img),
        (f"{label} - Gaussian σ={sigma1}", g1),
        (f"{label} - Gaussian σ={sigma2}", g2),
        (f"{label} - Salt & Pepper 1", sp1),
        (f"{label} - Salt & Pepper 2", sp2),
    ]

    for k, v in results.items():
        images.append((f"{label} - {k}", v))

    for title, image in images:
        plt.figure(figsize=(6, 6))
        plt.title(title)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.show()

    print(f"\n===== MSE RESULTS ({label}) =====")
    for k, v in results.items():
        print(k, ":", mse(img, v))

# ============================================================
# 12. JALANKAN UNTUK DUA GAMBAR
# ============================================================

process_image(img1, "POTRAIT")
process_image(img2, "LANDSCAPE")
