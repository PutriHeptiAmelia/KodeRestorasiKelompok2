import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import math

# ============================================================
# 1. BACA GAMBAR
# ============================================================

path = "bunga_potrait.png"   # ganti ke nama file gambarmu
img = Image.open(path).convert("L")
img = np.array(img)

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
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return z

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
    window = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            window.append(img[x+i][y+j])
    return window

# ============================================================
# 6. FILTER MANUAL
# ============================================================

def mean_filter(img):
    output = np.zeros_like(img)
    h, w = img.shape
    for x in range(1, h-1):
        for y in range(1, w-1):
            wdw = get_window_3x3(img, x, y)
            output[x, y] = sum(wdw) / 9
    return output

def median_filter(img):
    output = np.zeros_like(img)
    h, w = img.shape
    for x in range(1, h-1):
        for y in range(1, w-1):
            wdw = sorted(get_window_3x3(img, x, y))
            output[x, y] = wdw[4]
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
# 8. BUAT 4 JENIS NOISE (BERDASARKAN INPUT USER)
# ============================================================

g1 = gaussian_noise(img, mu=0, sigma=sigma1)
g2 = gaussian_noise(img, mu=0, sigma=sigma2)

sp1 = salt_and_pepper(img, Pa=pa1, Pb=pb1)
sp2 = salt_and_pepper(img, Pa=pa2, Pb=pb2)

# ============================================================
# 9. FILTER SEMUA HASIL
# ============================================================

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

# ============================================================
# 10. TAMPILKAN SEMUA (21 GAMBAR SATU PER SATU)
# ============================================================

images = [
    ("Citra Asli", img),

    (f"Gaussian σ={sigma1}", g1),
    (f"Gaussian σ={sigma2}", g2),

    (f"Salt & Pepper Pa={pa1} Pb={pb1}", sp1),
    (f"Salt & Pepper Pa={pa2} Pb={pb2}", sp2),

    ("Mean (G1)", results["Mean_g1"]),
    ("Mean (G2)", results["Mean_g2"]),
    ("Mean (SP1)", results["Mean_sp1"]),
    ("Mean (SP2)", results["Mean_sp2"]),

    ("Median (G1)", results["Median_g1"]),
    ("Median (G2)", results["Median_g2"]),
    ("Median (SP1)", results["Median_sp1"]),
    ("Median (SP2)", results["Median_sp2"]),

    ("Min (G1)", results["Min_g1"]),
    ("Min (G2)", results["Min_g2"]),
    ("Min (SP1)", results["Min_sp1"]),
    ("Min (SP2)", results["Min_sp2"]),

    ("Max (G1)", results["Max_g1"]),
    ("Max (G2)", results["Max_g2"]),
    ("Max (SP1)", results["Max_sp1"]),
    ("Max (SP2)", results["Max_sp2"]),
]

for title, image in images:
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()

# ============================================================
# 11. CETAK MSE
# ============================================================

print("\n===== MSE RESULTS =====")
for key in results:
    print(key, ":", mse(img, results[key]))
