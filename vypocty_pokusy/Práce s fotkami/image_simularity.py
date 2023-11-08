import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np

# Načtěte dva obrázky (jako příklad použijeme stejný obrázek pro ukázku)
image1 = cv2.imread("photos/0/IMG_1.png", cv2.IMREAD_GRAYSCALE)  # [1100:1170, 2470:2540]
image2 = cv2.imread("photos/0/IMG_2.png", cv2.IMREAD_GRAYSCALE)  # [1750:1820, 2500:2570]
# image2 = cv2.rotate(image2, cv2.ROTATE_90_CLOCKWISE)

# Ujistěte se, že oba obrázky mají stejnou velikost
if image1.shape != image2.shape:
    raise ValueError("Obrázky nemají stejnou velikost.")

# Výpočet rozdílu mezi obrázky
difference = cv2.absdiff(image1, image2)

# Výpočet průměrné hodnoty rozdílu (čím nižší hodnota, tím větší podobnost)
similarity = np.mean(difference)

# Zobrazíme průměrnou hodnotu rozdílu
print(f"Podobnost obrazků: {similarity}")

score, diff = ssim(image1, image2, full=True)

print("Podobnost Score: {:.3f}%".format(score * 100))

diff = (diff * 255).astype("uint8")

mask = np.zeros(image1.shape, dtype="uint8")
filled = image2.copy()

thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

for c in contours:
    area = cv2.contourArea(c)
    if area > 100:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image1, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.rectangle(image2, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
        cv2.drawContours(filled, [c], 0, (0, 255, 0), -1)

picture1 = cv2.imread("photos/IMG_0385.JPG", cv2.IMREAD_GRAYSCALE)[400:650, 2750:3000]  # np.random.rand(100, 100)
picture2 = cv2.imread("photos/IMG_0401.JPG", cv2.IMREAD_GRAYSCALE)[1075:1325, 2745:2995]
picture2 = cv2.rotate(picture2, cv2.ROTATE_180)
picture1_norm = picture1 / np.sqrt(np.sum(picture1 ** 2))
picture2_norm = picture2 / np.sqrt(np.sum(picture2 ** 2))
similarity = np.sum(picture2_norm * picture1_norm)
print("Score: {:.3f}%".format(similarity))

# 1. Standardizace hodnot v maticích (z-score normalizace)
a_mean = np.mean(picture1)
a_std = np.std(picture1)
a_normalized_z = (picture1 - a_mean) / a_std

b_mean = np.mean(picture2)
b_std = np.std(picture2)
b_normalized_z = (picture2 - b_mean) / b_std

similarity = np.sum(a_normalized_z * b_normalized_z)
print("Score Z NORM: {:.3f}%".format(similarity))

# 1. Normalizace hodnot v maticích (pomocí Min-Max normalizace)
a_normalized_mm = (picture1 - np.min(picture1)) / (np.max(picture1) - np.min(picture1))
b_normalized_mm = (picture2 - np.min(picture2)) / (np.max(picture2) - np.min(picture2))

similarity = np.sum(a_normalized_mm * b_normalized_mm)
print("Score MM NORM: {:.3f}%".format(similarity / 100))


def horn_method_similarity(matrix1, matrix2):
    U1, s1, V1t = np.linalg.svd(matrix1, full_matrices=False)
    U2, s2, V2t = np.linalg.svd(matrix2, full_matrices=False)

    # Uspořádat singulární hodnoty sestupně
    s1_sorted = np.sort(s1)[::-1]
    s2_sorted = np.sort(s2)[::-1]

    # Vypočítat podobnost pomocí kvadratické normy rozdílu
    similarity = 1 / (1 + np.linalg.norm(s1_sorted - s2_sorted) ** 2)
    return similarity


# Vypočítání kosínové podobnosti mezi normalizovanými maticemi a_normalized a b_normalized
similarity_score = horn_method_similarity(picture1, picture2)
print("horn podobnost:", similarity_score)
similarity_score = horn_method_similarity(a_normalized_z, a_normalized_z)
print("horn podobnost Z:", similarity_score)
similarity_score = horn_method_similarity(a_normalized_mm, a_normalized_mm)
print("horn podobnost Min_Max:", similarity_score)


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / np.sqrt(norm_a * norm_b)


# Vypočítání kosínové podobnosti mezi normalizovanými maticemi a_normalized a b_normalized
similarity_score = cosine_similarity(a_normalized_z.flatten(), a_normalized_z.flatten())
print("Kosínová podobnost Z:", similarity_score)
similarity_score = cosine_similarity(a_normalized_mm.flatten(), a_normalized_mm.flatten())
print("Kosínová podobnost Min_Max:", similarity_score)


def euclidean_distance(a, b):
    result = np.linalg.norm(a - b)  # (np.linalg.norm(a - b) / np.sqrt(np.sum(a) ** 2 + np.sum(b) ** 2)) * 100
    return result


# Vypočítání euklidovské vzdálenosti mezi normalizovanými maticemi a_normalized a b_normalized
distance_score = euclidean_distance(a_normalized_z.flatten(), b_normalized_z.flatten())

# Invertujeme vzdálenost, aby vyjadřovala podobnost (čím menší vzdálenost, tím větší podobnost)
similarity_score = 1 / (1 + distance_score)

print("Euklidovská vzdálenost Z:", distance_score)
print("Podobnost Z:", similarity_score)

distance_score = euclidean_distance(a_normalized_mm.flatten(), b_normalized_mm.flatten())
similarity_score = 1 / (1 + distance_score)

print("Euklidovská vzdálenost Min_Max:", distance_score)
print("Podobnost Min_Max:", similarity_score)

first_image_hist = cv2.calcHist([picture1], [0], None, [256], [0, 256])
second_image_hist = cv2.calcHist([picture2], [0], None, [256], [0, 256])

img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_CORREL)  # HISTCMP_BHATTACHARYYA
print("histogram:", img_hist_diff)


# 3. Porovnání normalizovaných matic pomocí korelačního koeficientu
def correlation_coefficient(a, b):
    return np.corrcoef(a.flatten(), b.flatten())[0, 1]


# Vypočítání korelačního koeficientu mezi normalizovanými maticemi a_normalized a b_normalized
correlation_score = correlation_coefficient(a_normalized_z, b_normalized_z)

print("Korelační koeficient Z:", correlation_score)

correlation_score = correlation_coefficient(a_normalized_mm, b_normalized_mm)
print("Korelační koeficient MM:", correlation_score)


def pearson_correlation(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    covariance = np.cov(a_flat, b_flat)[0, 1]
    std_a = np.std(a_flat)
    std_b = np.std(b_flat)
    correlation = covariance / (std_a * std_b)
    return correlation


# Vypočítání korelačního koeficientu Pearsonovy korelace mezi normalizovanými maticemi a_normalized a b_normalized
correlation_score = pearson_correlation(a_normalized_z, b_normalized_z)

print("Korelační koeficient Pearsonovy korelace Z:", correlation_score)

correlation_score = pearson_correlation(a_normalized_mm, b_normalized_mm)
print("Korelační koeficient Pearsonovy korelace MM:", correlation_score)

from scipy.stats import spearmanr


def spearman_correlation(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    correlation, _ = spearmanr(a_flat, b_flat)
    return correlation


# Vypočítání korelačního koeficientu Spearmanovy korelace mezi maticemi a_matrix a b_matrix
correlation_score = spearman_correlation(a_normalized_z, b_normalized_z)
print("Korelační koeficient Spearmanovy korelace Z:", correlation_score)

correlation_score = spearman_correlation(a_normalized_mm, b_normalized_mm)
print("Korelační koeficient Spearmanovy korelace MM:", correlation_score)

from scipy.ndimage import rotate

# Předpokládejme, že máme původní matici a_matrix a otočenou matici b_matrix

# Příklad původní matice
a_matrix = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

# Příklad otočené matice
angle = 30  # Úhel otočení ve stupních
b_matrix = rotate(a_matrix, angle, reshape=False)

# 1. Normalizace hodnot v maticích (pomocí Min-Max normalizace)
a_normalized = (a_matrix - np.min(a_matrix)) / (np.max(a_matrix) - np.min(a_matrix))
b_normalized = (b_matrix - np.min(b_matrix)) / (np.max(b_matrix) - np.min(b_matrix))


# 2. Porovnání podobnosti pro různé úhly otočení a vybrání úhlu s nejvyšší podobností
def find_best_rotation_angle(a, b):
    best_similarity = -1
    best_angle = 0

    for angle in range(0, 360, 5):  # zkoušíme úhly od 0 do 355 s krokem 5 stupňů
        rotated_b = rotate(b, angle, reshape=False)
        rotated_b_normalized = (rotated_b - np.min(rotated_b)) / (np.max(rotated_b) - np.min(rotated_b))
        similarity = np.dot(a.flatten(), rotated_b_normalized.flatten())

        if similarity > best_similarity:
            best_similarity = similarity
            best_angle = angle

    return best_angle, best_similarity


# Najít nejlepší úhel otočení a jeho podobnost s původní maticí
best_angle, similarity_score = find_best_rotation_angle(a_normalized, b_normalized)

print("Nejlepší úhel otočení:", 360 - best_angle, "stupňů")
print("Podobnost:", similarity_score)

plt.figure(figsize=(7, 6))
plt.subplot(221)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.subplot(222)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.subplot(223)
plt.imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
plt.subplot(224)
plt.imshow(cv2.cvtColor(filled, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()
