import numpy as np
import cv2
import matplotlib.pyplot as plt

# Načtení černobílé fotky (výška, šířka)
image = cv2.imread('photos/IMG_0385.JPG', cv2.IMREAD_GRAYSCALE)[450:500, 2900:2950]
height, width = image.shape

# Výpočet gradientů v x a y směru pro původní fotografii
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
grad_x2 = cv2.Sobel(image, cv2.CV_64F, 2, 0, ksize=3)
grad_y2 = cv2.Sobel(image, cv2.CV_64F, 0, 2, ksize=3)

# Nyní načteme novou fotografii
sx, sy, ex, ey = 450, 500, 2900, 2950
new_image = cv2.imread('photos/IMG_0385.JPG', cv2.IMREAD_GRAYSCALE)[sx:sy, ex:ey]
# Výpočet gradientů v x a y směru pro novou fotografii na bodě (10, 32)
target_x, target_y = np.int32(35), np.int32(40)

# Velikost oblasti pro porovnání
region_size = 3

grad_x_target = cv2.Sobel(new_image[target_y - 1:target_y + 2, target_x - 1:target_x + 2], cv2.CV_64F, 1, 0, ksize=3)
grad_y_target = cv2.Sobel(new_image[target_y - 1:target_y + 2, target_x - 1:target_x + 2], cv2.CV_64F, 0, 1, ksize=3)
grad_x2_target = cv2.Sobel(new_image[target_y - 1:target_y + 2, target_x - 1:target_x + 2], cv2.CV_64F, 2, 0, ksize=3)
grad_y2_target = cv2.Sobel(new_image[target_y - 1:target_y + 2, target_x - 1:target_x + 2], cv2.CV_64F, 0, 2, ksize=3)

"""# Výpočet magnitudy gradientu a směru pro bod (10, 32) na nové fotografii
gradient_magnitude_target = np.sqrt(grad_x_target[target_x, target_y]**2 + grad_y_target[target_x, target_y]**2)
gradient_direction_target = np.arctan2(grad_y_target[target_x, target_y], grad_x_target[target_x, target_y]) * 180 / np.pi
if gradient_direction_target < 0:
    gradient_direction_target += 180"""

# Vyhledání bodů na původní fotografii, které mají podobné gradienty jako bod (target_x, target_y) na nové fotografii
matching_points = []
for x in range(1, image.shape[0] - 1):
    for y in range(1, image.shape[1] - 1):
        # Výpočet gradientů v dané oblasti na původní fotografii
        """grad_x_region = grad_x[x - 1:x + 2, y - 1:y + 2]
        grad_y_region = grad_y[x - 1:x + 2, y - 1:y + 2]"""
        img = image[y - 1:y + 2, x - 1:x + 2]

        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad_x2 = cv2.Sobel(img, cv2.CV_64F, 2, 0, ksize=3)
        grad_y2 = cv2.Sobel(img, cv2.CV_64F, 0, 2, ksize=3)

        # Porovnání gradientů
        if np.allclose(grad_x_target, grad_x) and np.allclose(grad_y_target, grad_y) and \
                np.allclose(grad_x2_target, grad_x2) and np.allclose(grad_y2_target, grad_y2):
            matching_points.append((x, y))

# Vypsání bodů
for x, y in matching_points:
    print(f"Souřadnice: ({x}, {y})")

print(f"Správně je: ({ex - 2900 + target_x}, {sx - 450 + target_y})")

# Vykreslení výsledků
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.scatter(x, y, color='red', marker='x', label='Target Point (Original)')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.scatter(target_x, target_y, color='red', marker='x', label='Target Point (New)')
plt.title('New Image')

"""plt.subplot(1, 3, 3)
plt.imshow(new_image, cmap='gray')
plt.quiver(target_y, target_x, grad_x_target[target_x, target_y], grad_y_target[target_x, target_y],
           angles='xy', scale_units='xy', scale=1, color='red')
plt.scatter(target_y, target_x, color='red', marker='x', label='Target Point (New)')
plt.title('Gradient at (10, 32)')"""

# plt.legend()
plt.tight_layout()
plt.show()
