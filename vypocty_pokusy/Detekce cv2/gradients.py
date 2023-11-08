import numpy as np
import cv2
import matplotlib.pyplot as plt


# Načtení černobílé fotky (výška, šířka)
image = cv2.imread('photos/IMG_0385.JPG', cv2.IMREAD_GRAYSCALE)[450:500, 2900:2950]
height, width = image.shape

# Vytvoření mřížky směrů (osm směrů)
angles = np.arange(0, 360, 45)

# Vytvoření pole pro ukládání gradientů
gradients = np.zeros((height, width, len(angles)))

# Střed obrazu
center_x, center_y = width // 2, height // 2

# Výpočet gradientů ve směrech 0, 45, 90, 135, 180, 225, 270 a 315 stupňů na délce 10 od středu
for i, angle in enumerate(angles):
    dx = int(np.cos(np.deg2rad(angle)) * 10)
    dy = int(np.sin(np.deg2rad(angle)) * 10)
    kernel = np.array([[dx], [dy]])
    gradient = cv2.filter2D(image, cv2.CV_64F, kernel)
    gradients[:, :, i] = gradient[0:center_y+height, 0:center_x+width]

# Vykreslení výsledků
plt.figure(figsize=(12, 8))
plt.subplot(3, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

for i, angle in enumerate(angles):
    plt.subplot(3, 3, i + 2)
    plt.imshow(gradients[:, :, i], cmap='gray')
    plt.title(f'Gradient {angle}°')

plt.tight_layout()
plt.show()
