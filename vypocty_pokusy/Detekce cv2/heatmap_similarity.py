import cv2
import matplotlib.pyplot as plt

# Načtení původní fotografie
image1 = cv2.imread('photos/IMG_0385.JPG')
image2 = cv2.imread('photos/IMG_0417.JPG')

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Načtení hledané oblasti (například šablony)
"""x1, y1 = 2600, 400
x2, y2 = 3450, 650"""
x1, y1 = 2750, 450
x2, y2 = x1 + 100, y1 + 100

"""x1, y1 = 2720, 1100
x2, y2 = 2890, 1170"""

template = gray1[y1:y2, x1:x2]

# Použití pixel Matching pro nalezení podobných míst
result = cv2.matchTemplate(gray2, template, cv2.TM_CCOEFF_NORMED)
# ### TM_CCOEFF_NORMED / TM_CCORR_NORMED /  TM_SQDIFF_NORMED -> min_loc

# Získání souřadnic nejlepšího výskytu hledané oblasti
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc
bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

# Vytvoření kopie původní fotografie s označenou podobnou oblastí
marked_image = image2.copy()
cv2.rectangle(marked_image, top_left, bottom_right, (0, 0, 255), 10)
cv2.rectangle(image1, (x1, y1), (x2, y2), (0, 0, 255), 10)

# Zobrazení grafu
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.title('Původní fotografie')
plt.tight_layout()
plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
plt.title('Označená podobná oblast')
plt.tight_layout()

# Normalizace výsledků na rozsah 0-1
normalized_result = cv2.normalize(result, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

"""
import numpy as np

# Vytvoření grafu tepelné mapy
heatmap = cv2.applyColorMap(np.uint8(255 * normalized_result), cv2.COLORMAP_JET)

# Zobrazení grafu tepelné mapy
fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
ax.set_title('Tepelná mapa podobnosti')
ax.axis('off')
plt.tight_layout()"""

# Vytvoření grafu tepelné mapy
plt.figure(figsize=(8, 6))
plt.imshow(normalized_result, cmap='jet', vmin=0, vmax=1)
plt.colorbar()
plt.title('Tepelná mapa podobnosti')
plt.axis('off')
plt.tight_layout()

plt.show()
