import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Načtení obrázku
image1 = cv2.imread('photos/IMG_0385.JPG', cv2.IMREAD_GRAYSCALE)  # [1100:1170, 2470:2540]
image2 = image1.copy()[400:650, 2600:3450]
image3 = cv2.imread('photos/IMG_0417.JPG', cv2.IMREAD_GRAYSCALE)  # [1750:1820, 2500:2570]

points_to_track = np.int32(
    [[2508 - 2470, 1140 - 1100], [1935, 2130], [4228, 2138], [1932, 2131], [2515, 1145], [3648, 3128],
     [2979, 2105]])
point2 = np.int32([[2535 - 2500, 1775 - 1750]])

index = 0
radius = 28

px1, py1 = points_to_track[index]
px2, py2 = point2[index]

image1_norm = cv2.normalize(image1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
image2_norm = cv2.normalize(image2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

corr = cv2.filter2D(image1_norm, ddepth=-1, kernel=image2_norm)

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
edges_x = cv2.filter2D(image1, cv2.CV_8U, sobel_x)  # Detekce hran

laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
sharp_edges = cv2.filter2D(image1, cv2.CV_8U, laplacian_kernel)  # Detekce hran ostré

kernel = np.ones((5, 5), np.float32) / 25  # Průměrovací jádro 5x5
blurred_image = cv2.filter2D(image1, -1, kernel)

g_blurred_image = cv2.GaussianBlur(image1, (5, 5), 0)

kernel = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]], dtype=np.float32)
embossed_image = cv2.filter2D(image1, cv2.CV_8U, kernel)

sobel_x = cv2.Sobel(image1, cv2.CV_16S, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image1, cv2.CV_16S, 0, 1, ksize=3)
gradient_image = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)

canny = cv2.Canny(image1, threshold1=100, threshold2=200)
laplacian = cv2.Laplacian(image1, cv2.CV_16S)

# Najděte kontury
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
hex_image = image1.copy()

# Vykreslete kontury na kopii původního obrazu
image_with_contours = image1.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

# Procházení nalezených kontur
for contour in contours:
    # Aproximace kontury (použijeme relativně malý parametr epsilon)
    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Pokud byl nalezený tvar málo aproximovaný nebo příliš zjednodušený (např. kvůli šumu),
    # můžeme zkontrolovat, zda má tvar 6 hran (šestiúhelník).
    if len(approx) == 6:
        # Můžete také přidat další podmínky pro ověření, že úhly jsou blízko 120 stupňů, apod.
        # ...

        # Nakreslíme nalezený šestiúhelník na obrázku
        cv2.drawContours(hex_image, [approx], 0, (0, 255, 0), 2)

img = image1.copy()
# img = cv2.pyrDown(img)
laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
sharp_edges_lap = cv2.filter2D(img, cv2.CV_8U, laplacian_kernel)  # Detekce hran ostré
size = 25
kernel = np.ones((size, size), np.float32) / (size ** 2)
blurred_image_lap = cv2.filter2D(sharp_edges_lap, -1, kernel)

threshold_value = 7
_, binary_mask = cv2.threshold(blurred_image_lap, threshold_value, 255, cv2.THRESH_BINARY)

# Morfologická operace dilatace pro spojení blízkých hran
kernel = np.ones((60, 60), np.uint8)
dilated_mask = cv2.dilate(binary_mask, kernel, iterations=3)

# Morfologická operace eroze pro odstranění malých objektů a zúžení hran
eroded_mask_lap = cv2.erode(dilated_mask, kernel, iterations=2)

"""size = 5
kernel = np.ones((size, size), np.float32) / (size * size)
blurred_image_lap = cv2.filter2D(mask, -1, kernel)
_, mask = cv2.threshold(blurred_image_lap, threshold_value, 255, cv2.THRESH_BINARY)"""

result_lap = cv2.bitwise_and(img, img, mask=eroded_mask_lap)

plt.figure(figsize=(18, 6))
plt.subplot(131)
plt.title("Blur")
plt.imshow(blurred_image_lap)
plt.axis("equal")

plt.subplot(132)
plt.title("Mask")
plt.imshow(binary_mask)
plt.axis("equal")

plt.subplot(133)
plt.title("Result")
plt.imshow(cv2.cvtColor(result_lap, cv2.COLOR_BGR2RGB))
plt.axis("equal")

plt.tight_layout()

sobel_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
gradient_image = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)
size = 15
kernel = np.ones((size, size), np.float32) / (size ** 2)
blurred_image_gradient = cv2.filter2D(gradient_image, -1, kernel)

threshold_value = 50
_, binary_mask = cv2.threshold(blurred_image_gradient, threshold_value, 255, cv2.THRESH_BINARY)

# Morfologická operace dilatace pro spojení blízkých hran
kernel = np.ones((25, 25), np.uint8)
dilated_mask = cv2.dilate(binary_mask, kernel, iterations=3)

# Morfologická operace eroze pro odstranění malých objektů a zúžení hran
eroded_mask_grad = cv2.erode(dilated_mask, kernel, iterations=2)

result_grad = cv2.bitwise_and(img, img, mask=eroded_mask_grad)

plt.figure(figsize=(18, 6))
plt.subplot(131)
plt.title("Gradient")
plt.imshow(blurred_image_gradient)
plt.axis("equal")

plt.subplot(132)
plt.title("Mask")
plt.imshow(binary_mask)
plt.axis("equal")

plt.subplot(133)
plt.title("Result_gradient")
plt.imshow(cv2.cvtColor(result_grad, cv2.COLOR_BGR2RGB))
plt.axis("equal")

plt.tight_layout()

plt.figure(figsize=(18, 6))
plt.subplot(131)
plt.title("binary_mask")
plt.imshow(binary_mask)
plt.subplot(132)
plt.title("dilated_mask")
plt.imshow(dilated_mask)
plt.subplot(133)
plt.title("eroded_mask")
plt.imshow(eroded_mask_grad)
plt.tight_layout()

plt.figure(figsize=(17, 6))
photo = cv2.addWeighted(cv2.cvtColor(result_grad, cv2.COLOR_BGR2RGB)[1730:3170, 1530:4630], 0.5,
                        cv2.cvtColor(result_lap, cv2.COLOR_BGR2RGB)[1730:3170, 1530:4630], 0.5, 0)
total_mask = cv2.add(eroded_mask_grad[1730:3170, 1530:4630],
                     eroded_mask_lap[1730:3170, 1530:4630])
photo = cv2.cvtColor(cv2.bitwise_and(img[1730:3170, 1530:4630], img[1730:3170, 1530:4630], mask=total_mask),
                     cv2.COLOR_BGR2RGB)
plt.subplot(131)
mask2 = cv2.cvtColor(eroded_mask_lap, cv2.COLOR_BGR2RGB)
plt.imshow(mask2)
plt.subplot(132)
mask2 = cv2.cvtColor(eroded_mask_grad, cv2.COLOR_BGR2RGB)
plt.imshow(mask2)
plt.subplot(133)
mask_add = cv2.cvtColor(cv2.bitwise_and(eroded_mask_lap, eroded_mask_grad), cv2.COLOR_BGR2RGB)
plt.imshow(mask_add)
plt.tight_layout()

plt.show()

# Zobrazení obrázků s klíčovými body
"""plt.figure()
plt.subplot(331)
plt.title("Original")
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.axis("equal")
plt.subplot(332)
plt.title("Correlation")
plt.imshow(corr)
plt.axis("equal")
plt.subplot(333)
plt.title("Blurr")
plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
plt.axis("equal")
plt.subplot(334)
plt.title("Nose cancel")
plt.imshow(cv2.cvtColor(g_blurred_image, cv2.COLOR_BGR2RGB))
plt.axis("equal")
plt.subplot(335)
plt.title("3D efekt")
plt.imshow(embossed_image)
plt.axis("equal")

plt.tight_layout()"""

plt.figure()
plt.subplot(331)
plt.title("Original")
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.axis("equal")

plt.subplot(332)
plt.title("Edges soft")
plt.imshow(edges_x)
plt.axis("equal")

plt.subplot(333)
plt.title("Edges sharp")
plt.imshow(sharp_edges)
plt.axis("equal")

plt.subplot(334)
plt.title("3D efekt")
plt.imshow(embossed_image)
plt.axis("equal")

plt.subplot(335)
plt.title("Gradient x,y")
plt.imshow(gradient_image)
plt.axis("equal")

plt.subplot(336)
plt.title("Canny edges")
plt.imshow(canny)
plt.axis("equal")

plt.subplot(336)
plt.title("Laplacian Edges")
plt.imshow(cv2.convertScaleAbs(laplacian))
plt.axis("equal")

plt.subplot(337)
plt.title("Contours")
plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
plt.axis("equal")

plt.subplot(338)
plt.title("Contours")
plt.imshow(cv2.cvtColor(hex_image, cv2.COLOR_BGR2RGB))
plt.axis("equal")

plt.subplot(339)
plt.title("Masked")
plt.imshow(cv2.cvtColor(result_lap, cv2.COLOR_BGR2RGB))
plt.axis("equal")

plt.tight_layout()

"""mng = plt.get_current_fig_manager()
mng.full_screen_toggle()"""
"""fig = plt.gcf()
fig.canvas.manager.full_screen_toggle()"""

# plt.show()
