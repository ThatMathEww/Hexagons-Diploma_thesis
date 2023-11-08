import cv2
import numpy as np

# Načtení obrazů
image1 = cv2.imread('photos/IMG_0385.JPG', 0)  # První fotografie (šedotónová)
image2 = cv2.imread('photos/IMG_0417.JPG', 0)  # Druhá fotografie (šedotónová)

# Definice vrcholů prvního mnohoúhelníku
vertices1 = np.array([[100, 100], [200, 50], [300, 150], [200, 200]], np.int32)
vertices2 = np.array([[400, 200], [500, 150], [600, 250]], np.int32)

# Vytvoření prázdných mask pro obě oblasti
mask1 = np.zeros(image1.shape[:2], dtype=np.uint8)
mask2 = np.zeros(image2.shape[:2], dtype=np.uint8)

# Vykreslení mnohoúhelníků na maskách
cv2.fillPoly(mask1, [vertices1], 255)
cv2.fillPoly(mask1, [vertices2], 255)  # nebo zvlášť masky: cv2.fillPoly(mask2, [vertices2], 255)

# Aplikace mask na obraz
masked_image1 = cv2.bitwise_and(image1, image1, mask=mask1)
# masked_image2 = cv2.bitwise_and(image, image, mask=mask2)

# Vytvoření obrazu s intenzitou 0.2
image_half_intensity = (image1 * 0.2).astype(np.uint8)

# Spojení původního obrazu s intenzitou 0.5 a sledovanými oblastmi
combined_image = cv2.addWeighted(image_half_intensity, 1.0, masked_image1, 0.5, 0)

# combined_image = cv2.addWeighted(combined_image, 1.0, masked_image2, 0.5, 0)

bar = 0


def bar1(*args):
    global bar
    bar = cv2.getTrackbarPos("Presnost:", "Parameters")
    print(bar)


def bar2(*args):
    pass


height, width = combined_image.shape

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 650, 240)
cv2.createTrackbar("Presnost:", "Parameters", 155, 255, bar1)
cv2.createTrackbar("Jas:", "Parameters", 50, 100, bar2)

# Zobrazení výsledků
cv2.namedWindow("Masked image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Masked image", int(0.15 * width), int(0.15 * height))
cv2.imshow("Masked image", combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
