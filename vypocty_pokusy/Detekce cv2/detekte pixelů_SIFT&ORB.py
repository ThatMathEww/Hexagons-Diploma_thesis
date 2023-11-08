import sys
import cv2
import numpy as np

##########################################
# Hledání pomocí "SIFT x ORB"
# mezi dvěma fotkami
# Bodů v oblasti
###########################################

# Načtení obrazů
image1 = cv2.imread('photos/IMG_0385.JPG', 0)  # První fotografie (šedotónová)
image2 = cv2.imread('photos/IMG_0417.JPG', 0)  # Druhá fotografie (šedotónová)

height, width = image1.shape

c1 = 300
c2 = 900
c3 = 1000
image1, image2 = image1[c1:height - c2, c3:width - c3], image2[c1:height - c2, c3:width - c3]

# Inicializace detektoru
option = "sift"

if option == "sift":
    method = cv2.SIFT_create()
elif option == "orb":
    method = cv2.ORB_create()

    # Získání a výpis vybraných nastavení
    print("nFeatures: ", method.getMaxFeatures())
    print("scaleFactor: ", method.getScaleFactor())
    print("nLevels: ", method.getNLevels())
    print("edgeThreshold: ", method.getEdgeThreshold())
    print("patchSize: ", method.getPatchSize())
else:
    print("Špatně zvolená metoda.")
    sys.exit()

# Definice souřadnic oblasti
"""
# x1, y1, x2, y2 = 2491, 3099, 3655, 3147
x1_1, y1_1, x2_1, y2_1 = 2650, 400, 3450, 650
x1_2, y1_2, x2_2, y2_2 = x1_1 - 100, 0, x2_1 + 100, 1500
# x1, y1, x2, y2 = 0, 0, 6000, 4000

# Střed a poloměr kruhové oblasti
center_x, center_y = 3050, 525
radius = 100

# x1, y1, x2, y2 = 2650, 400, 3450, 650
# image1 = image1[y1:y2, x1:x2]
"""

# Definice vrcholů mnohoúhelníku
vertices = np.array([[3012 - c3, 2035 - c1], [2991 - c3, 2074 - c1], [3173 - c3, 2182 - c1],
                     [3200 - c3, 2141 - c1]], np.int32)

# Vytvoření masky oblasti
mask1 = np.zeros(image1.shape[:2], dtype=np.uint8)
mask2 = np.zeros(image2.shape[:2], dtype=np.uint8)
mask2 = None

# Vykreslení mnohoúhelníku na masce
cv2.fillPoly(mask1, [vertices], 255)

# Invertování masky
inverted_mask = cv2.bitwise_not(mask1)

# Aplikace masky na obraz s požadovanou intenzitou
image_masked = image1.copy()
image_masked[inverted_mask > 0] = (image_masked[inverted_mask > 0] * 0.2).astype(np.uint8)

# Zobrazení výsledku
cv2.namedWindow('Masked image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Masked image', int(0.15 * width), int(0.15 * height))
cv2.imshow("Masked image", image_masked)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""cv2.rectangle(mask1, (x1_1, y1_1), (x2_1, y2_1), 255, -1)
# cv2.circle(mask, (center_x, center_y), radius, 255, -1)
cv2.rectangle(mask2, (x1_2, y1_2), (x2_2, y2_2), 255, -1)"""

# Nalezení klíčových bodů a popisovačů pro oba obrazy
keypoints1, descriptors1_sift = method.detectAndCompute(image1, mask1)
keypoints2, descriptors2 = method.detectAndCompute(image2, mask2)

"""
# Seřazení shod podle přesnosti
matches = sorted(matches, key=lambda x: x.distance)

# Počet nejlepších shod, které chcete použít
num_matches = 50
good_matches = matches[:num_matches]"""

# Porovnání popisovačů pomocí algoritmu BFMatcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1_sift, descriptors2, k=2)

# Aplikace prahu na shody mezi popisovači
good_matches = []
precision = 0.05  # menší číslo => přísnější kritérium
while True:
    for m, n in matches:
        if m.distance < precision * n.distance:
            good_matches.append(m)
    if 29 < len(good_matches):  # menší číslo => přísnější kritérium
        break
    else:
        precision += 0.025

good_matches.sort(key=lambda d: d.distance)

"""cv2.rectangle(image1, (x1_1, y1_1), (x2_1, y2_1), (0, 255, 0), 10)
# cv2.circle(area, (center_x, center_y), radius, (0, 255, 0), 10)"""

# Vykreslení shod na obrazu
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Výpis souřadnic klíčových bodů
for i in range(2):
    x, y = keypoints1[i].pt
    r = keypoints1[i].response
    print(f"Souřadnice bodu: ({x:.2f}, {y:.2f}),\t Důvěryhodnost: {r:.6f}")

print("")

for j in range(2):
    x, y = keypoints2[j].pt
    r = keypoints2[j].response
    print(f"Souřadnice bodu: ({x:.2f}, {y:.2f}),\t Důvěryhodnost: {r:.6f}")


"""
x1, y1, x2, y2 = [], [], [], []
for g in range(len(good_matches)):
    pt1 = keypoints1[g].pt
    pt2 = keypoints2[g].pt
    x1.append(pt1[0])
    y1.append(pt1[1])
    x2.append(pt2[0])
    y2.append(pt2[1])
"""

x1, y1 = zip(*[keypoints1[m.queryIdx].pt for m in good_matches])
x2, y2 = zip(*[keypoints2[m.queryIdx].pt for m in good_matches])
angles1 = [keypoints1[m.queryIdx].angle for m in good_matches]
angles2 = [keypoints2[m.queryIdx].angle for m in good_matches]


dist1_x, dist1_y = 1521, 1457
dist2_x, dist2_y = 4338, 1468

dist_mm = 120
dist_px = np.sqrt((dist2_x - dist1_x) ** 2 + (dist2_y - dist1_y) ** 2)

scale = dist_mm / dist_px

x1, y1, x2, y2 = np.array(x1), np.array(y1), np.array(x2), np.array(y2)

print("\nPočet dat:", len(good_matches), ", koeficient přesnost: ", round(precision, 2))

print("\nx", round(abs(np.mean(x1 - x2)), 4), "px")
print("y", round(abs(np.mean(y1 - y2)), 4), "px")

print("\nx", round(abs(np.mean(x1 - x2) * scale), 4), "mm")
print("y", round(abs(np.mean(y1 - y2) * scale), 4), "mm")
print("dist:", round(np.sqrt(np.mean(x1 - x2) ** 2 + np.mean(y1 - y2) ** 2) * scale, 4), "mm")

# Zobrazení výsledku
cv2.namedWindow('Matched image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Matched image', int(0.25 * width), int(0.25 * height))
cv2.imshow("Matched image", matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""import cv2

# Načtení obrazů
# image1 = cv2.imread('photos/IMG_0385.JPG', 0)  # První fotografie (šedotónová)
# image2 = cv2.imread('photos/IMG_0417.JPG', 0)  # Druhá fotografie (šedotónová)

img_grey = cv2.imread('photos/IMG_0385.JPG', cv2.IMREAD_GRAYSCALE)

# Načtení obrázku
img = cv2.imread('photos/IMG_0385.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Vytvoření objektu pro detekci SIFT klíčových bodů
sift = cv2.SIFT_create()
kp = sift.detect(gray, None)

# Zakreslení klíčových bodů na obrázek pomocí kruhů
img_with_key_points = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# Zobrazení výsledku
height, width, _ = img_with_key_points.shape

# Zobrazení obrázků vedle sebe
combined_img = cv2.hconcat([img, img_with_key_points])

# Zobrazení kombinovaného obrázku
cv2.namedWindow('Combined image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Combined image', int(0.25 * width), int(0.25 * height))
cv2.imshow("Combined image", combined_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Uložení obrázku s klíčovými body
#cv2.imwrite('image-with-keypoints.jpg', img_with_keypoints)
"""
