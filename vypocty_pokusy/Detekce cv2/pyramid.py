import cv2
import numpy as np
import matplotlib.pyplot as plt

# Načtení obrázku
image1 = cv2.imread('photos/IMG_0385.JPG', cv2.IMREAD_GRAYSCALE)[1100:1170, 2470:2540]
image2 = cv2.imread('photos/IMG_0417.JPG', cv2.IMREAD_GRAYSCALE)[1750:1820, 2500:2570]

points_to_track = np.int32(
    [[2508 - 2470, 1140 - 1100], [1935, 2130], [4228, 2138], [1932, 2131], [2515, 1145], [3648, 3128],
     [2979, 2105]])
point2 = np.int32([[2535 - 2500, 1775 - 1750]])

index = 0
radius = 28

px1, py1 = points_to_track[index]
px2, py2 = point2[index]

# Inicializace detektoru SIFT
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

print("Začátek")

mask1 = np.zeros(image1.shape[:2], dtype=np.uint8)
mask2 = np.zeros(image2.shape[:2], dtype=np.uint8)
cv2.circle(mask1, (px1, py1), radius, 1, -1)
cv2.circle(mask2, (px2, py2), radius, 1, -1)
keypoints_normal1, descriptors_normal1 = sift.detectAndCompute(image1, mask1)
keypoints_normal2, descriptors_normal2 = sift.detectAndCompute(image2, mask2)
matches = bf.knnMatch(descriptors_normal1, descriptors_normal2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
matched_image = cv2.drawMatches(image1, keypoints_normal1, image2, keypoints_normal2, good_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

print("NORMAL hotový")

# Scale-space pyramid - Zmenšení obrazu na 1/2 velikosti
image_small1 = cv2.pyrDown(image1)
image_small2 = cv2.pyrDown(image2)
mask_small1 = np.zeros(image_small1.shape[:2], dtype=np.uint8)
mask_small2 = np.zeros(image_small2.shape[:2], dtype=np.uint8)
cv2.circle(mask_small1, (px1 // 2, py1 // 2), radius // 2, 1, -1)
cv2.circle(mask_small2, (px2 // 2, py2 // 2), radius // 2, 1, -1)
keypoints_small1, descriptors_small1 = sift.detectAndCompute(image_small1, mask_small1)
keypoints_small2, descriptors_small2 = sift.detectAndCompute(image_small2, mask_small2)
matches = bf.knnMatch(descriptors_small1, descriptors_small2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
matched_image_small = cv2.drawMatches(image_small1, keypoints_small1, image_small2, keypoints_small2, good_matches,
                                      None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

print("SMALL hotový")

# Scale-space pyramid - Zmenšení obrazu na 1/4 velikosti
image_smaller1 = cv2.pyrDown(image_small1)
image_smaller2 = cv2.pyrDown(image_small2)
mask_smaller1 = np.zeros(image_smaller1.shape[:2], dtype=np.uint8)
mask_smaller2 = mask_smaller1.copy()
cv2.circle(mask_smaller1, (px1 // 4, py1 // 4), radius // 4, 1, -1)
cv2.circle(mask_smaller2, (px2 // 4, py2 // 4), radius // 4, 1, -1)
keypoints_smaller1, descriptors_smaller1 = sift.detectAndCompute(image_smaller1, mask_smaller1)
keypoints_smaller2, descriptors_smaller2 = sift.detectAndCompute(image_smaller2, mask_smaller2)
matches = bf.knnMatch(descriptors_smaller1, descriptors_smaller2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
matched_image_smaller = cv2.drawMatches(image_smaller1, keypoints_smaller1, image_smaller2, keypoints_smaller2,
                                        good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

print("SMALLER hotový")

# Scale-space pyramid - Zvětšení obrazu
image_large1 = cv2.pyrUp(image1)
image_large2 = cv2.pyrUp(image2)
mask_large1 = np.zeros(image_large1.shape[:2], dtype=np.uint8)
mask_large2 = np.zeros(image_large2.shape[:2], dtype=np.uint8)
cv2.circle(mask_large1, (np.int32(px1 * 2), np.int32(py1 * 2)), np.int32(radius * 2), 1, -1)
cv2.circle(mask_large2, (np.int32(px2 * 2), np.int32(py2 * 2)), np.int32(radius * 2), 1, -1)
keypoints_large1, descriptors_large1 = sift.detectAndCompute(image_large1, mask_large1)
keypoints_large2, descriptors_large2 = sift.detectAndCompute(image_large2, mask_large2)
matches = bf.knnMatch(descriptors_large1, descriptors_large2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
matched_image_large = cv2.drawMatches(image_large1, keypoints_large1, image_large2, keypoints_large2, good_matches,
                                      None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

print("LARGE hotový")

# Scale-space pyramid - Zvětšení obrazu zpět
image_largeb1 = cv2.pyrUp(image_small1)
image_largeb2 = cv2.pyrUp(image_small2)
mask_largeb1 = np.zeros(image_largeb1.shape[:2], dtype=np.uint8)
mask_largeb2 = np.zeros(image_largeb2.shape[:2], dtype=np.uint8)
cv2.circle(mask_largeb1, (px1, py1), radius, 1, -1)
cv2.circle(mask_largeb2, (px2, py2), radius, 1, -1)
keypoints_largeb1, descriptors_largeb1 = sift.detectAndCompute(image_largeb1, mask_largeb1)
keypoints_largeb2, descriptors_largeb2 = sift.detectAndCompute(image_largeb2, mask_largeb2)
matches = bf.knnMatch(descriptors_largeb1, descriptors_largeb2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
matched_image_largeb = cv2.drawMatches(image_largeb1, keypoints_largeb1, image_largeb2, keypoints_largeb2, good_matches,
                                       None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

print("LARGE Back hotový")

# Scale-space pyramid - Zvětšení obrazu zpět
image_largebb1 = cv2.pyrDown(image_large1)
image_largebb2 = cv2.pyrDown(image_large2)
mask_largebb1 = np.zeros(image_largebb1.shape[:2], dtype=np.uint8)
mask_largebb2 = np.zeros(image_largebb2.shape[:2], dtype=np.uint8)
cv2.circle(mask_largebb1, (px1, py1), radius, 1, -1)
cv2.circle(mask_largebb2, (px2, py2), radius, 1, -1)
keypoints_largebb1, descriptors_largebb1 = sift.detectAndCompute(image_largebb1, mask_largebb1)
keypoints_largebb2, descriptors_largebb2 = sift.detectAndCompute(image_largebb2, mask_largebb2)
matches = bf.knnMatch(descriptors_largebb1, descriptors_largebb2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
matched_image_largebb = cv2.drawMatches(image_largebb1, keypoints_largebb1, image_largebb2, keypoints_largebb2,
                                        good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

print("LARGE small hotový")

# Zobrazení klíčových bodů na obrázcích
"""image_with_keypoints_normal1 = cv2.drawKeypoints(image1, keypoints_normal1, None)
image_with_keypoints_small1 = cv2.drawKeypoints(image_small1, keypoints_small1, None)
image_with_keypoints_smaller1 = cv2.drawKeypoints(image_smaller1, keypoints_smaller1, None)
image_with_keypoints_large1 = cv2.drawKeypoints(image_large1, keypoints_large1, None)"""

# Zobrazení obrázků s klíčovými body
plt.figure()
plt.subplot(231)
plt.title("Keypoints (Normal)")
plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
plt.subplot(232)
plt.title("Keypoints (Large)")
plt.imshow(cv2.cvtColor(matched_image_large, cv2.COLOR_BGR2RGB))
plt.subplot(233)
plt.title("Keypoints (Small to Large)")
plt.imshow(cv2.cvtColor(matched_image_largeb, cv2.COLOR_BGR2RGB))
plt.subplot(234)
plt.title("Keypoints (Large to Small)")
plt.imshow(cv2.cvtColor(matched_image_largebb, cv2.COLOR_BGR2RGB))
plt.subplot(235)
plt.title("Keypoints (Small)")
plt.imshow(cv2.cvtColor(matched_image_small, cv2.COLOR_BGR2RGB))
plt.subplot(236)
plt.title("Keypoints (Smaller)")
plt.imshow(cv2.cvtColor(matched_image_smaller, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()
