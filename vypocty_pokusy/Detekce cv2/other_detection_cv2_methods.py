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

bf = cv2.BFMatcher()

mask1 = np.zeros(image1.shape[:2], dtype=np.uint8)
mask2 = np.zeros(image2.shape[:2], dtype=np.uint8)
cv2.circle(mask1, (px1, py1), radius, 1, -1)
cv2.circle(mask2, (px2, py2), radius, 1, -1)


def calculate(method):
    keypoints1, descriptors1 = method.detectAndCompute(image1, mask1)
    keypoints2, descriptors2 = method.detectAndCompute(image2, mask2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_image


print("Začátek")

m = cv2.SIFT_create()
image_sift = calculate(m)

print("1 hotový")

# Inicializace detektoru a popisovače FAST
fast = cv2.FastFeatureDetector_create()
brief = cv2.cv2.xfeatures2d

# Detekce klíčových bodů
keypoints1 = fast.detect(image1, None)
keypoints2 = fast.detect(image2, None)

# Výpočet popisů klíčových bodů
keypoints1, descriptors1 = brief.compute(image1, keypoints1)
keypoints2, descriptors2 = brief.compute(image2, keypoints2)

# Inicializace BFMatcher s Hammingovou vzdáleností (vhodná pro binární popisy)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Párování klíčových bodů
matches = bf.match(descriptors1, descriptors2)

# Seřazení shod podle vzdálenosti
matches = sorted(matches, key=lambda x: x.distance)

# Vytvoření výstupního obrázku s vizualizací shod
output_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

print("2 hotový")

"""star = cv2.xfeatures2d_StarDetector()
brief = cv2.xfeatures2d_BriefDescriptorExtractor()
kp1 = star.detect(image1, mask1)
kp2 = star.detect(image2, mask2)
kp1, des1 = brief.compute(image1, kp1)
kp2, des2 = brief.compute(image2, kp2)
matches = bf.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
image_brief = cv2.drawMatches(image1, kp1, image2, kp2, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)"""

print("3 hotový")

m = cv2.xfeatures2d_SURF(400)
kp, des = m.detectAndCompute(image1,mask1)

print("4 hotový")

print("5 hotový")

print("6 hotový")

# Zobrazení klíčových bodů na obrázcích
"""image_with_keypoints_normal1 = cv2.drawKeypoints(image1, keypoints_normal1, None)
image_with_keypoints_small1 = cv2.drawKeypoints(image_small1, keypoints_small1, None)
image_with_keypoints_smaller1 = cv2.drawKeypoints(image_smaller1, keypoints_smaller1, None)
image_with_keypoints_large1 = cv2.drawKeypoints(image_large1, keypoints_large1, None)"""

# Zobrazení obrázků s klíčovými body
plt.figure()
plt.subplot(231)
plt.title("Keypoints (SIFT)")
plt.imshow(cv2.cvtColor(image_sift, cv2.COLOR_BGR2RGB))
plt.subplot(232)
"""plt.title("Keypoints (FAST)")
plt.imshow(cv2.cvtColor(image_fast, cv2.COLOR_BGR2RGB))
plt.subplot(233)
plt.title("Keypoints (BRIEF)")
plt.imshow(cv2.cvtColor(image_brief, cv2.COLOR_BGR2RGB))"""
plt.subplot(234)
plt.title("Keypoints (SURF)")
plt.imshow(cv2.cvtColor(image_surf, cv2.COLOR_BGR2RGB))
"""plt.subplot(235)
plt.title("Keypoints (Small)")
plt.imshow(cv2.cvtColor(matched_image_small, cv2.COLOR_BGR2RGB))
plt.subplot(236)
plt.title("Keypoints (Smaller)")
plt.imshow(cv2.cvtColor(matched_image_smaller, cv2.COLOR_BGR2RGB))"""

plt.tight_layout()
plt.show()
