import cv2
import numpy as np

# Načtěte první obrázek
image1 = cv2.imread(r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\Hexagons-Diploma_thesis'
                    r'\vypocty_pokusy\photos\IMG_0385.JPG', cv2.IMREAD_GRAYSCALE)
image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)[1050:3200, 1550:4630]

# Načtěte druhý obrázek
image2 = cv2.imread(r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\Hexagons-Diploma_thesis'
                    r'\vypocty_pokusy\photos\IMG_0400.JPG', cv2.IMREAD_GRAYSCALE)
image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)[1050:3200, 1550:4630]

# Počet prvků ve směru x a y
num_points_x = 30
num_points_y = 30

# Vytvoření mřížky bodů
x_coords = np.linspace(0, image1.shape[1], num_points_x)[:-1]
y_coords = np.linspace(0, image1.shape[0], num_points_y)[:-1]
grid_x, grid_y = np.meshgrid(x_coords, y_coords)

grid_x = grid_x.astype(np.int32)
grid_y = grid_y.astype(np.int32)

# Barva a tloušťka obdélníků
rectangle_color = (0, 255, 0)  # Zelená barva (BGR formát)
rectangle_thickness = 3

step_x = int(image1.shape[1] / (num_points_x - 1))
step_y = int(image1.shape[0] / (num_points_y - 1))

image1_grid, image2_grid = image1.copy(), image2.copy()
# Procházení bodů mřížky
for x, y in zip(grid_x.flatten(), grid_y.flatten()):
    # Vykreslení obdélníku pro každý bod mřížky
    cv2.rectangle(image1_grid, (x, y), (x + step_x, y + step_y), rectangle_color, rectangle_thickness)
    cv2.rectangle(image2_grid, (x, y), (x + step_x, y + step_y), rectangle_color, rectangle_thickness)

# Zobrazení výsledků
cv2.namedWindow('Image 1 with Grid', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image 1 with Grid', 800, 500)
cv2.namedWindow('Image 2 with Grid', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image 2 with Grid', 800, 500)
cv2.imshow('Image 1 with Grid', image1_grid)
cv2.imshow('Image 2 with Grid', image2_grid)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Inicializace detektoru a deskriptoru SIFT
sift = cv2.SIFT_create()

# Nalezení klíčových bodů a deskriptorů pro obě fotografie
keypoints1, descriptors1_sift = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Inicializace BFMatcheru (Brute-Force Matcheru)
bf = cv2.BFMatcher()

# Párování klíčových bodů mezi oběma fotografiemi
# matches = bf.match(descriptors1_sift, descriptors2)
# matches = sorted(matches, key=lambda x: x.distance)
matches = bf.knnMatch(descriptors1_sift, descriptors2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

# Seznam odpovídajících párů bodů uvnitř čtverečku
new_image = image1.copy()

step_x = step_x + 1
step_y = step_y + 1

points_pos = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 2)
points_neg = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 2)

# Průchod každým čtverečkem mřížky
for x, y in zip(grid_x.flatten(), grid_y.flatten()):
    conditions = np.where((x <= points_pos[:, 0]) & (points_pos[:, 0] < x + step_x) &
                          (y <= points_pos[:, 1]) & (points_pos[:, 1] < y + step_y))

    if len(conditions[0]) > 3:
        """for i in conditions[0]:
            cv2.circle(image1_grid, (np.int32(points_pos[i, 0]), np.int32(points_pos[i, 1])), 5, (255, 0, 0), -1)
            cv2.circle(image2_grid, (np.int32(points_neg[i, 0]), np.int32(points_neg[i, 1])), 5, (255, 0, 0), -1)

        # Zobrazení výsledků
        cv2.namedWindow('Image 1 with Grid', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image 1 with Grid', 800, 500)
        cv2.namedWindow('Image 2 with Grid', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image 2 with Grid', 800, 500)
        cv2.imshow('Image 1 with Grid', image1_grid)
        cv2.imshow('Image 2 with Grid', image2_grid)

        cv2.namedWindow('Grid', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Grid', 300, 300)
        cv2.imshow('Grid', image1[y:y + step_y, x:x + step_x])
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""
        M, _ = cv2.findHomography(points_pos[conditions].reshape(-1, 1, 2),
                                  points_neg[conditions].reshape(-1, 1, 2), cv2.RANSAC)
        img = image1[y:y + step_y, x:x + step_x]
        transformed_square = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
        new_image[y:y + step_y, x:x + step_x] = transformed_square
        # new_image = cv2.add(new_image, transformed_square)

        """cv2.namedWindow('Grid', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Grid', 300, 300)
        cv2.imshow('Grid', img)
        cv2.namedWindow('g', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('g', 300, 300)
        cv2.imshow('g', transformed_square)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""

# Zobrazení výsledku
cv2.namedWindow("New Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("New Image", 800, 500)
cv2.imshow("New Image", new_image)
cv2.waitKey(0)

conditions = np.where((x <= points_pos[:, 0]) & (points_pos[:, 0] < x + step_x) &
                      (y <= points_pos[:, 1]) & (points_pos[:, 1] < y + step_y))

M, _ = cv2.findHomography(points_pos[conditions].reshape(-1, 1, 2), points_neg[conditions].reshape(-1, 1, 2), cv2.RANSAC)
transformed_square = cv2.warpPerspective(image1, M, (image1.shape[1], image1.shape[0]))
new_image[y:y + step_y, x:x + step_x] = transformed_square

homography, _ = cv2.findHomography(points_neg.reshape(-1, 1, 2), points_pos.reshape(-1, 1, 2), cv2.RANSAC, 5.0)

# Transformace druhé fotografie na základě homografické matice
transformed_image = cv2.warpPerspective(image2, homography, (image2.shape[1], image2.shape[0]))

# Spojení obou fotografií
result = cv2.hconcat([image1, transformed_image])

# Zobrazení výsledného obrazu
cv2.namedWindow('Výsledek', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Výsledek', 800, 500)
cv2.imshow('Výsledek', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""# Počet bodů pro identifikaci čtverečku
min_points_per_square = 4

# Seznam pro ukládání indexů čtverečků s dostatečným počtem bodů
squares_with_enough_points = []

# Procházení bodů mřížky
for i, (x, y) in enumerate(zip(grid_x.flatten(), grid_y.flatten())):
    # Počet bodů v aktuálním čtverečku
    points_in_square = 0

    # Procházení nalezených bodů
    for m in matches:
        if x <= keypoints1[m.queryIdx].pt[0] <= x + step_x and y <= keypoints1[m.queryIdx].pt[1] <= y + step_y:
            points_in_square += 1

    # Kontrola, zda čtvereček obsahuje dostatečný počet bodů
    if points_in_square >= min_points_per_square:
        squares_with_enough_points.append(i)

print(squares_with_enough_points)

matched_keypoints1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
matched_keypoints2 = np.array([keypoints2[m.trainIdx].pt for m in matches])

# Výpočet rozsahů x a y pro čtverečky
x_ranges = np.column_stack((grid_x.flatten(), grid_x.flatten() + step_x))
y_ranges = np.column_stack((grid_y.flatten(), grid_y.flatten() + step_y))

# Kontrola polohy klíčových bodů v čtverečcích
for i in range((num_points_x - 1) * (num_points_y - 1)):
    if x_ranges[:, 0] <= matched_keypoints1[i, 0] <= x_ranges[:, 1] and \
            y_ranges[:, 0] <= matched_keypoints1[i, 1] <= y_ranges[:, 1]:


# Indexy čtverečků s dostatečným počtem bodů
squares_with_enough_points = np.where(points_in_squares >= min_points_per_square)[0]

print(squares_with_enough_points)"""

"""# Zobrazení výsledku pro daný bod
    cv2.namedWindow('Transformed Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Transformed Image', 800, 500)
    cv2.imshow('Transformed Image', transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

"""x1, y1 = 3919, 2643
xf, yf = 4102, 2840

# Označení bodu na první fotografii

cv2.namedWindow('Fotografie 1 s označeným bodem', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Fotografie 1 s označeným bodem', 800, 500)
cv2.circle(image1, (x1, y1), 20, (0, 0, 255), -1)
cv2.imshow('Fotografie 1 s označeným bodem', image1)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Inicializace detektoru a deskriptoru SIFT
sift = cv2.SIFT_create()

# Nalezení klíčových bodů a deskriptorů pro obě fotografie
keypoints1, descriptors1_sift = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Inicializace BFMatcheru (Brute-Force Matcheru)
bf = cv2.BFMatcher()

# Párování klíčových bodů mezi oběma fotografiemi
matches = bf.match(descriptors1_sift, descriptors2)

# Seřazení shod podle vzdálenosti
matches = sorted(matches, key=lambda x: x.distance)

# Výběr nejlepších shod (např. 10 shod)
best_matches = matches[:4]

# Seznam bodů z první a druhé fotografie
points_pos = np.float32([keypoints1[match.queryIdx].pt for match in best_matches]).reshape(-1, 1, 2)
points_neg = np.float32([keypoints2[match.trainIdx].pt for match in best_matches]).reshape(-1, 1, 2)

# Nalezení homografické matice
homography, _ = cv2.findHomography(points_pos, points_neg, cv2.RANSAC, 5.0)

# Definice souřadnic bodu z první fotografie
point1 = np.array([[x1, y1]], dtype='float32')

# Přidání jedné dimenze k bodu
point1 = np.expand_dims(point1, axis=0)

# Transformace souřadnic bodu na druhé fotografii
point2 = cv2.perspectiveTransform(point1, homography)

# Nové souřadnice bodu na druhé fotografii
x2, y2 = point2[0][0]


print(x1, y1)
print(x2, y2)

# Označení bodu na druhé fotografii
cv2.namedWindow('Fotografie 2 s označeným bodem', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Fotografie 2 s označeným bodem', 800, 500)
cv2.circle(image2, (int(xf), int(yf)), 20, (100, 0, 100), -1)
cv2.circle(image2, (int(x2), int(y2)), 20, (0, 0, 255), -1)
cv2.circle(image2, (x1, y1), 20, (0, 255, 0), -1)
cv2.imshow('Fotografie 2 s označeným bodem', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Transformace druhé fotografie na základě homografické matice
transformed_image = cv2.warpPerspective(image2, homography, (image1.shape[1], image1.shape[0]))

# Označení bodu na transformované fotografii
cv2.circle(transformed_image, (x1, y1), 5, (0, 0, 255), -1)

# Spojení obou fotografií
result = cv2.hconcat([image1, transformed_image])

# Zobrazení výsledného obrazu
cv2.namedWindow('Výsledek', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Výsledek', 800, 500)
cv2.imshow('Výsledek', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
