import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_affine_transformation_matrix(source_points, target_points):
    """
    Vytvoří afinní transformační matici na základě tří bodů.
    """
    source_points = np.array(source_points, dtype=np.float32)
    target_points = np.array(target_points, dtype=np.float32)

    # Vytvoříme afinní transformační matici pomocí cv2.getAffineTransform
    transformation_matrix = cv2.getAffineTransform(source_points, target_points)

    return transformation_matrix


def transform_point(transformation_matrix, point):
    # Transformuje bod pomocí transformační matice.

    point_ = np.array(point, dtype=np.float32)  # TODO np.float64 - přesnost ????

    # Přidáme homogenní souřadnici (1) k bodu
    point_homog = np.append(point_, [1])

    # Použijeme afinní transformační matici na bod
    transformed_point_homog = np.dot(transformation_matrix, point_homog)

    # Převedeme homogenní souřadnice zpět na 2D souřadnice (x, y, 1) -> (x, y)
    transformed_point_1 = (transformed_point_homog[0], transformed_point_homog[1])

    # Transformuje bod pomocí transformační matice.

    point_ = np.array(point, dtype=np.float64)  # TODO np.float64 - přesnost ????

    # Přidáme homogenní souřadnici (1) k bodu
    point_homog = np.append(point_, [1])

    # Použijeme afinní transformační matici na bod
    transformed_point_homog = np.dot(transformation_matrix, point_homog)

    # Převedeme homogenní souřadnice zpět na 2D souřadnice (x, y, 1) -> (x, y)
    transformed_point_2 = (transformed_point_homog[0], transformed_point_homog[1])

    transformed__point = ((transformed_point_1[0] + transformed_point_2[0]) / 2,
                          (transformed_point_1[1] + transformed_point_2[1]) / 2)

    return transformed__point


def get_affine_matrix(A, B):
    """# Výpočet afinní transformační matice mezi body A a B
    # Předpokládá se, že A a B jsou pole se dvěma body každým
    # Přidáme třetí řádek [0, 0, 1] pro zajištění afinní transformace
    A_extended = np.vstack((A.T, np.ones((1, A.shape[0]))))
    B_extended = np.vstack((B.T, np.ones((1, B.shape[0]))))
    M = np.dot(B_extended, np.linalg.pinv(A_extended))
    return M[:2, :]"""

    #  ### Přesnější ### #

    # Výpočet afinní transformační matice mezi body A a B
    # Předpokládá se, že A a B jsou pole se dvěma body každým
    A_extended = np.vstack((A.T, np.ones((1, A.shape[0]))))
    B_extended = np.vstack((B.T, np.ones((1, B.shape[0]))))
    M, _ = np.linalg.lstsq(A_extended.T, B_extended.T, rcond=None)[:2]
    return M.T


# Ukázkové body pro demonstraci
source_points = np.array([(2229.777777777778, 1562.7777777777776), (2248.0, 1648.0), (2272.0, 1595.0)])
target_points = np.array([(2089.6251772501255, 2006.0266479862269), (2062.4620285910282, 2087.7634755368695),
                          (2110.4087925380186, 2055.4892163508757)])

image = cv2.imread('photos/IMG_0385.JPG', cv2.IMREAD_GRAYSCALE)
height, width = image.shape[:2]

center = np.mean(source_points, axis=0)

# Vytvoření afinní transformační matice
transformation_matrix = create_affine_transformation_matrix(source_points, target_points)
M = get_affine_matrix(source_points, target_points)

# inv_M = cv2.invert(transformation_matrix)[1]
# inverse_transformation_matrix = np.linalg.inv(transformation_matrix)
inverse_transformation_matrix = create_affine_transformation_matrix(target_points, source_points)
M_inv = get_affine_matrix(target_points, source_points)

# Transformace bodu ze source_points na target_points
point_to_transform = (2229.777777777778, 1562.7777777777776)
correct_point = (2089.6251772501255, 2006.0266479862269)
transformed_point = transform_point(transformation_matrix, point_to_transform)
print(point_to_transform)
print(transformation_matrix)
print(transformed_point)
print(M)
re_transformed_point = transform_point(inverse_transformation_matrix, transformed_point)

print("Původní bod:", point_to_transform)
print("Transformovaný bod:", transformed_point)
print("Přetransformovaný bod:", re_transformed_point)
# print("Správný bod:", np.mean(target_points, axis=0))

transformed_point2 = transform_point(M, point_to_transform)
re_transformed_point2 = transform_point(M_inv, transformed_point2)

print("Transformovaný bod:", transformed_point2)
print("Přetransformovaný bod:", re_transformed_point2)

print(np.linalg.norm(np.array(point_to_transform) - re_transformed_point))
print(np.linalg.norm(np.array(point_to_transform) - re_transformed_point2))

# Aplikace transformace pomocí cv2.warpAffine
rotated_image = cv2.warpAffine(image, transformation_matrix, (width, height))
rotated_image2 = cv2.warpPerspective(image, M, (width, height))

c = np.array(point_to_transform, dtype=np.float32).reshape(-1, 1, 2)
a = cv2.perspectiveTransform(np.array(point_to_transform, dtype=np.float32).reshape(-1, 1, 2), M).reshape(2)
print(list(a))
b = cv2.perspectiveTransform(np.array(a, dtype=np.float32).reshape(-1, 1, 2), M_inv).reshape(2)
print(list(b))

print("\nPřesnosti")
print(np.linalg.norm(np.array(correct_point) - transformed_point))
print(np.linalg.norm(np.array(correct_point) - transformed_point2))
print(np.linalg.norm(np.array(correct_point) - a))

print(cv2.perspectiveTransform(np.array(transformed_point2, dtype=np.float32).reshape(-1, 1, 2), M_inv).reshape(2))

plt.figure()
plt.subplot(131)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
plt.scatter(point_to_transform[0], point_to_transform[1])
plt.subplot(132)
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_GRAY2BGR))
plt.scatter(transformed_point[0], transformed_point[1])
plt.subplot(133)
plt.imshow(cv2.cvtColor(rotated_image2, cv2.COLOR_GRAY2BGR))
plt.scatter(transformed_point2[0], transformed_point2[1])
plt.tight_layout()
plt.show()
