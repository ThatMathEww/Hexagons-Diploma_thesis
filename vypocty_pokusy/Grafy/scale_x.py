import matplotlib.pyplot as plt
import numpy as np

# Definice vrcholů polygonu (zde předpokládáme, že máte seznam vrcholů polygonu)
original_points = np.array([[100, 100], [150, 110], [200, 100], [150, 200], [120, 120]], dtype=np.float32) * 2

moved_points = original_points + (1000, 500)

# Zjistěte střed polygonu
center = np.mean(moved_points, axis=0)

# Přesuňte vrcholy tak, aby střed byl v počátku
translated_points = moved_points - center

angle = 5

# Úhel rotace v radiánech
angle_rad = np.deg2rad(angle)

# Vytvoření matice rotace
rot_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                    [np.sin(angle_rad), np.cos(angle_rad)]])

# Proveďte rotaci pro každý vrchol
rotated_points = np.dot(translated_points, rot_mat.T)

# Přesuňte vrcholy zpět na původní pozice
final_points = rotated_points + center

# Vytvoření polygonu
polygon = plt.Polygon(original_points, closed=True, fill=False, edgecolor='red', label='Polygon original')

x_max, x_min = 850, 0
c = ((x_max - x_min) // 50) + 1
a = max(min(((x_max - x_min) // 50) + 1, 7), 2)

scale_factor = 2  # Násobitel pro změnu souřadnic

dir = 0

coordinates = polygon.get_path().vertices[:-1].copy()

# Přidání polygonu do grafu
plt.gca().add_patch(polygon)

center = np.mean(coordinates[:, :], axis=0)

# Získání aktuálních souřadnic všech bodů polygonu
current_verts = polygon.get_xy()[:-1].copy()
current_verts = final_points

# Změna souřadnic bodů ve směru osy x od středu
current_verts[:, :] = center + scale_factor * (current_verts[:, :].copy() - center)

import cv2

# Odhad transformační matice
transform_matrix, _ = cv2.estimateAffinePartial2D(original_points, current_verts)

# Matice obsahuje informace o translaci (posunu) a rotaci
translation = transform_matrix[:, 2]
rotation = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])

print("Translace (posun):", translation)
print("Rotace (úhel):", np.round(np.degrees(rotation), decimals=4), np.linalg.norm(np.degrees(rotation) - angle))

# Odhad transformační matice pomocí cv2.findHomography()
translation_homo, _ = cv2.findHomography(original_points, current_verts)

# Matice obsahuje informace o posunu, rotaci a další transformaci

# Extrahujte translaci (posun) a rotaci
translation = translation_homo[:2, 2]
rotation = np.arctan2(translation_homo[1, 0], translation_homo[0, 0])

print("Translace (posun):", translation)
print("Rotace (úhel):", np.round(np.degrees(rotation), decimals=4), np.linalg.norm(np.degrees(rotation) - angle))

print(np.mean(current_verts, axis=0) - np.mean(original_points, axis=0))

# Rozdíl mezi počátečními a transformovanými body
diff = current_verts - original_points

# Průměr rozdílů ve směru x a y
mean_diff_x = np.mean(diff[:, 0])
mean_diff_y = np.mean(diff[:, 1])

# Odhad úhlu rotace
rotation_angle = np.arctan2(mean_diff_y, mean_diff_x)

# Převod na stupně
rotation_angle_degrees = np.degrees(rotation_angle)

print("Odhadnutý úhel rotace:", rotation_angle_degrees)

# Aktualizace polygonu s novými souřadnicemi
plt.gca().add_patch(plt.Polygon(current_verts, closed=True, fill=False, edgecolor='blue', label='Polygon transform'))

im = cv2.imread(r"C:\Users\matej\Downloads\IMG_0385.JPG", 1)
h, w = im.shape[:2]
rotated_image = cv2.warpAffine(im, transform_matrix, (w * 3, h * 3))
# rotated_image = cv2.warpPerspective(im, translation_homo, (w*3, h*3))

plt.imshow(rotated_image)

# Zobrazte legendu (jen pro vizualizaci)
plt.gcf().autofmt_xdate()
plt.legend()
plt.gca().axis('equal')
plt.gca().relim()
plt.gca().autoscale_view()
plt.show()
