import numpy as np
import matplotlib.pyplot as plt
import cv2


def transform_points(points, end_point, start_point):
    delta_x = start_point[0] - end_point[0]
    delta_y = start_point[1] - end_point[1]
    angle_rad = np.arctan2(delta_y, delta_x) + np.pi / 2

    # Rotuje body kolem počátku o daný úhel
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])

    return np.dot((points - main_point_start), rotation_matrix) + main_point_start


def transform_photo(photo, end_point, start_point):
    angle_rad = np.arctan2(start_point[1] - end_point[1], start_point[0] - end_point[0]) + np.pi / 2

    # Rotuje body kolem počátku o daný úhel

    rotation_matrix = np.column_stack((np.array([[np.cos(angle_rad), np.sin(angle_rad)],
                                                 [-np.sin(angle_rad), np.cos(angle_rad)]]), [0, 0]))

    # Získání velikosti původního obrazu
    height, width = photo.shape[:2]

    return cv2.warpAffine(photo, rotation_matrix, (width, height))


# Ukázková data bodů
main_point_start = np.array([0, 5])  # Počáteční bod hlavního bodu
main_point_end = np.array([130, 220])  # Konečný bod hlavního bodu
other_points = np.array([[1, 1],
                         [2, 2],
                         [3, 3],
                         [3, 0],
                         [0, 2],
                         [580, 555]])

# other_points[:, 1] = other_points[:, 1] + 800

# Transformace Korekce souřadnic ostatních bodů
corrected_points = transform_points(other_points, main_point_end, main_point_start)
corrected_end_point = transform_points(main_point_end, main_point_end, main_point_start)

image = cv2.imread(r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\Hexagons-Diploma_thesis'
                   r'\vypocty_pokusy\book2.JPG', 0)
rotated_image = transform_photo(image, main_point_end, main_point_start)
height, width = image.shape[:2]

# Vykreslení výsledku
plt.figure()
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
plt.scatter(other_points[:, 0], other_points[:, 1], label='Původní body', zorder=3)
plt.scatter(corrected_points[:, 0], corrected_points[:, 1], label='Korigované body', zorder=3)
plt.plot([main_point_start[0], main_point_end[0]], [main_point_start[1], main_point_end[1]],
         'r-', label='Dráha hlavního bodu', zorder=2)
plt.plot([main_point_start[0], corrected_end_point[0]], [main_point_start[1], corrected_end_point[1]],
         'b--', label='Dráha hlavního opravého bodu', zorder=2)
plt.legend()
# plt.gca().invert_yaxis()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Korekce bodů na základě úhlu odchylky')
plt.axis('equal')
plt.tight_layout()
plt.grid(True)
plt.show()
