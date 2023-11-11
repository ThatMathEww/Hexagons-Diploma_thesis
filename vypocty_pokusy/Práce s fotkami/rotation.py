import cv2
import numpy as np
import matplotlib.pyplot as plt

# Načtěte fotografii
p = (r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos\H01_04_12s\modified'
     r'\IMG_0489_mod.JPG')
img = cv2.imread(p, 0)

point = np.array((1040, 820))

# Získání rozměrů obrázku
height, width = img.shape[:2]

# Vypočtěte střed obrázku
center = (width // 2, height // 2)

# Definujte transformační matici pro rotaci o 35 stupňů

s = (772, 312)
e = (2465, 320)
angle = np.rad2deg(np.arctan2(e[1] - s[1], e[0] - s[0]))
matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

# Aplikujte transformační matici na obrázek
rotated_img = cv2.warpAffine(img, matrix, (width, height))
# Aplikujte transformační matici na body
transformed_points = cv2.transform(point.reshape(1, -1, 2), matrix).reshape(2, )

plt.figure()
plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.scatter(*point, c='orange')
plt.subplot(122)
plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
plt.scatter(*transformed_points, c='orange')
plt.show()
