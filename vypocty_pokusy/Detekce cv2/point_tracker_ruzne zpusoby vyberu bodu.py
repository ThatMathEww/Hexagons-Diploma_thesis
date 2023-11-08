import os
import sys
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

##########################################
# Hledání pomocí "calcOpticalFlowPyrLK"
# mezi dvěma fotkami
# Pouze body
##########################################

# Nastavení cesty k složce s obrázky
image_folder = 'photos'

# Načtení seznamu obrázků ve složce
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

###############################################################
# Které fotografie porovnávám
first_photo = image_files[0]
second_photo = image_files[32]  # [-1]

# Načtení prvního snímku
frame1 = cv2.imread(os.path.join(image_folder, first_photo))
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Načtení druhého snímku
frame2 = cv2.imread(os.path.join(image_folder, second_photo))
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

print("\nPrvní fotografie:\t", first_photo)
print("Druhá fotografie:\t", second_photo)

###############################################################
#         Měření souřadnic bodů pomocí matplotlib             #
"""# Převod barevného prostoru z BGR na RGB
image = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

# Vytvoření grafu pomocí Matplotlib
plt.imshow(image)
plt.axis('off')  # Vypnutí os
plt.tight_layout()
plt.show()
"""

x = None
y = None

#######################################################################################################################
#######################################################################################################################

# Způsob vytvořčení bodů
print("\nZvolte jednu z možností:"
      "\n\t 0 -     fixed coordinates"
      "\n\t 1 -     range"
      "\n\t 2 -     random"
      "\n\t 3 -     grid"
      "\n\t 4 -     some points"
      "\n\t 5 -     automatic corners (everywhere)"
      "\n\t 6 -     automatic corners (in ROI)")

data_type = input("\nZvolte typ výpočtu\t(0-6):\t")
print("Zvolena možnost:", data_type)
try:
    data_type = abs(int(data_type))
except ValueError:
    print("Chybné zadání.")
    sys.exit()

##############################
if data_type == 0:
    points_to_track = np.array([[3125, 2125], [2979, 2105]])
    # points_to_track = np.array([[3100, 650]])
    x = points_to_track[:, 0]
    y = points_to_track[:, 1]

##############################
elif data_type == 1:
    # Seznam souřadnic bodů x a y
    x = range(2650, 3450+80, 80)  # Doplnění x-ových souřadnic bodů
    y = range(400, 650+25, 25)  # Doplnění y-ových souřadnic bodů

    # Vytvoření seznamu souřadnic bodů pro sledování
    points_to_track = [(x[i], y[i]) for i in range(len(x))]

##############################
elif data_type == 2:
    # Počet bodů
    num_points = 20

    # Generování náhodných souřadnic pro x a y
    x = [random.randint(2650, 3450) for _ in range(num_points)]
    y = [random.randint(400, 650) for _ in range(num_points)]

    # Kontrola, zda mají oba seznamy stejnou délku
    if len(x) != len(y):
        print("Chyba: Seznamy x a y musí mít stejnou délku.")
        exit()

    # Vytvoření seznamu souřadnic bodů pro sledování
    points_to_track = [(x[i], y[i]) for i in range(len(x))]

##############################
elif data_type == 3:
    # Definice oblasti
    x_min, x_max = 2650, 3450  # Minimální a maximální hodnota souřadnice x
    y_min, y_max = 400, 650  # Minimální a maximální hodnota souřadnice y
    grid_size_x = 21  # Velikost mřížky (rozdělení)
    grid_size_y = 7  # Velikost mřížky (rozdělení)

    # Vytvoření mřížky bodů
    x, y = np.meshgrid(np.linspace(x_min, x_max, grid_size_x),
                       np.linspace(y_min, y_max, grid_size_y))
    points_to_track = np.column_stack((x.ravel(), y.ravel()))

##############################
elif data_type == 4:
    """x1, y1 = 3000, 550
    x2, y2 = 3100, 650
    x3, y3 = 2800, 600"""

    x = [3000, 3100, 2800]
    y = [550, 650, 600]

    # Vytvoření seznamu souřadnic bodů pro sledování
    points_to_track = [(x[i], y[i]) for i in range(len(x))]

##############################
elif data_type == 5 or data_type == 6:
    pass

##############################
else:
    print("Program zastaven: špatné zvolení kategorie způsobu vytvořčení bodů")
    sys.exit()

#######################################################################################################################
#######################################################################################################################

if data_type == 5 or data_type == 6:
    # Nastavení parametrů pro sledování
    lk_params = dict(winSize=(10, 10),
                     maxLevel=5,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 0.001))

    # Určení počátečních bodů pro sledování
    if data_type == 6:
        # Definice omezené oblasti
        x_1, y_1, x_2, y_2 = 2650, 400, 3450, 650

        # Vytvoření masky pro omezení oblasti
        mask = np.zeros(gray1.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x_1, y_1), (x_2, y_2), 255, -1)

        # Nastavení parametrů pro metodu goodFeaturesToTrack
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.7,
                              minDistance=5,
                              blockSize=3)  # citlivost -> větší = ignorace (detekte pixelů / šum)

        # Získání bodů v omezené oblasti pomocí goodFeaturesToTrack a masky
        p0 = cv2.goodFeaturesToTrack(gray1, mask=mask, **feature_params)

        # Příprava bodů pro sledování
        # p0 = points.reshape(-1, 1, 2).astype(np.float32)

    else:
        p0 = cv2.goodFeaturesToTrack(gray1, mask=None, maxCorners=500, qualityLevel=0.1, minDistance=50)

    # Výpočet optického toku
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

    plot_point_size = 2

    # Výběr pouze bodů
    point_new = p1[st == 1]
    point_old = p0[st == 1]

    x, y = point_old[:, 0], point_old[:, 1]

else:
    # Převod souřadnic bodů na formát akceptovaný funkcí cv2.goodFeaturesToTrack
    p0 = np.array(points_to_track, dtype=np.float32).reshape(-1, 1, 2)

    # Nastavení parametrů pro sledování
    lk_params = dict(winSize=(70, 70),
                     maxLevel=100,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 0.001))

    # Výpočet optického toku
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

    plot_point_size = 5

    x = np.array(x)
    y = np.array(y)

    # Výběr pouze bodů
    point_new = p1[st == 1]
    point_old = p0[st == 1]

x_new = []
y_new = []

# Vykreslení sledovaných bodů a jejich trajektorie
for j in range(2):
    frame = [frame1, frame2]
    for i, (new, old) in enumerate(zip(point_new, point_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        frame[j] = cv2.line(frame[j], (a, b), (c, d), (0, 255, 0), plot_point_size)
        frame[j] = cv2.circle(frame[j], (a, b), 5, (0, 0, 255), plot_point_size * 3)
        frame[j] = cv2.circle(frame[j], (c, d), 5, (0, 0, 255), plot_point_size * 3)
        if j == 0:
            x_new = np.append(x_new, new[0])
            y_new = np.append(y_new, new[1])
    cv2.rectangle(frame[j], (2650, 400), (3450, 650), (255, 255, 255), 3)

# Získání rozměrů obrázku
height, width, _ = frame2.shape

dif_x = abs(np.mean(x_new - x.reshape(len(x_new), )))
dif_y = abs(np.mean(y_new - y.reshape(len(y_new), )))

dist1 = 100
dist2 = 220
dist1_x, dist1_y = 1521, 1457
dist2_x, dist2_y = 4338, 1468

dist_mm = dist2 - dist1
dist_px = np.sqrt((dist2_x - dist1_x) ** 2 + (dist2_y - dist1_y) ** 2)

final_dist_x_mm = dist_mm / dist_px * dif_x
final_dist_y_mm = dist_mm / dist_px * dif_y

print("\nVelikost fotografie:\t\t", height, "x", width, "px")
print("Průměrný výškový rozdíl:\t\t", round(dif_y, 5), "px")
print("Průměrný vodorovný rozdíl:\t\t", round(dif_x, 5), "px")

print("\nPrůměrný výškový rozdíl:\t\t", round(final_dist_y_mm, 5), "mm")
print("Průměrný vodorovný rozdíl:\t\t", round(final_dist_x_mm, 5), "mm")

window_name_1 = 'Fotografie 1:   ' + first_photo
window_name_2 = 'Fotografie 2:   ' + second_photo

# Vytvoření okna s rozměry odpovídajícími oříznutému výřezu
cv2.namedWindow(window_name_1, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_name_2, cv2.WINDOW_NORMAL)

cv2.resizeWindow(window_name_1, int(0.25 * width), int(0.25 * height))
cv2.resizeWindow(window_name_2, int(0.25 * width), int(0.25 * height))

# Zobrazení výsledného snímku s vykreslenými body a trajektoriemi
cv2.imshow(window_name_1, frame1)
cv2.imshow(window_name_2, frame2)
cv2.waitKey(0)
cv2.destroyAllWindows()
