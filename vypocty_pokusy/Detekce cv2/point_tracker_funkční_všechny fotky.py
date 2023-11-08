import os
import cv2
import numpy as np

##########################################
# Hledání pomocí "calcOpticalFlowPyrLK"
# mezi všemi fotkami
# Pouze body
##########################################

dist1 = 100
dist2 = 220
dist1_x, dist1_y = 1521, 1457
dist2_x, dist2_y = 4338, 1468

# Nastavení cesty k složce s obrázky
image_folder = 'photos'

# Načtení seznamu obrázků ve složce
file_list = os.listdir(image_folder)
image_files = [f for f in file_list if os.path.isfile(os.path.join(image_folder, f))]

# Omezeni počtu snímků
image_files = image_files[:]  # jaké snímky budu načítat (první je 0) př: "image_files[2:5] od 2 do 5"

# Definice oblasti
x_min, x_max = 2650, 3450  # Minimální a maximální hodnota souřadnice x
y_min, y_max = 400, 650  # Minimální a maximální hodnota souřadnice y
grid_size_x = 31  # Velikost mřížky (rozdělení ve směru x)
grid_size_y = 7  # Velikost mřížky (rozdělení ve směru y)

# Vytvoření mřížky bodů
x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, grid_size_x),
                             np.linspace(y_min, y_max, grid_size_y))
points_to_track = np.column_stack((x_grid.ravel(), y_grid.ravel()))

len_grid = len(points_to_track)

# další body
next_points = np.array([[4228, 2138], [1932, 2131], [2515, 1145], [3648, 3128], [2979, 2105]])

points_to_track = np.append(points_to_track, next_points, axis=0)
del next_points  # zapomenutí "next_points"

# Převod souřadnic bodů na formát akceptovaný funkcí cv2.goodFeaturesToTrack
p0 = np.array(points_to_track, dtype=np.float32).reshape(-1, 1, 2)

# Nastavení parametrů pro sledování
lk_params = dict(winSize=(25, 25),  # 10, 10 - prohledávaná oblast
                 maxLevel=40,  # 55 - počet použitých snímků různé kvality
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
                 )  # 10, 0.03 - max počet iterací, odchylka

X = np.zeros((len(image_files), len(points_to_track)))
Y = np.zeros((len(image_files), len(points_to_track)))

X[0] = points_to_track[:, 0]
Y[0] = points_to_track[:, 1]

# Sekvence hledání mezi fotkami
for p in range(len(image_files) - 1):
    first_photo = image_files[p]
    second_photo = image_files[p + 1]

    # Načtení prvního snímku
    frame1 = cv2.imread(os.path.join(image_folder, first_photo))
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Načtení druhého snímku
    frame2 = cv2.imread(os.path.join(image_folder, second_photo))
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Výpočet optického toku
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

    # Výběr pouze správně vyhodnocených bodů
    point_new = p1[st == 1]
    point_old = p0[st == 1]

    X[p + 1] = p1[:, 0, 0]
    Y[p + 1] = p1[:, 0, 1]

    p0 = p1

    print("\nkrok: ", p + 1)
    print("První fotografie:\t", image_files[p])
    print("Druhá fotografie:\t", image_files[p + 1])

# VYKRESLENÍ
for d in range(1):
    photo = -1 + d  # kterou fotku chci použít (poslední a jaké další)

    window_name = "Posledni fotografie:   " + image_files[photo]
    frame = cv2.imread(os.path.join(image_folder, image_files[photo]))

    # které body se na konci nebudou vykreslovat
    point_del = 1

    # Vykreslení sledovaných bodů a jejich trajektorie
    for i in range(len(points_to_track) - point_del):
        a, b = X[-1][i].astype(int), Y[-1][i].astype(int)
        c, d = X[0][i].astype(int), Y[0][i].astype(int)
        frame = cv2.line(frame, (a, b), (c, d), (0, 255, 0), 3)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), 15)
        frame = cv2.circle(frame, (c, d), 5, (50, 0, 220), 15)
    cv2.rectangle(frame, (2650, 400), (3450, 650), (255, 255, 255), 3)

    # Vykreslení čáry průběhu bodu
    point_pos = -1  # který bod vykreslý trasu

    for i in range(len(image_files) - 1):
        x = int(X[i][point_pos])
        y = int(Y[i][point_pos])
        cv2.circle(frame, (x, y), 2, (0, 255, 255), 10)  # Vykreslení bodu
        if i < len(X) - 1:
            x_next = int(X[i + 1][point_pos])
            y_next = int(Y[i + 1][point_pos])
            # cv2.line(frame, (x, y), (x_next, y_next), (255, 0, 0), 10)  # Vykreslení čáry mezi body
    # Vykreslení čáry mezi body
    cv2.polylines(frame, [np.int32(np.vstack((X[:, point_pos], Y[:, point_pos])).T)], isClosed=False, color=(255, 0, 0),
                  thickness=5)

    # Získání rozměrů obrázku
    height, width, _ = frame.shape

    dif_x = abs(np.mean(X[-1, :len_grid] - points_to_track[:len_grid, 0]))
    dif_y = abs(np.mean(Y[-1, :len_grid] - points_to_track[:len_grid, 1]))

    dist_mm = dist2 - dist1
    dist_px = np.sqrt((dist2_x - dist1_x) ** 2 + (dist2_y - dist1_y) ** 2)

    final_dist_x_mm = dist_mm / dist_px * dif_x
    final_dist_y_mm = dist_mm / dist_px * dif_y

    print("\nVelikost fotografie:\t\t", height, "x", width, "px")
    print("Průměrný výškový rozdíl:\t\t", round(dif_y, 5), "px")
    print("Průměrný vodorovný rozdíl:\t\t", round(dif_x, 5), "px")

    print("\nPrůměrný výškový rozdíl:\t\t", round(final_dist_y_mm, 5), "mm")
    print("Průměrný vodorovný rozdíl:\t\t", round(final_dist_x_mm, 5), "mm")

    # Vytvoření okna s rozměry odpovídajícími oříznutému výřezu
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(window_name, int(0.25 * width), int(0.25 * height))

    # Zobrazení výsledného snímku s vykreslenými body a trajektoriemi
    cv2.imshow(window_name, frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
