import os
import numpy as np
import cv2

# Nastavení cesty k složce s obrázky
image_folder = 'photos'

# Načtení seznamu obrázků ve složce
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# Načtení prvního obrázku
image_path = os.path.join(image_folder, image_files[0])
image = cv2.imread(image_path)
gray_prev = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Funkce pro oříznutí fotografie

def crop_image(img):
    bbox = cv2.selectROI('Crop Image', img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    crop_img = img[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
    return crop_img, bbox


# Oříznutí prvního obrázku
crop_img, bbox = crop_image(image)

# Definice začátečních bodů, které chcete sledovat
points_to_track = []
selected_points = []
selected_distance = None


# Funkce pro označení bodu myší na snímku
def select_point(event, x, y, flags, param):
    global points_to_track, selected_points, selected_distance
    if event == cv2.EVENT_LBUTTONDOWN:
        points_to_track.append([x, y])
        if len(points_to_track) == 2:
            selected_points = points_to_track.copy()
            distance = input("Zadejte vzdálenost mezi označenými body v milimetrech: ")
            selected_distance = float(distance)
            cv2.setMouseCallback('Point Tracker', lambda *args: None)


# Označení bodů myší na oříznutém snímku
cv2.namedWindow('Point Tracker')
cv2.setMouseCallback('Point Tracker', select_point)
while len(points_to_track) < 2:
    cv2.imshow('Point Tracker', crop_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Inicializace point trackeru
points_status = np.ones(len(points_to_track), dtype=np.bool)
points_prev = np.array(points_to_track, dtype=np.float32)

# Definice parametrů pro sledování optických toků
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Sledování optických toků pro všechny obrázky ve složce
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Výpočet nových poloh bodů pomocí optického toku (Lucas-Kanade metoda)
    points_next, status, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray, points_prev, None, **lk_params)

    # Aktualizace stavu bodů a jejich pozic
    points_status = np.logical_and(status.squeeze(), points_status)
    points_prev = points_next[points_status]

    # Vykreslení sledovaných bodů
    for point in points_prev:
        x, y = point.ravel()
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    # Hledání zájmové oblasti na snímku
    if crop_img is not None:
        res = cv2.matchTemplate(gray, crop_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= 0.8:
            x, y = max_loc
            points_to_track = [[x + int(bbox[0]), y + int(bbox[1])],
                               [x + int(bbox[0]) + int(bbox[2]), y + int(bbox[1]) + int(bbox[3])]]
            points_prev = np.array(points_to_track, dtype=np.float32)
            crop_img = None

    # Přepočet pixelů na milimetry
    if selected_distance:
        pixel_to_mm_ratio = selected_distance / np.linalg.norm(selected_points[1] - selected_points[0])
        for point in points_prev:
            x, y = point.ravel()
            mm_x = (x - selected_points[0][0]) * pixel_to_mm_ratio
            mm_y = (y - selected_points[0][1]) * pixel_to_mm_ratio
            cv2.putText(image, f'({mm_x:.2f} mm, {mm_y:.2f} mm)', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Zobrazení výsledného obrázku s vyznačenými body
    cv2.imshow('Point Tracker', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Příprava pro další iteraci
    gray_prev = gray.copy()

cv2.destroyAllWindows()
