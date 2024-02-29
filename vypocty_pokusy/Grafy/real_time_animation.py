import os
import cv2
import time
from numba import jit
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
# from matplotlib.widgets import Button
# from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RectangleSelector
import concurrent.futures


def bar_left(_):
    global left
    left = int(cv2.getTrackbarPos("LEFT", "Crop"))


def bar_right(_):
    global right
    right = int(img_width - cv2.getTrackbarPos("RIGHT", "Crop"))


def bar_top(_):
    global top
    top = int(cv2.getTrackbarPos("TOP", "Crop"))


def bar_down(_):
    global down
    down = int(img_height - cv2.getTrackbarPos("DOWN", "Crop"))


def mark_rectangle_on_canvas(image):
    def onselect(_, __):
        pass

    figure, axis = plt.subplots(num="Mark rectangle")
    axis.set_title("Mark rectangle", wrap=True)

    axis.imshow(image)
    selector = RectangleSelector(axis, onselect, useblit=True, button=[1],
                                 minspanx=5, minspany=5, spancoords='pixels', interactive=True,
                                 props=dict(facecolor="yellowgreen", edgecolor="darkgreen", alpha=0.25,
                                            linestyle='dashed', linewidth=1.5))

    axis.set_aspect('equal', adjustable='box')
    figure.tight_layout()
    axis.autoscale(True)
    plt.show()

    rectangle = np.float64(selector.extents).reshape(2, 2).T

    # Procházení sloupců a nastavení horní hodnoty
    for column, limit in enumerate((image.shape[1], image.shape[0])):
        rectangle[:, column] = np.clip(rectangle[:, column], a_min=0, a_max=limit)

    return np.round(rectangle).astype(int)


def subdivide_triangulation(trin):
    # Vytvoření nových bodů (např. středy stran)
    # Přidání nových bodů do seznamu vrcholů
    # Aktualizace seznamu vrcholů
    # Vytvoření nové triangulace
    return Delaunay(np.vstack(
        [trin.points, np.array([[np.average([trin.points[triangle[0]], trin.points[triangle[1]]], axis=0),
                                 np.average([trin.points[triangle[1]], trin.points[triangle[2]]], axis=0),
                                 np.average([trin.points[triangle[2]], trin.points[triangle[0]]], axis=0)]
                                for triangle in trin.simplices]).reshape(-1, 2)]))


def subdivide_roi(x_min, x_max, y_min, y_max, num_points_x, num_points_y):
    x_points = np.linspace(x_min, x_max, num_points_x)
    y_points = np.linspace(y_min, y_max, num_points_y)
    return np.meshgrid(x_points, y_points)


@jit(nopython=True, fastmath=True, cache=True)
def normalize_value(x):
    return ((min(max(x, min_value), max_value) - min_value) / (max_value - min_value)) * (255 - 0) + 0


def process_reference_point(reference_point, p_old, p_new, radius):
    selected_ind = []
    c = 1
    while np.sum(selected_ind) < 6:
        distances = np.linalg.norm(p_old - reference_point, axis=1)
        selected_ind = distances <= radius * c  # Výběr bodů vzdálených o distance
        c += 0.05
    # print("C:", c)
    """if c > 2.5:
        return"""

    tran_mat = cv2.findHomography(p_old[selected_ind], p_new[selected_ind], cv2.RANSAC, 5.0)[0]
    def_roi_single = cv2.perspectiveTransform(np.float32(reference_point).reshape(-1, 1, 2), tran_mat)[0][0]
    return def_roi_single


# ROI
triangulation_type = 'Mesh'  # Mesh // Delaunay
num_subdivisions = 3  # Počet podrozdělení
x_divider = 10
y_divider = 3
direction = 1  # 0 = x, 1 = y

# Zdrojový typ
webcam = 0
source_type = 'webcam'  # pohotos // webcam
folder = r'foo'
alpha = 0.5
window_width = 1000

# Definice hodnot popisků colorbaru
min_value = -50
max_value = 130
num_ticks = 7

# SIFT
n_features = 0  # 0 def = 0
n_octave_layers = 1  # 3 def = 3
contrast_threshold = 0.03  # 0.08 def = 0.04
edge_threshold = 7  # 15 def = 10
sigma = 1.6  # 1.6  def = 1.6
radius = 200

cv2.setUseOptimized(True)  # Zapnutí optimalizace (může využívat akceleraci)
cv2.setNumThreads(cv2.getNumThreads())  # Přepnutí na použití CPU počet jader
print("Počet využitých jader:", cv2.getNumThreads())

average_area = 25
threshold_value = 6
dilate_area = 50
laplacian_kernel = np.float32([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
kernel1 = np.ones((average_area, average_area), dtype=np.float32) / (average_area ** 2)
kernel2 = np.ones((dilate_area, dilate_area), np.uint8)

if source_type == 'webcam':
    camera = cv2.VideoCapture(webcam, cv2.CAP_ANY)  # cv2.CAP_ANY // cv2.CAP_MSMF
    # 4032×3040@10 fps; 3840×2160@20 fps; 2592×1944@30 fps; 2560×1440@30 fps; 1920×1080@60 fps; 1600×1200@50 fps;
    # 1280×960@100 fps; 1280×760@100 fps; 640×480@80 fps
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    camera.set(cv2.CAP_PROP_FPS, 10)

    reference_image = None
    for _ in range(2):
        reference_image = camera.read()[1]
        time.sleep(1)

elif source_type == 'photos':
    # Seznam fotografií
    photos = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith("jpg")]
    photos = sorted(photos, key=lambda filename: int(os.path.splitext(filename)[0].split("_")[-1]))
    photos = [photos[0], photos[11], photos[23], photos[35], photos[-1]]

    # images = [cv2.imread(os.path.join(folder, f)) for f in photos]
    photo = 0
    tot_photos = len(photos) - 1
    reference_image = cv2.imread(os.path.join(folder, photos[0]))
else:
    raise ValueError("Neplatný zdroj dat!")  # Neplatný zdroj dat

img_height, img_width = reference_image.shape[:2]
left, right, top, down = 0, img_width, 0, img_height

cv2.namedWindow('Reference Image', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('Reference Image', window_width, round(img_height / img_width * window_width))
cv2.imshow('Reference Image', reference_image)

cv2.namedWindow("Crop")
cv2.resizeWindow("Crop", 700, 0)
cv2.createTrackbar("LEFT", "Crop", int(left), int(img_width / 2) - 1, bar_left)
cv2.createTrackbar("RIGHT", "Crop", img_width - right, int(img_width / 2) - 1, bar_right)
cv2.createTrackbar("TOP", "Crop", int(top), int(img_height / 2) - 1, bar_top)
cv2.createTrackbar("DOWN", "Crop", img_height - down, int(img_height / 2) - 1, bar_down)
cv2.resizeWindow("Crop", 350, 85)

key = None
while True:
    cv2.imshow('Reference Image', reference_image[top:down, left:right])

    key = cv2.waitKey(1)  # Zpoždění 1 sekundy pro každý obrázek (1000 ms)
    if key == 27:  # Kód pro klávesu ESC
        break

    if cv2.getWindowProperty('Reference Image', cv2.WND_PROP_VISIBLE) < 1:
        cv2.namedWindow('Reference Image', cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('Reference Image', window_width, round(img_height / img_width * window_width))
        cv2.imshow('Reference Image', reference_image[top:down, left:right])

    if cv2.getWindowProperty("Crop", cv2.WND_PROP_VISIBLE) < 1:
        cv2.namedWindow("Crop")
        cv2.resizeWindow("Crop", 700, 0)
        cv2.createTrackbar("LEFT:", "Crop", int(left), int(img_width / 2) - 1, bar_left)
        cv2.createTrackbar("RIGHT:", "Crop", img_width - right, int(img_width / 2) - 1, bar_right)
        cv2.createTrackbar("TOP:", "Crop", int(top), int(img_height / 2) - 1, bar_top)
        cv2.createTrackbar("DOWN:", "Crop", img_height - down, int(img_height / 2) - 1, bar_down)
        cv2.resizeWindow("Crop", 350, 85)
cv2.destroyAllWindows()

reference_image = reference_image[top:down, left:right]
img_height, img_width = reference_image.shape[:2]

bar_width: int = round(max(100, min(img_width * 0.1, 200)))
window_height = round(img_height / (img_width + 4 * bar_width) * window_width)

# Označení oblasti
roi = mark_rectangle_on_canvas(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
# roi = np.array([[176, 256], [4150, 400]])

points = subdivide_roi(roi[0, 0], roi[1, 0], roi[0, 1], roi[1, 1], max(x_divider, 2), max(y_divider, 2))
points = np.vstack([points[0].ravel(), points[1].ravel()]).T

if triangulation_type == 'Mesh':
    # Delaunay triangulace
    tri = Delaunay(points)
elif triangulation_type == 'Delaunay':
    # Podrozdělení triangulace
    for _ in range(max(num_subdivisions, 1)):
        tri = subdivide_triangulation(tri)
else:
    raise ValueError("Neplatný typ triangulace!")

# Vykreslení trojúhelníků
plt.figure()
plt.gca().set_aspect('equal')
plt.triplot(tri.points[:, 0], tri.points[:, 1], tri.simplices)
plt.plot(tri.points[:, 0], tri.points[:, 1], 'o')
plt.tight_layout()
plt.show()

first_cmap_values = [np.mean(tri.points[triangle_indices], axis=0)[direction] for triangle_indices in tri.simplices]

sift = cv2.SIFT_create(
    nfeatures=n_features,  # __________________ Počet detekovaných rysů (0 = všechny dostupné) ______ def = 0
    nOctaveLayers=n_octave_layers,  # _________ Počet vrstev v každé oktávě _________________________ def = 3
    contrastThreshold=contrast_threshold,  # __ Práh kontrastu pro platnost rysu ____________________ def = 0.04
    edgeThreshold=edge_threshold,  # __________ Práh hrany pro platnost rysu blízko k okraji ________ def = 10
    sigma=sigma  # ___________________________ Gaussovská hladina oktáv ____________________________ def = 1.6
)
fast = cv2.FastFeatureDetector_create()
fast.setNonmaxSuppression(False)

# Porovnání popisovačů pomocí algoritmu BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

mask = np.zeros(reference_image.shape[:2], dtype=np.uint8)
mask[roi[0, 1]:roi[1, 1], roi[0, 0]:roi[1, 0]] = 255
keypoints1, descriptors1 = sift.detectAndCompute(reference_image, mask)

cv2.namedWindow('Image with keypoints', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('Image with keypoints', window_width, round(img_height / img_width * window_width))
cv2.imshow('Image with keypoints',
           cv2.drawKeypoints(reference_image, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Detekce hran ostré
detection_image = cv2.filter2D(cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY), cv2.CV_8U, laplacian_kernel)
blurred_image = cv2.filter2D(detection_image, -1, kernel1)
binary_mask = cv2.threshold(blurred_image, threshold_value, 255, cv2.THRESH_BINARY)[1]
# Morfologická operace dilatace pro spojení blízkých hran
dilated_mask = cv2.dilate(binary_mask, kernel2, iterations=3)
# Morfologická operace eroze pro odstranění malých objektů a zúžení hran
mask = cv2.erode(dilated_mask, kernel2, iterations=2)

print("Colorbar making...")
color_bar = np.ones((img_height, 3 * bar_width, 3)) * 255
color_bar[int(img_height * 0.05):int(img_height * 0.95), :bar_width, :] = cv2.resize(
    cv2.applyColorMap(np.arange(256, dtype=np.uint8)[::-1].reshape(1, 256).T, cv2.COLORMAP_JET),
    (1, int(img_height * 0.9)))

font_size = img_height * 0.9 / num_ticks / 65

# Vytvoření popisků
tick_labels = [str(int(value)) for value in np.linspace(max_value, min_value, num_ticks)]

# Rozmístění popisků podle výšky colorbaru
tick_positions = np.linspace(int(img_height * 0.05), int(img_height * 0.95), num_ticks, endpoint=True).astype(int)

text_size, _ = cv2.getTextSize(tick_labels[0], cv2.FONT_HERSHEY_SIMPLEX, font_size, 10)

# Vykreslení popisků
for i, (label, y) in enumerate(zip(tick_labels, tick_positions)):
    cv2.putText(color_bar, label, (int(bar_width * 1.3), y + text_size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                (0, 0, 0), round(font_size * 3), cv2.LINE_AA)
    cv2.line(color_bar, (bar_width, y), (int(bar_width * 1.15), y), (0, 0, 0), int(font_size * 2))

combined_image = np.ones((img_height, img_width + 4 * bar_width, 3), dtype=np.uint8) * 255
combined_image[:, img_width + bar_width:, :] = color_bar

del color_bar, tick_labels, tick_positions, text_size, font_size, points, _, i, label, y, roi

print("Window making...")
cv2.namedWindow('Image with Heatmap', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('Image with Heatmap', window_width, window_height)

# Cyklus pro vykreslení tepelné mapy na každý obrázek
while True:
    ttt = time.time()
    if source_type == 'webcam':
        image = camera.read()[1][down:top, left:right]
    elif source_type == 'photos':
        if photo == tot_photos:
            photo = 0
        else:
            photo += 1
        image = cv2.imread(os.path.join(folder, photos[photo]))[down:top, left:right]

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    tm = time.time()

    detection_image = cv2.filter2D(gray_image, cv2.CV_8U, laplacian_kernel)  # Detekce hran ostré
    blurred_image = cv2.filter2D(detection_image, -1, kernel1)
    binary_mask = cv2.threshold(blurred_image, threshold_value, 255, cv2.THRESH_BINARY)[1]
    # Morfologická operace dilatace pro spojení blízkých hran
    dilated_mask = cv2.dilate(binary_mask, kernel2, iterations=3)
    # Morfologická operace eroze pro odstranění malých objektů a zúžení hran
    mask = cv2.erode(dilated_mask, kernel2, iterations=2)

    keypoints2, descriptors2 = sift.detectAndCompute(gray_image, mask)

    """keypoints2 = fast.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
    keypoints2, descriptors2 = sift.compute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), keypoints2)"""
    print("\nSIFT time:", time.time() - tm)

    tm = time.time()
    p_old, p_new = [], []

    for m in sorted(bf.match(descriptors1, descriptors2), key=lambda x: x.distance):
        p_old.extend([keypoints1[m.queryIdx].pt])
        p_new.extend([keypoints2[m.trainIdx].pt])

    p_old, p_new = np.array(p_old), np.array(p_new)
    # Use concurrent.futures to process reference_points in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = [executor.submit(process_reference_point, reference_point, p_old, p_new, radius)
                   for reference_point in tri.points]

        # Combine results
    def_roi = np.array([result.result() for result in results if result.result() is not None])

    """
    def_roi = np.array([process_reference_point(reference_point, p_old, p_new, radius) for reference_point in tri.points])
    """

    """# def_roi = []
    def_roi = np.empty((0, 2))
    for reference_point in tri.points:
        selected_ind = []
        c = 1
        # Výpočet vzdálenosti mezi každým bodem a referenčním bodem
        while np.sum(selected_ind) < 6:
            distances = np.linalg.norm(p_old - reference_point, axis=1)
            selected_ind = distances <= radius * c  # Výběr bodů vzdálených o distance
            c += 0.05

        tran_mat = cv2.findHomography(p_old[selected_ind], p_new[selected_ind], cv2.RANSAC, 5.0)[0]
        # def_roi.extend([cv2.perspectiveTransform(np.float32(reference_point).reshape(-1, 1, 2), tran_mat)[0][0]])
        def_roi = np.vstack(
            [def_roi, cv2.perspectiveTransform(np.float32(reference_point).reshape(-1, 1, 2), tran_mat)[0][0]])
    # def_roi = np.array(def_roi)"""

    print("Homography time:", time.time() - tm)

    tm = time.time()
    # Vytvoření kopie aktuálního obrázku pro aplikaci tepelné mapy
    cmap = np.zeros_like(reference_image, dtype=np.uint8)
    for j, triangle_indices in enumerate(tri.simplices):
        color = cv2.applyColorMap(
            np.uint8(normalize_value(
                np.mean(def_roi[triangle_indices], axis=0)[direction] - first_cmap_values[j])).reshape(
                (1, 1)),
            cv2.COLORMAP_JET)[0][0].tolist()
        cv2.drawContours(cmap, [def_roi[triangle_indices].astype(int)], -1, color, -1)

    # Přidání colorbaru vedle obrázku s mezerou
    combined_image[:img_height, :img_width] = cv2.addWeighted(image, alpha, cmap, 1 - alpha, 0)
    print("Colorbar time:", time.time() - tm)

    if cv2.getWindowProperty('Image with Heatmap', cv2.WND_PROP_VISIBLE) < 1:
        cv2.namedWindow('Image with Heatmap', cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('Image with Heatmap', window_width, window_height)

    # Zobrazení obrázku s tepelnou mapou
    cv2.imshow('Image with Heatmap', combined_image)
    key = cv2.waitKey(1)  # Zpoždění 1 sekundy pro každý obrázek (1000 ms)

    # Kontrola stisknutí klávesy ESC
    if key == 27:  # Kód pro klávesu ESC
        break

    print("Total time:", time.time() - ttt)

camera.release()
cv2.destroyAllWindows()

if False:
    # Funkce pro změnu proměnné
    def change_variable(_):
        global close_window
        close_window = not close_window
        plt.close()


    # Funkce pro aktualizaci
    def update(frame):
        ax.clear()
        img = ax.imshow(images[frame])
        # mesh = ax.pcolormesh(images[frame])
        """clip_path = plt.Circle((50, 50), 20, edgecolor='black', linewidth=3, facecolor='none', alpha=1, zorder=6)
        ax.add_patch(clip_path)
        mesh.set_clip_path(clip_path)"""


    # Seznam fotografií
    folder = (r'C:\Users\matej\Documents\Škola\.semestry\8. semestr\Bakalářská práce\Experimentální měření\TEST 1\Fotky'
              r'\Rozdělěné upravené\.ČB\1.1a')
    photos = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith("jpg")]
    photos = sorted(photos, key=lambda filename: int(os.path.splitext(filename)[0].split("_")[-1]))

    images = [plt.imread(os.path.join(folder, f)) for f in photos]

    # Inicializace proměnné
    close_window = False

    while True:
        # Inicializace grafu
        fig, ax = plt.subplots()
        ax.axis('equal')
        # mesh = ax.pcolormesh(images[0])
        img = ax.imshow(images[0])
        """clip_path = plt.Circle((50, 50), 20, edgecolor='black', linewidth=3, facecolor='none', alpha=1, zorder=6)
        ax.add_patch(clip_path)
        mesh.set_clip_path(clip_path)"""

        # Vytvoření tlačítka
        button = Button(plt.axes((1 - 0.2 - 0.03, 0.02, 0.2, 0.075)), 'Close')
        button.on_clicked(change_variable)

        # Zobrazení grafu
        # fig.tight_layout()
        fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.17)

        # Animace
        ani = FuncAnimation(fig, update, frames=len(images), interval=200)

        plt.show()

        # Kontrola, zda bylo okno zavřeno
        if not plt.fignum_exists(fig.number):
            if close_window:
                break
            pass

"""
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def grab_frame(cap):
    ret, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# Initiate the two cameras
cap1 = cv2.VideoCapture(0)

# create two subplots
ax = plt.gca()
fig = plt.gcf()

# create two image plots
im1 = ax.imshow(grab_frame(cap1))


def update(i):
    im1.set_data(grab_frame(cap1))


ani = FuncAnimation(fig, update, cache_frame_data=False, interval=200)


def close(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)


cid = fig.canvas.mpl_connect("key_press_event", close)

plt.show()
"""
