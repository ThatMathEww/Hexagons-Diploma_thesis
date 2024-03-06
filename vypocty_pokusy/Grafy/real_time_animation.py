import os
import cv2
import time
from numba import jit
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
# from matplotlib.widgets import Button
# from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RectangleSelector, PolygonSelector
# import concurrent.futures
from pyzbar.pyzbar import decode
import yaml


@jit(nopython=True, fastmath=True, cache=True)
def rotate_points_180_degrees(input_points):
    # Spočítání středu polygonu
    center_point = np.array([np.sum(input_points[:, 0]) / len(input_points),
                             np.sum(input_points[:, 1]) / len(input_points)], dtype=np.float32)

    # Posun všech bodů tak, aby střed byl v počátku souřadnic
    # Otočení bodů o 180 stupňů
    # Posun všech bodů zpět na původní místo
    return np.dot(input_points - center_point, np.array([[-1, 0], [0, -1]], dtype=np.float32)) + center_point


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


def grid_polygon(points_input, num_points_x, num_points_y):
    if len(points_input) != 4:
        return points_input

    points_input = np.float64(points_input)
    # Vypočítání středu bodů
    center = np.mean(points_input, axis=0)

    # Výpočet úhlů bodů od středu
    angles = np.arctan2(points_input[:, 1] - center[1], points_input[:, 0] - center[0])

    # Seřazení bodů podle úhlů
    sorted_indices = np.argsort(angles)

    # Přeuspořádání bodů
    sorted_points = points_input[sorted_indices]

    # Přesunutí prvního bodu na začátek
    first_index = np.where(sorted_indices == 0)[0][0]
    sorted_points = np.roll(sorted_points, -first_index, axis=0).reshape((-1, 2))

    def divide(p_1, p_2, div):
        x_p = np.linspace(p_1[0], p_2[0], div)
        y_p = np.linspace(p_1[1], p_2[1], div)
        return np.vstack([x_p, y_p]).T

    grid_points = np.array([divide(d1, d2, num_points_x) for d1, d2 in
                            zip(divide(sorted_points[0], sorted_points[3], num_points_y),
                                divide(sorted_points[1], sorted_points[2], num_points_y))],
                           dtype=np.float32).reshape(-1, 2)

    """plt.scatter(grid_points[:, 0], grid_points[:, 1])
    plt.show()"""

    return grid_points


def mark_area_on_canvas(source_image, name="Mark rectangle", face_color="yellowgreen", edge_color="darkgreen",
                        selecting_function="rectangle"):
    def onselect(*__):
        pass

    figure, axis = plt.subplots(num=name)
    axis.set_title(name, wrap=True)

    axis.imshow(source_image)
    selector = None
    if selecting_function == "rectangle":
        selector = RectangleSelector(axis, onselect, useblit=True, button=[1],
                                     minspanx=5, minspany=5, spancoords='pixels', interactive=True,
                                     props=dict(facecolor=face_color, edgecolor=edge_color, alpha=0.25,
                                                linestyle='dashed', linewidth=1.5))
    elif selecting_function == "polygon":
        selector = PolygonSelector(axis, onselect,
                                   props=dict(color=edge_color, alpha=0.7, linestyle='dashed', linewidth=1.5),
                                   useblit=True, draw_bounding_box=False,
                                   box_props=dict(facecolor=face_color, edgecolor=edge_color, alpha=0.25,
                                                  linewidth=1))

    axis.set_aspect('equal', adjustable='box')
    figure.tight_layout()
    axis.autoscale(True)
    plt.show()

    area = []
    if selecting_function == "rectangle":
        area = np.float64(selector.extents).reshape(2, 2).T
    elif selecting_function == "polygon":
        area = np.float64(selector.verts)

    # Procházení sloupců a nastavení horní hodnoty
    if len(area) > 0:
        for column, limit in enumerate((source_image.shape[1], source_image.shape[0])):
            area[:, column] = np.clip(area[:, column], a_min=0, a_max=limit)

    return np.round(area).astype(int)


def subdivide_triangulation(in_tri):
    # Vytvoření nových bodů (např. středy stran)
    # Přidání nových bodů do seznamu vrcholů
    # Aktualizace seznamu vrcholů
    # Vytvoření nové triangulace
    return Delaunay(np.vstack(
        [in_tri.points, np.array([[np.average([in_tri.points[triangle[0]], in_tri.points[triangle[1]]], axis=0),
                                   np.average([in_tri.points[triangle[1]], in_tri.points[triangle[2]]], axis=0),
                                   np.average([in_tri.points[triangle[2]], in_tri.points[triangle[0]]], axis=0)]
                                  for triangle in in_tri.simplices], dtype=np.float32).reshape(-1, 2)]))


def subdivide_roi(x_min, x_max, y_min, y_max, num_points_x, num_points_y):
    x_points = np.linspace(x_min, x_max, num_points_x)
    y_points = np.linspace(y_min, y_max, num_points_y)
    return np.meshgrid(x_points, y_points)


@jit(nopython=True, fastmath=True, cache=True)
def normalize_value(x):
    return ((min(max(x, min_value), max_value) - min_value) / (max_value - min_value)) * (255 - 0) + 0


@jit(nopython=True, fastmath=True, cache=True)
def calc_strain(disp_points_, length_):
    x1, y1 = (disp_points_[1] + disp_points_[2]) / 2
    x2, y2 = (disp_points_[0] + disp_points_[3]) / 2
    return 100 * (1 - (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / length_))


# Funkce pro výpočet vzdálenosti mezi bodem a referenčním bodem
def calculate_distances1(reference_point, p_old_, radius_):
    c = 0.95
    selected_ind = np.zeros(p_old_.shape[0], dtype=bool)

    # Výpočet vzdáleností
    while np.sum(selected_ind) < 6:
        c += 0.05
        distances = np.linalg.norm(p_old_ - reference_point, axis=1)
        selected_ind = distances <= radius_ * c

    return selected_ind


@jit(nopython=True, fastmath=True, cache=True)
def calculate_distances2(p_old_, radius_, c_min=0.95, c_step=0.05, min_points=6):
    distances = np.sqrt((p_old_[:, 0] ** 2) + (p_old_[:, 1] ** 2))
    selected_ind = np.zeros(p_old_.shape[0], dtype=np.bool_)

    for _ in range(p_old_.shape[0]):
        c = c_min
        while not np.any(selected_ind):
            c += c_step
            selected_ind = distances <= radius_ * c
            if np.sum(selected_ind) >= min_points:
                break

    return selected_ind


def process_reference_point(reference_point_, p_old_, p_new_, radius_):
    selected_ind = []
    c = 0.95
    while np.sum(selected_ind) < 6:
        c += 0.05
        distances = np.linalg.norm(p_old_ - reference_point_, axis=1)
        selected_ind = distances <= radius_ * c  # Výběr bodů vzdálených o distance
    if c > 1:
        print("C:", c)
    """)if c > 2.5:
        return"""

    tran_mat = cv2.findHomography(p_old_[selected_ind], p_new_[selected_ind], cv2.RANSAC, 5.0)[0]
    def_roi_single = cv2.perspectiveTransform(np.float32(reference_point_).reshape(-1, 1, 2), tran_mat)[0][0]
    return def_roi_single


def main():
    global left, top, right, down

    # Nastavení parametrů
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
        print("Camera connecting...")
        camera = cv2.VideoCapture(webcam, cv2.CAP_MSMF)  # cv2.CAP_ANY // cv2.CAP_MSMF
        # 4032×3040@10 fps; 3840×2160@20 fps; 2592×1944@30 fps; 2560×1440@30 fps; 1920×1080@60 fps; 1600×1200@50 fps;
        # 1280×960@100 fps; 1280×760@100 fps; 640×480@80 fps
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        camera.set(cv2.CAP_PROP_FPS, 100)
        print("Camera connected!")

        print("Dostupná nastavení kamery:")
        properties = [camera.get(prop_id) for prop_id in
                      range(19)]  # range(cv2.CAP_PROP_POS_MSEC, cv2.CAP_PROP_MODE + 1):
        for i, s in enumerate(properties):
            print(f"ID: {i}, Hodnota: {s}")

        """
        0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
        1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
        2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
        3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
        4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
        5. CV_CAP_PROP_FPS Frame rate.
        6. CV_CAP_PROP_FOURCC 4-character code of codec.
        7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
        8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
        9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
        10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
        11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
        12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
        13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
        14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
        15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
        16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
        17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
        18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras 
        (note: only supported by DC1394 v 2.x backend currently)
        """

        if os.path.isfile(calibration_file):
            with open(calibration_file) as file:
                loaded_dict = yaml.safe_load(file)

            mtx_loaded = np.array(loaded_dict.get('camera_matrix'))
            dist_loaded = np.array(loaded_dict.get('dist_coeff'))
            del loaded_dict, yaml, file
        else:
            del yaml

            reference_image = camera.read()[1]
            mtx_loaded = np.array([[1, 0, reference_image.shape[1] / 2],
                                   [0, 1, reference_image.shape[0] / 2],
                                   [0, 0, 1]])
            dist_loaded = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        reference_image = cv2.undistort(camera.read()[1], mtx_loaded, dist_loaded, None, mtx_loaded)


    elif source_type == 'photos':
        # Seznam fotografií
        photos = [f for f in os.listdir(folder) if
                  os.path.isfile(os.path.join(folder, f)) and f.lower().endswith("jpg")]
        photos = sorted(photos, key=lambda filename: int(os.path.splitext(filename)[0].split("_")[-1]))
        photos = [photos[0], photos[11], photos[23], photos[35], photos[-1]]

        # images = [cv2.imread(os.path.join(folder, f)) for f in photos]
        photo = 0
        tot_photos = len(photos) - 1
        reference_image = cv2.imread(os.path.join(folder, photos[0]))
    else:
        raise ValueError("Neplatný zdroj dat!")  # Neplatný zdroj dat

    img_height, img_width = reference_image.shape[:2]
    """left, right, top, down = 0, img_width, 0, img_height
    
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
    cv2.destroyAllWindows()"""

    if source_type == 'webcam':
        cv2.namedWindow('Reference Image', cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('Reference Image', window_width, round(img_height / img_width * window_width))
        cv2.imshow('Reference Image', reference_image)

        while True:
            reference_image = cv2.undistort(camera.read()[1], mtx_loaded, dist_loaded, None, mtx_loaded)
            cv2.imshow('Reference Image', reference_image)

            # [camera.set(i, s) for i, s in enumerate(properties) if s != -1]

            key = cv2.waitKey(1)  # Zpoždění 1 sekundy pro každý obrázek (1000 ms)
            if key == 27:  # Kód pro klávesu ESC
                break

            if cv2.getWindowProperty('Reference Image', cv2.WND_PROP_VISIBLE) < 1:
                cv2.namedWindow('Reference Image', cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow('Reference Image', window_width, round(img_height / img_width * window_width))
                cv2.imshow('Reference Image', reference_image)
        cv2.destroyAllWindows()

    # left, top, right, down = (0, 0, 0, 1)
    left, top, right, down = mark_area_on_canvas(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB), name="Crop photo",
                                                 face_color="pink", edge_color="firebrick").ravel()

    if (left, top, right, down) == (0, 0, 0, 1):
        left, top, right, down = 0, 0, reference_image.shape[1], reference_image.shape[0]

    reference_image = reference_image[top:down, left:right]
    img_height, img_width = reference_image.shape[:2]

    bar_width: int = round(max(100, min(img_width * 0.1, 200)))
    window_height = round(img_height / (img_width + 4 * bar_width) * window_width)

    # Označení oblasti
    # roi = np.array([[170, 250], [4150, 420]])
    roi = mark_area_on_canvas(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB), selecting_function="rectangle")

    tri = None
    if cal_type == 'Displacement' and (triangulation_type == 'TRI_Mesh' or triangulation_type == 'TRI_Delaunay'):
        points = subdivide_roi(roi[0, 0], roi[1, 0], roi[0, 1], roi[1, 1], max(x_divider, 2), max(y_divider, 2))
        points = np.vstack([points[0].ravel(), points[1].ravel()]).T

        # points = grid_polygon(roi, max(x_divider, 2), max(y_divider, 2))

        if triangulation_type == 'TRI_Mesh':
            # Delaunay triangulace
            tri = Delaunay(points)
        elif triangulation_type == 'TRI_Delaunay':
            tri = Delaunay(points)
            # Podrozdělení triangulace
            for _ in range(max(num_subdivisions, 0)):
                tri = subdivide_triangulation(tri)
        else:
            raise ValueError("Neplatný typ triangulace!")

        roi_points = tri.points
        roi_ind = tri.simplices

        # Vykreslení trojúhelníků
        plt.figure()
        plt.gca().set_aspect('equal')
        plt.gca().invert_yaxis()
        plt.triplot(roi_points[:, 0], roi_points[:, 1], roi_ind)
        plt.plot(roi_points[:, 0], roi_points[:, 1], 'o')
        plt.tight_layout()
        plt.show()

        first_cmap_values = [np.mean(roi_points[indices], axis=0)[direction] for indices in roi_ind]

        del points, tri

    elif cal_type == 'Strain' or triangulation_type == 'Mesh':
        max_x = max(x_divider, 4)
        max_y = max(y_divider, 4)

        roi_p = subdivide_roi(roi[0, 0], roi[1, 0], roi[0, 1], roi[1, 1], max_x, max_y)
        roi_p = np.vstack([roi_p[0].ravel(), roi_p[1].ravel()]).T

        # roi_pp = grid_polygon(roi, max_x, max_y)

        roi_points = None
        if direction == 0:
            original_length = roi_p[1, 0] - roi_p[0, 0]
            p1, p2, p3, p4 = 1, 2, 0, 3

            roi_points = subdivide_roi(roi[0, 0], roi[1, 0],
                                       np.average((roi_p[0, 1], roi_p[max_x, 1])),
                                       np.average((roi_p[-max_x - 1, 1], roi_p[-1, 1])),
                                       max_x, max_y - 1)
            roi_points = np.vstack(
                [np.concatenate([roi_p[:max_x, 0], roi_points[0].ravel(), roi_p[-max_x:, 0]]),
                 np.concatenate([roi_p[:max_x, 1], roi_points[1].ravel(), roi_p[-max_x:, 1]])]).T
        elif direction == 1:
            original_length = roi_p[1, 1] - roi_p[0, 1]
            p1, p2, p3, p4 = 0, 1, 2, 3

            roi_points = subdivide_roi(np.average((roi_p[0, 0], roi_p[1, 0])),
                                       np.average((roi_p[-2, 0], roi_p[-1, 0])),
                                       np.average((roi_p[0, 1], roi_p[1, 1])),
                                       np.average((roi_p[-2, 1], roi_p[-1, 1])),
                                       max_x - 1, max_y)
            roi_points = np.vstack([np.concatenate(
                [roi_p[np.arange(0, len(roi_p), max_x, dtype=int), 0], roi_points[0].ravel(),
                 roi_p[np.arange(max_x - 1, len(roi_p), max_x, dtype=int), 0]]),
                np.concatenate(
                    [roi_p[np.arange(0, len(roi_p), max_x, dtype=int), 1], roi_points[1].ravel(),
                     roi_p[np.arange(max_x - 1, len(roi_p), max_x, dtype=int), 1]])]).T
            roi_points = roi_points[np.lexsort((roi_points[:, 0], roi_points[:, 1]))]

        roi_ind = []
        [[roi_ind.append([i + j, i + 1 + j, i + 1 + direction + max_x + j, i + direction + max_x + j]) for i in
          range(max_x - (1 if direction == 0 else 0))] for j in
         range(0, (max_x * (max_y - direction)), max_x + direction)]

        # Vykreslení trojúhelníků
        plt.figure()
        plt.gca().set_aspect('equal')
        plt.gca().invert_yaxis()
        plt.plot(roi_p[:, 0], roi_p[:, 1], 'o')
        plt.plot(roi_points[:, 0], roi_points[:, 1], 'o')
        """for _ in range(len(disp_points)):
            plt.text(disp_points[_, 0], disp_points[_, 1], f"{_}", fontsize=8)"""
        # Vykreslení čtverců
        for i in roi_ind:
            plt.gca().add_patch(
                plt.Polygon(roi_points[i], closed=True, facecolor=np.random.rand(1, 3), edgecolor='black', alpha=0.3))
        plt.tight_layout()
        plt.show()

        del roi_p

        if cal_type != 'Strain':
            first_cmap_values = [np.mean(roi_points[indices], axis=0)[direction] for indices in roi_ind]

    else:
        raise ValueError("Neplatný typ výpočtu!")

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

    print("Colorbar making...")
    font_size = img_height * 0.9 / num_ticks / 65

    # Vytvoření popisků
    condition = float(max_value).is_integer() and float(min_value).is_integer()
    tick_labels = [str(int(value) if condition else round(value, 3))
                   for value in np.linspace(max_value, min_value, num_ticks)]
    tick_labels[0] = f"{tick_labels[0]}   {'[%]' if cal_type == 'Strain' else '[mm]'}"

    # Rozmístění popisků podle výšky colorbaru
    bar_start, bar_end = int(img_height * 0.05), int(img_height * 0.95)
    tick_positions = np.linspace(int(img_height * 0.05), int(img_height * 0.95), num_ticks, endpoint=True).astype(int)

    text_size = np.max([cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, font_size, 10)[0] for lab in tick_labels],
                       axis=0)

    if text_size[0] >= bar_width * 1.5:
        n = 2 + np.ceil(text_size[0] / bar_width)
    else:
        n = 4

    color_bar = np.ones((img_height, int((n - 1) * bar_width), 3)) * 255
    color_bar[int(img_height * 0.05):int(img_height * 0.95), :bar_width, :] = cv2.resize(
        cv2.applyColorMap(np.arange(256, dtype=np.uint8)[::-1].reshape(1, 256).T, cv2.COLORMAP_JET),
        (1, bar_end - bar_start))

    # Vykreslení popisků
    for i, (label, y) in enumerate(zip(tick_labels, tick_positions)):
        cv2.putText(color_bar, label, (int(bar_width * 1.3), y + text_size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (0, 0, 0), round(font_size * 3), cv2.LINE_AA)
        cv2.line(color_bar, (bar_width, y), (int(bar_width * 1.15), y), (0, 0, 0), int(font_size * 2))

    combined_image = np.ones((img_height, int(img_width + n * bar_width), 3), dtype=np.uint8) * 255
    combined_image[:, img_width + bar_width:, :] = color_bar

    print("QR detection making...")
    s = int(round(max(50, min(img_width * 0.025, 250))))
    arrow_cor = []
    arrow_colors = [(30, 144, 255)[::-1], (255, 165, 0)[::-1]]
    arrow_points = np.array(
        [[-s, s], [0, 0], [s, s], [0.5 * s, s], [0.5 * s, s], [-0.5 * s, s], [-0.5 * s, s]], dtype=int)

    # TODO: Šipky
    # Přidání nových dat do souboru
    # open("stock.txt", "w").close()  # Vymazání obsahu souboru
    frame = 0

    scale = 1
    """qr_decoder = cv2.QRCodeDetector()
    qr_decoder.setEpsX(0.2)  # Tolerance na nepřesnost v horizontálním směru
    qr_decoder.setEpsY(0.2)  # Tolerance na nepřesnost ve vertikálním směru
    
    # Najděte QR kódy v obraze
    success, decoded_info, points, _ = qr_decoder.detectAndDecodeMulti(cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY))"""

    decoded_object = decode(cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY))
    # Pokud byl QR kód nalezen
    if decoded_object:
        decoded_info = [obj.data.decode('utf-8') for obj in decoded_object]
        points = [np.array([point for point in obj.polygon], dtype=np.int32) for obj in decoded_object]
        points, decoded_info = zip(*sorted(zip(points, decoded_info), key=lambda x: int(x[1].split('#')[1])))
        qr_size = []
        for i in range(len(decoded_info)):
            print(f"\tQR codes {i + 1}: {decoded_info[i]}")
            if ".*SP*." in decoded_info[i]:
                qr_size.append(np.mean(np.abs(np.diff(np.mean(np.vstack([points[i], points[i][0]]), axis=1)))))
                arrow_cor.append(arrow_points.copy() + np.mean(points[i], axis=0, dtype=int))
                # Nakreslete polygon kolem QR kódu
                cv2.polylines(reference_image, [points[i].astype(int)], isClosed=True, color=(0, 255, 0), thickness=5)

        scale = scale_qr_ratio / np.mean(qr_size)
        del qr_size, scale_qr_ratio, decoded_info, points
        # Zobrazte obrázek s označenými QR kódy
        cv2.namedWindow('QR codes', cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('QR codes', window_width, round(img_height / img_width * window_width))
        cv2.imshow('QR codes', reference_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\tSet number of QR codes")
        while True:
            n = input("\t\tNumber of QR codes:  ")
            if n.isdigit():
                n = int(n)
                break
            else:
                print("\tInvalid input")
        for s in range(n):
            points = mark_area_on_canvas(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB), selecting_function="polygon",
                                         face_color="dodgerblue", edge_color="navy",
                                         name=f"Mark QR codes [{s + 1} / {n}]")
            arrow_cor.append(arrow_points.copy() + np.mean(points, axis=0, dtype=int))

    del color_bar, tick_labels, tick_positions, text_size, font_size, label, y, roi, n, s, arrow_points
    del bar_width, condition, bar_start, bar_end, decoded_object
    del contrast_threshold, edge_threshold, n_features, n_octave_layers, sigma, fast, reference_image
    del mark_area_on_canvas, subdivide_triangulation, subdivide_roi, calc_strain
    del PolygonSelector, RectangleSelector, Delaunay
    if cal_type == 'Strain':
        del scale
    del plt

    print("Window making...")
    cv2.namedWindow('Image with Heatmap', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('Image with Heatmap', window_width, window_height)

    # Cyklus pro vykreslení tepelné mapy na každý obrázek
    image = None
    err = 0
    while True:
        try:
            ttt = time.time()
            if source_type == 'webcam':
                image = cv2.undistort(camera.read()[1], mtx_loaded, dist_loaded, None, mtx_loaded)[top:down, left:right]
            elif source_type == 'photos':
                if photo == tot_photos:
                    photo = 0
                else:
                    photo += 1
                image = cv2.imread(os.path.join(folder, photos[photo]))[top:down, left:right]

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

            p_old, p_new = np.array(p_old, dtype=np.float32), np.array(p_new, dtype=np.float32)
            # Use concurrent.futures to process reference_points in parallel
            """with concurrent.futures.ThreadPoolExecutor() as executor:
                results = [executor.submit(process_reference_point, reference_point, p_old, p_new, radius)
                           for reference_point in roi_points]
        
                # Combine results
            def_roi = np.array([result.result() for result in results if result.result() is not None], dtype=np.float32)"""

            """def_roi = np.array([process_reference_point(reference_point, p_old, p_new, radius) 
            for reference_point in roi_points], dtype=np.float32)"""

            """# def_roi = []
            def_roi = np.empty((0, 2))
            for reference_point in roi_points:
                selected_ind = []
                c = 0.95
                # Výpočet vzdálenosti mezi každým bodem a referenčním bodem
                while np.sum(selected_ind) < 6:
                    c += 0.05
                    distances = np.linalg.norm(p_old - reference_point, axis=1)
                    selected_ind = distances <= radius * c  # Výběr bodů vzdálených o distance
        
                tran_mat = cv2.findHomography(p_old[selected_ind], p_new[selected_ind], cv2.RANSAC, 5.0)[0]
                # def_roi.extend([cv2.perspectiveTransform(np.float32(reference_point).reshape(-1, 1, 2), tran_mat)[0][0]])
                def_roi = np.vstack(
                    [def_roi, cv2.perspectiveTransform(np.float32(reference_point).reshape(-1, 1, 2), tran_mat)[0][0]])
            # def_roi = np.array(def_roi, dtype=np.float32)"""

            # def_roi = np.empty((0, 2), dtype=np.float32)

            """for reference_point in roi_points:
                selected_ind = calculate_distances2(p_old - reference_point, radius)
                tran_mat = cv2.findHomography(p_old[selected_ind], p_new[selected_ind], cv2.RANSAC, 5.0)[0]
                transformed_point = cv2.perspectiveTransform(np.float32(reference_point).reshape(-1, 1, 2), tran_mat)[0][0]
                def_roi = np.vstack([def_roi, transformed_point])"""

            # Vektorizovaná verze funkce pro výpočet vzdáleností
            vectorized_distances = np.vectorize(calculate_distances1, signature='(n),(m,n),()->(m)')

            # Výpočet vzdáleností pro všechny referenční body
            selected_indices = vectorized_distances(roi_points, p_old, radius)

            def_roi = np.array([cv2.perspectiveTransform(
                np.float32(reference_point).reshape(-1, 1, 2),
                cv2.findHomography(p_old[selected_ind], p_new[selected_ind], cv2.RANSAC, 5.0)[0])[0][0]
                                for reference_point, selected_ind in zip(roi_points, selected_indices)],
                               dtype=np.float32)
            # def_roi = np.array(def_roi, dtype=np.float32)

            """# Použití výsledků pro další výpočty nebo operace
            for reference_point, selected_ind in zip(roi_points, selected_indices):
                # tran_mat = cv2.findHomography(p_old[selected_ind], p_new[selected_ind], cv2.RANSAC, 5.0)[0]
                # transformed_point = cv2.perspectiveTransform(np.float32(reference_point).reshape(-1, 1, 2), tran_mat)[0][0]
                # def_roi = np.vstack([def_roi, transformed_point])
                def_roi = np.vstack([def_roi,
                                     cv2.perspectiveTransform(np.float32(reference_point).reshape(-1, 1, 2),
                                                              cv2.findHomography(p_old[selected_ind], p_new[selected_ind],
                                                                                 cv2.RANSAC, 5.0)[0])[0][0]])"""
            # def_roi[i] = transformed_point
            print("Homography time:", time.time() - tm)

            tm = time.time()
            # Vytvoření kopie aktuálního obrázku pro aplikaci tepelné mapy
            cmap = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            if cal_type == 'Displacement':
                for i, indices in enumerate(roi_ind):
                    color = cv2.applyColorMap(
                        np.uint8(normalize_value(
                            np.mean(def_roi[indices], axis=0)[direction] - first_cmap_values[i]) * scale).reshape(
                            (1, 1)),
                        cv2.COLORMAP_JET)[0][0].tolist()
                    cv2.drawContours(cmap, [def_roi[indices].astype(int)], -1, color, -1)
            elif cal_type == 'Strain':
                disp_diff = (1 - np.array([np.linalg.norm(np.mean((def_roi[i[p1]], def_roi[i[p2]]), axis=0) -
                                                          np.mean((def_roi[i[p3]], def_roi[i[p4]]), axis=0)) for i in
                                           roi_ind], dtype=np.float32) / original_length) * 100

                # disp_diff = [calc_strain(def_roi[i], original_length) for i in roi_ind]

                """# with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = [executor.submit(process_reference_point, reference_point, p_old, p_new)
                               for reference_point in roi_p]
        
                    # Combine results
                disp_points = np.array(
                    [result.result() for result in results if result.result() is not None], 
                    dtype=np.float32).reshape(max_y, max_x, 2)
        
                disp_diff = (1 - np.array([[np.linalg.norm(disp_points[_, i + 1] - disp_points[_, i]) for i in range(max_x - 1)]
                                           for _ in range(max_y)], dtype=np.float32).ravel() / original_length) * 100"""
                # disp_diff = (1 - (np.diff(disp_points[:,:,0], axis=1) / original_length).ravel()) * 100

                """# Vytvoření grafu
                fig, ax = plt.subplots()
        
                # Vykreslení dat po řádcích
                disp = np.array(
                    [result.result() for result in results if result.result() is not None]).reshape(max_y, max_x, 2)
        
                scalar_map = plt.cm.ScalarMappable(cmap=str('jet'))
                scalar_map.set_clim(vmin=min_value, vmax=max_value)
                cbar = plt.colorbar(scalar_map, ax=ax)
        
                n = 0
                m = 0
                l = 0
                dd = []
                for i in range(disp.shape[0]):
                    for j in range(disp.shape[1] - 1):
                        ax.plot(disp[i, j, 0], disp[i, j, 1], 'o')
                        dd.append(disp[i, j + 1, 0] - disp[i, j, 0])
                        # ax.text(disp[i, j, 0], disp[i, j, 1], f"{n}", fontsize=8)
                        n += 1
                n = 0
                for i in roi_ind:
                    # b = np.mean((def_roi[i[1]], def_roi[i[2]]), axis=0) - np.mean((def_roi[i[0]], def_roi[i[-1]]), axis=0)
                    d = 1 - (dd[n] / original_length)
                    plt.gca().add_patch(
                        plt.Polygon(def_roi[i], closed=True, facecolor=scalar_map.to_rgba(d), edgecolor='black',
                                    alpha=0.3))
                    ax.text(*np.mean(def_roi[i], axis=0), f"{n}", fontsize=8)
                    n += 1
        
                    m, l = m + 1, l + 1
                    if l == max_x - 1:
                        l = 0
                    if m == max_y - 1:
                        m = 0
        
                # Zobrazení grafu
                ax.invert_yaxis()
                plt.show()"""

                for i, indices in enumerate(roi_ind):
                    """d = (1 - ((np.mean((def_roi[indices[1]], def_roi[indices[2]]), axis=0) -
                               np.mean((def_roi[indices[0]], def_roi[indices[-1]]), axis=0))[0] / original_length)) * 100"""

                    """d = (1 - (np.linalg.norm(np.mean((def_roi[indices[1]], def_roi[indices[2]]), axis=0) -
                               np.mean((def_roi[indices[0]], def_roi[indices[-1]]), axis=0)) / original_length)) * 100
        
                    color = cv2.applyColorMap(
                        np.uint8(normalize_value(d)).reshape((1, 1)),
                        cv2.COLORMAP_JET)[0][0].tolist()"""

                    color = cv2.applyColorMap(
                        np.uint8(normalize_value(disp_diff[i])).reshape((1, 1)),
                        cv2.COLORMAP_JET)[0][0].tolist()
                    cv2.drawContours(cmap, [def_roi[indices].astype(int)], -1, color, -1)

            # Přidání colorbaru vedle obrázku s mezerou
            combined_image[:img_height, :img_width] = cv2.addWeighted(image, alpha, cmap, 1 - alpha, 0)
            print("Colorbar time:", time.time() - tm)

            err = 0

        except (cv2.error, Exception) as e:
            """err += 1
            if err > 20:
                raise e"""
            if source_type == 'webcam':
                image = cv2.undistort(camera.read()[1], mtx_loaded, dist_loaded, None, mtx_loaded)[top:down, left:right]
            elif source_type == 'photos':
                if photo == tot_photos:
                    photo = 0
                else:
                    photo += 1
                image = cv2.imread(os.path.join(folder, photos[photo]))[top:down, left:right]

            combined_image[:img_height, :img_width] = cv2.addWeighted(
                image, alpha, np.zeros((img_height, img_width, 3), dtype=np.uint8), 1 - alpha, 0)
            pass

        try:
            tm = time.time()
            if load_forces:
                # Načtení nových dat ze souboru
                with open(forces_file, "r") as file:
                    forces = np.float64(file.readlines()[-1].strip().split(","))
                for i, arrow in enumerate(arrow_cor, start=0):
                    arrow = arrow.copy()
                    arrow[4, 1] = arrow[5, 1] = arrow[4, 1] + abs(forces[i + 1]) * 1
                    if 0 > forces[i + 1]:
                        # arrow = rotate_points_180_degrees(arrow).astype(int)
                        # Spočítání středu polygonu
                        center = np.mean(arrow, axis=0)
                        arrow = arrow - center
                        # Posun všech bodů zpět na původní místo
                        arrow = np.int32(np.dot(arrow, np.array([[-1, 0], [0, -1]])) + center)
                    cv2.drawContours(combined_image, [arrow], -1, arrow_colors[i], -1)
            print("Forces time:", time.time() - tm)
        except (IndexError, Exception):
            pass

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

    if source_type == 'webcam':
        camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    global left, top, right, down

    # Calculation
    cal_type = "Strain"  # Displacement // Strain

    # ROI
    triangulation_type = 'Mesh'  # Mesh // TRI_Mesh // TRI_Delaunay
    num_subdivisions = 3  # Počet podrozdělení
    x_divider = 25  # 15
    y_divider = 7  # 5
    direction = 0  # 0 = x, 1 = y

    # Zdrojový typ
    webcam = 0
    cam_width = 1920
    cam_height = 1080
    source_type = 'webcam'  # photos // webcam
    folder = r'foo'
    load_forces = True  # True // False
    forces_file = 'stock.txt'
    calibration_file = fr'calibration_DIC.yaml'  # calibration_{cam_width}x{cam_height}.yaml
    alpha = 0.5
    window_width = 1000
    scale_qr_ratio = 15 * 210 / 270  # mm * px-x / px-y

    # Definice hodnot popisků colorbaru
    min_value = -0.7 * 4  # -20 //  -0.007 * 100
    max_value = 0.7 * 4  # 130 // 0.007 * 100
    num_ticks = 7

    # SIFT
    n_features = 0  # 0 def = 0
    n_octave_layers = 1  # 3 def = 3
    contrast_threshold = 0.02  # 0.08 def = 0.04
    edge_threshold = 6  # 7, 15 def = 10
    sigma = 1.15  # 1.4  def = 1.6
    radius = 100  # 200

    main()
