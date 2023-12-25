import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode

scale_qr_ratio = 12 * 210 / 270

show_images = False
show_match = False
show_graph = False

use_averaging = False
window_size = 3  # Velikost klouzavého průměru

tolerance = 0.25

image_folder = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\Friction_photos'

folders = [f for f in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, f)) and
           not f.startswith(("_", "."))]

accelerations = []
tot_len = len(folders)
for i, folder in enumerate(folders):
    print(os.path.basename(folder), f'( {i + 1} / {tot_len} )')

    folder = os.path.join(image_folder, folder)

    loaded_settings = json.load(open(os.path.join(folder, "settings.json"), 'r'))

    try:
        if loaded_settings["moment"] != 'Waiting':
            print("\t\033[31;1mChyba: Sklouznutí při pohybu.\033[0m")
            continue
    except KeyError:
        if len(os.listdir(folder)) < 30:
            print("\t\033[31;1mChyba: Sklouznutí při pohybu.\033[0m")
            continue

    time = 1 / loaded_settings["fps"]

    images = [img for img in os.listdir(folder) if img.endswith((".jpg", ".jpeg", ".png", ".JPG")) and
              not img.startswith("_first_photo")]
    images = sorted(images, key=lambda filename: int(os.path.splitext(filename)[0].split('_')[-1]))

    img = cv2.imread(os.path.join(folder, images[0]), 1)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """plt.figure()
    plt.imshow(img_gray, cmap='gray')
    plt.show()"""

    # images = images[215:240]

    scale = 1
    found_points = []
    decoded_object = decode(img_gray)
    if not decoded_object:
        print("\t\033[31;1mChyba: Nebyl nalezen QR kód.\033[0m")
        continue
    elif len(decoded_object) > 1:
        print("\t\033[33;1mPozor: Bylo detekováné více QR kódů.\033[0m")
        continue
    else:
        temp_x, temp_y, temp_w, temp_h = 4 * [None]
        for obj in decoded_object:
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

            points = np.array([point for point in obj.polygon], dtype=np.int32)
            qr_size = np.mean(np.abs(np.diff(np.mean(np.vstack([points, points[0]]), axis=1))))

            scale = scale_qr_ratio / qr_size
            # B = scale_qr_ratio / np.linalg.norm(np.mean((149, 208)) - np.mean((247, 208)))
            # c = np.linalg.norm(np.mean((149, 208)) - np.mean((247, 208)))

            # scale = 50 / np.linalg.norm(np.mean((283, 9)) - np.mean((281, 533)))

            if show_images:
                cv2.polylines(img_gray, [cv2.convexHull(points)], True, (0, 255, 0), 2)
                cv2.namedWindow("QR Codes", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("QR Codes", img_gray.shape[1] // 2, img_gray.shape[0] // 2)
                cv2.imshow("QR Codes", img_gray)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            """plt.figure()
            plt.imshow(img_gray, cmap='gray')
            plt.show()"""

            temp_x, temp_y, temp_w, temp_h = cv2.boundingRect(np.array([point for point in obj.polygon],
                                                                       dtype=np.int32))
            temp_x, temp_y = np.int32(temp_x + temp_w * 0.05), np.int32(temp_y + temp_h * 0.05)
            temp_w, temp_h = np.int32(temp_x + temp_w * 0.95), np.int32(temp_y + temp_h * 0.95)

        template_gray = cv2.cvtColor(img_gray[temp_y:temp_h, temp_x:temp_w].copy(), cv2.COLOR_BGR2GRAY)
        if show_images:
            cv2.namedWindow("QR Codes", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("QR Codes", template_gray.shape[1] // 2, template_gray.shape[0] // 2)
            cv2.imshow("QR Codes", template_gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    previous_position = (-10000, -10000)
    for image in images:
        img = cv2.imread(os.path.join(folder, image))  # [100:, 50:-50]

        """# Zobrazení výsledného obrázku
        cv2.imshow('template', template)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""

        # Převedení na odstíny šedi
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Hledání template pomocí cv2.matchTemplate
        res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

        # Nalezení polohy s maximálním korelačním koeficientem
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > tolerance:
            # Získání rozměrů template
            w, h = template_gray.shape[::-1]
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

            position = (top_left[0] + w / 2, top_left[1])

            if np.linalg.norm(np.array(position) - np.array(previous_position)) != 0:
                found_points.append(position)
                previous_position = position
        else:
            continue

        if show_match:
            h, w = img.shape[:2]
            # Zobrazení výsledného obrázku
            cv2.namedWindow(f'Found template {max_val:.04f}', cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f'Found template {max_val:.04f}', int(w * 0.7), int(h * 0.7))
            cv2.imshow(f'Found template {max_val:.04f}', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if len(found_points) < 5:
        print("\t\033[31;1mChyba: Není dostatek bodů.\033[0m")
        continue

    found_points = np.array(found_points) * scale
    found_points = np.vstack([found_points[0], found_points])[:10]
    found_points -= found_points[0]
    found_points = found_points[found_points[:, 1] < 45]
    found_points /= 1000  # metres

    time_stamps = np.arange(0, len(found_points) * time, time)

    speed = np.diff(np.round(np.mean(found_points, axis=1), 6), axis=0) / np.diff(time_stamps)
    speed = np.hstack([speed[0], speed])

    av_speed = np.mean(speed[speed > 0])

    if use_averaging:  # Použití klouzavého průměru
        kernel = np.ones(window_size) / window_size
        speed_av = speed.copy()
        for _ in range(window_size - 1):
            speed_av = np.hstack([speed_av, speed_av[-1]])
        speed_av = np.convolve(speed_av, kernel, mode='valid')
    else:
        speed_av = speed.copy()

    """# Lineární regrese
    y = time_stamps[:len(speed_av)]
    fit = np.polyfit(speed_av, y, 1)
    regression = np.polyval(fit, speed_av)
    residuum = time_stamps - regression  # Rezidua
    limit = 0.01  # Práh pro určení, které body považujeme za lineární
    linear_points_indexes = np.where(np.abs(residuum) < limit)[0]  # indexů bodů, které jsou blízko lineární regrese
    
    fit = np.polyfit(speed_av[linear_points_indexes], y[linear_points_indexes], 1)
    regression = np.polyval(fit, speed_av[linear_points_indexes])
    residuum = time_stamps[linear_points_indexes] - regression  # Rezidua
    limit = 0.01  # Práh pro určení, které body považujeme za lineární
    
    plt.figure()
    plt.scatter(y, speed_av, label='Data')
    plt.plot(regression, speed_av[linear_points_indexes], '-', label='Lineární regrese', color='red')
    
    linear_points_indexes = linear_points_indexes[np.where(np.abs(residuum) < limit)[0]]
    
    plt.scatter(y[linear_points_indexes], speed_av[linear_points_indexes], label='Lineární body', color='orange')
    plt.legend()
    plt.show()
    
    print("Indexy lineárních bodů:", linear_points_indexes)
    """

    lin_ind = np.arange(round(len(speed_av) * 0.2), round(len(speed_av) * 0.85), 1)

    acceleration = np.diff(speed_av, axis=0) / np.diff(time_stamps[:len(speed_av)])
    acceleration = np.hstack([acceleration[0], acceleration])

    av_acc = np.mean(acceleration[acceleration > 0])
    # print(f"\tAverage acceleration: {av_acc:.4f} m/s^2")

    mean_acc = (speed[-1] - speed[0]) / (time_stamps[-1] - time_stamps[0])
    # print(f"\tMean acceleration: {mean_acc:.4f} m/s^2")

    lin_acc = np.mean(acceleration[lin_ind])
    if 'F01' in os.path.basename(folder):
        accelerations.append(lin_acc)
    print(f"\tLinear mean acceleration: {lin_acc:.4f} m/s^2")

    if show_graph:
        plt.figure()

        ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
        ax3 = plt.subplot2grid((2, 3), (1, 1), colspan=2)

        ax1.set_title("Positions")
        ax1.plot(found_points[:, 0], found_points[:, 1], 'o-')
        ax1.axis('equal')
        ax1.invert_yaxis()
        ax1.set_xlabel('x [$m$]')
        ax1.set_ylabel('y [$m$]')

        ax2.set_title("Velocity")
        ax2.hlines(av_speed, time_stamps[0], time_stamps[-1], color='tomato', linestyle='--',
                   label=f"Average speed: {av_speed:.2f} mm/s")
        ax2.plot(time_stamps[:len(speed_av)], speed_av, 'o-', c='orange')
        # ax2.plot(regression, speed_av[linear_points_indexes], '-', color='darkred')
        ax2.set_xlabel('time [$s$]')
        ax2.set_ylabel('v [$m/s$]')

        ax3.set_title("Acceleration")
        ax3.hlines(av_acc, time_stamps[0], time_stamps[-1], color='darkorange', linestyle='--',
                   label=f"Average acceleration: {av_acc:.2f} mm/s")
        ax3.plot(time_stamps[:len(acceleration)], acceleration, 'o-', color='red')
        ax3.hlines(lin_acc, time_stamps[lin_ind[0]], time_stamps[lin_ind[-1]],
                   linestyle='--', color='darkred')
        ax3.set_xlabel('time [$s$]')
        ax3.set_ylabel('a [$m/s^2$]')

        # plt.tight_layout()
        plt.subplots_adjust(right=0.92, left=0.12, top=0.92, bottom=0.12, wspace=0.6, hspace=0.6)
        plt.show()

print(f"\nAverage acceleration: {np.mean(accelerations):.4f} m/s^2")
print(f"\nSTD acceleration: {np.std(accelerations):.4f} m/s^2")
print(f"\nMedian acceleration: {np.median(accelerations):.4f} m/s^2")
