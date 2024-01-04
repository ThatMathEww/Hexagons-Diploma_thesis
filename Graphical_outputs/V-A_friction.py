import os
import cv2
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode

scale_qr_ratio = 12 * 210 / 270

show_images = False
show_match = False
show_graph = True
plot_real_data = False

use_averaging = False
window_size = 3  # Velikost klouzavého průměru

tolerance = 0.24

kinetic_type = True

distance_limit = {'static': 35, 'kinetic': 10000}  # mm

image_folder = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\Friction_photos'

folders = [f for f in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, f)) and
           not f.startswith(("_", "."))]

if kinetic_type:
    folders = [f for f in folders if "_K" in f]
else:
    folders = [f for f in folders if "_K" not in f]

folders = folders[8:]  # [26:]  # [:21]  # [8:9]

accelerations_print = []
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

    with open(r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\Hexagons-Diploma_thesis'
              fr'\Lens_correction\calibration_{loaded_settings["width"]}x{loaded_settings["height"]}.yaml') as f:
        loaded_dict = yaml.safe_load(f)

    mtx_loaded = np.array(loaded_dict.get('camera_matrix'))
    dist_loaded = np.array(loaded_dict.get('dist_coeff'))

    time = 1 / loaded_settings["fps"]

    images = [img for img in os.listdir(folder) if img.endswith((".jpg", ".jpeg", ".png", ".JPG")) and
              not img.startswith("_first_photo")]
    images = sorted(images, key=lambda filename: int(os.path.splitext(filename)[0].split('_')[-1]))  # [579:]

    img_original = cv2.imread(os.path.join(folder, images[0]), 1)

    img = np.zeros((loaded_settings["height"], loaded_settings["width"], 3), dtype=np.uint8)

    h_small, w_small = img_original.shape[:2]

    start_x = (loaded_settings["width"] - w_small) // 2
    start_y = (loaded_settings["height"] - h_small) // 2

    img[start_y:start_y + h_small, start_x:start_x + w_small] = img_original

    img = cv2.undistort(img, mtx_loaded, dist_loaded, None, mtx_loaded)

    img = img[:, start_x:start_x + w_small]

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """plt.figure()
    plt.imshow(img_gray, cmap='gray')
    plt.show()"""

    # images = images[215:240]

    scale = 1
    found_points = [(0, 0)]
    image_time = [0]

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
                cv2.polylines(img_gray, [cv2.convexHull(points)], True, (250, 50, 50), 5)
                # cv2.imwrite('QR_code_gray_undist.jpg', img_gray)
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

    img_original = cv2.imread(os.path.join(folder, images[0]))
    img = np.zeros((loaded_settings["height"], loaded_settings["width"], 3), dtype=np.uint8)
    img[start_y:start_y + h_small, start_x:start_x + w_small] = img_original
    img = cv2.undistort(img, mtx_loaded, dist_loaded, None, mtx_loaded)
    img_gray = cv2.cvtColor(img[:, start_x:start_x + w_small], cv2.COLOR_BGR2GRAY)
    _, max_val, _, max_loc = cv2.minMaxLoc(cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED))
    if max_val > tolerance:
        previous_position = (max_loc[0] + template_gray.shape[1] / 2, max_loc[1])
    else:
        previous_position = (-1, -1)

    ending_counter = 0

    for image in images:
        if ending_counter > 10:
            break

        img_original = cv2.imread(os.path.join(folder, image))  # [100:, 50:-50]

        img = np.zeros((loaded_settings["height"], loaded_settings["width"], 3), dtype=np.uint8)

        img[start_y:start_y + h_small, start_x:start_x + w_small] = img_original

        img = cv2.undistort(img, mtx_loaded, dist_loaded, None, mtx_loaded)

        img = img[:, start_x:start_x + w_small]

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
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > tolerance:
            ending_counter = 0
            # Získání rozměrů template
            w, h = template_gray.shape[::-1]
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            position = (top_left[0] + w / 2, top_left[1])

            if np.linalg.norm(np.array(position) - np.array(previous_position)) != 0:
                found_points.append(position)
                image_time.append(os.path.getmtime(os.path.join(folder, image)))
                previous_position = position
            else:
                found_points[-1] = position
                image_time[-1] = os.path.getmtime(os.path.join(folder, image))
        else:
            ending_counter += 1
            continue

        if show_match:
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            h, w = img.shape[:2]
            # Zobrazení výsledného obrázku
            cv2.namedWindow(f'{image} Found template {max_val:.04f}', cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f'{image} Found template {max_val:.04f}', int(w * 0.7), int(h * 0.7))
            cv2.imshow(f'{image} Found template {max_val:.04f}', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    found_points = np.array(found_points) * scale
    # found_points = np.vstack([found_points[0], found_points])[:10]
    found_points -= found_points[0]
    found_points = found_points[found_points[:, 1] < distance_limit['kinetic' if kinetic_type else 'static']]

    if len(found_points) < 3:
        print("\t\033[31;1mChyba: Není dostatek bodů.\033[0m")
        continue

    found_points /= 1000  # metres

    if kinetic_type:
        time_stamps = [image_time[i] - image_time[0] for i in range(len(image_time))][:len(found_points)]

    else:
        time_stamps = np.arange(0, len(found_points) * time, time)[:len(found_points)]

    positions = [np.array([np.linalg.norm(found_points[i] - found_points[0]) for i in range(len(found_points))])]

    # Aproximace křivky polynomem
    coefficients = np.polyfit(time_stamps, positions[0], 3)
    polynomial = np.poly1d(coefficients)

    # Výpočet hodnot aproximovaného polynomu
    positions.append(polynomial(time_stamps))

    """plt.figure()
    plt.plot(time_stamps, positions[0], 'o-')
    plt.plot(time_stamps, positions[1], linestyle="--", color="red")"""

    speeds = []
    average_speeds = []
    for p in positions:
        s = np.diff(p) / np.diff(time_stamps)
        # speed = np.diff(np.round(np.mean(found_points, axis=1), 6), axis=0) / np.diff(time_stamps)
        s = np.hstack([0, s])

        if use_averaging:  # Použití klouzavého průměru
            s = np.hstack([s[0], s])
            average_speeds.append(np.mean(s[s > 0]))

            kernel = np.ones(window_size) / window_size
            speed_av = s.copy()
            for _ in range(window_size - 1):
                speed_av = np.hstack([speed_av, speed_av[-1]])
            speed_av = np.convolve(speed_av, kernel, mode='valid')
            speeds.append(speed_av.copy())
        else:
            average_speeds.append(np.mean(s[s > 0]))
            speeds.append(s.copy())

        # mean_acc = (s[-1] - s[0]) / (time_stamps[-1] - time_stamps[0])
        # print(f"\tMean acceleration: {mean_acc:.4f} m/s^2")

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

    accelerations = []
    average_accelerations = []
    mean_acceleration = []
    linear_accelerations_indexes = []
    mean_linear_accelerations = []
    for n in range(2):
        linear_accelerations_indexes.append(np.arange(round(len(speeds[n]) * 0.2),
                                                      round(len(speeds[n]) * 0.85), 1))

        acc = np.diff(speeds[n], axis=0) / np.diff(time_stamps)[:len(speeds[n])]
        acc = np.hstack([0, acc])
        accelerations.append(acc.copy())

        average_accelerations.append(np.mean(acc[acc > 0]))

        mean_linear_accelerations.append(np.mean(accelerations[n][linear_accelerations_indexes[n]]))
        # lin_acc = acceleration[acceleration > 0][0]
        if 'F01' in os.path.basename(folder):
            accelerations_print.append(mean_linear_accelerations[n])

    print(f"\tAverage acceleration: {average_accelerations[1]:.4f} m/s^2")
    print(f"\tLinear mean acceleration: {mean_linear_accelerations[1]:.4f} m/s^2")
    print(f"\tLinear median acceleration: {np.median(accelerations[1][linear_accelerations_indexes[1]]):.4f} m/s^2")

    if show_graph:
        plt.figure()

        ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=3)
        ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)
        ax3 = plt.subplot2grid((3, 3), (1, 1), colspan=2)
        ax4 = plt.subplot2grid((3, 3), (2, 1), colspan=2)

        ax1.set_title("Position")
        ax1.plot(found_points[:, 0], found_points[:, 1], 'o-')
        ax1.axis('equal')
        ax1.invert_yaxis()
        ax1.set_xlabel('x [$m$]')
        ax1.set_ylabel('y [$m$]')

        ax2.set_title("Total displacement")
        ax2.plot(time_stamps, positions[0], '-', lw=3)
        ax2.plot(time_stamps, positions[1], linestyle="--", color="yellow")
        ax2.set_xlabel('t [$s$]')
        ax2.set_ylabel('d [$m$]')

        ax3.set_title("Velocity")
        if plot_real_data:
            ax3.plot(time_stamps[:len(speeds[0])], speeds[0], '-.', c='#D9A465', alpha=0.5)
        ax3.hlines(average_speeds[1], time_stamps[0], time_stamps[-1], color='tomato', linestyle='--',
                   label=f"Average speed: {average_speeds[1]:.2f} mm/s")
        ax3.plot(time_stamps[:len(speeds[1])], speeds[1], '-', c='orange')
        # ax2.plot(regression, speed_av[linear_points_indexes], '-', color='darkred')
        ax3.set_xlabel('t [$s$]')
        ax3.set_ylabel('v [$m/s$]')

        ax4.set_title("Acceleration")
        if plot_real_data:
            ax4.plot(time_stamps[:len(accelerations[0])], accelerations[0], '-.', c='#FF7476', alpha=0.5)
        ax4.hlines(average_accelerations[1], time_stamps[0], time_stamps[-1], color='darkorange', linestyle='--',
                   label=f"Average acceleration: {average_accelerations[1]:.2f} mm/s")
        ax4.plot(time_stamps[:len(accelerations[1])], accelerations[1], '-', color='red')
        ax4.hlines(mean_linear_accelerations[1], time_stamps[linear_accelerations_indexes[1][0]],
                   time_stamps[linear_accelerations_indexes[1][-1]], linestyle='--', color='darkred')
        ax4.set_xlabel('t [$s$]')
        ax4.set_ylabel('a [$m/s^2$]')

        # plt.tight_layout()
        plt.subplots_adjust(right=0.97, left=0.15, top=0.92, bottom=0.12, wspace=0.6, hspace=0.8)
        plt.show()

print(f"\nAverage acceleration: {np.mean(accelerations_print):.4f} m/s^2")
print(f"\nSTD acceleration: {np.std(accelerations_print):.4f} m/s^2")
print(f"\nMedian acceleration: {np.median(accelerations_print):.4f} m/s^2")
