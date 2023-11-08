import os
import cv2
import time
import serial
import numpy as np
from pyzbar.pyzbar import decode
from serial.tools import list_ports


def execute_command(port, command):
    if not port.is_open:
        print(f"{port} nelze otevřít.")
    else:
        port.write(command.encode())
        print("Příkaz odeslán:", command)
        time.sleep(1)

        while True:
            received_data = port.readline().decode().strip()
            if received_data != command and received_data != "" or command == "P":
                if command == "P":
                    print("\t", "Photo taken")
                else:
                    print("\t", received_data)

                time.sleep(1)
                break


def get_available_cameras(cam_type=cv2.CAP_MSMF):
    devices = []
    for c in range(10):
        capture = cv2.VideoCapture(c, cam_type)  # cv2.CAP_ANY
        if capture.isOpened():
            device_name = capture.getBackendName()
            devices.append(f"Kamera {c + 1} - {device_name}")
            capture.release()
    return devices


def show_available_cameras(cams):
    photos = []
    for n, camera in enumerate(cams):
        print(f"\t{n + 1}: {camera}")

        cam = cv2.VideoCapture(n)
        if not cam.isOpened():
            exit("Kamera nenalezena!")
        ret, frame = cam.read()
        cam.release()

        if not ret:
            exit("Nepodařilo se získat snímek z kamery!")

        photos.append(frame)

    labeled_photos = []
    for j, image in enumerate(photos, start=1):
        labeled_image = image.copy()
        label = f"WEBCAM {j}"

        # Přidání popisu v levém horním rohu
        cv2.putText(labeled_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        labeled_photos.append(labeled_image)

    combined_image = cv2.hconcat(labeled_photos)
    cv2.namedWindow("Combined Images", cv2.WINDOW_NORMAL)
    cv2.imshow("Combined Images", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def crop_image(x, y, w, h, frame):
    global v_lims, h_lims

    def get_width(*args):
        global v_lims
        v_lims = max(np.int32(cv2.getTrackbarPos("Vodorovne", "Crop")), 1)

    def get_height(*args):
        global h_lims
        h_lims = max(np.int32(cv2.getTrackbarPos("Svisle", "Crop")), 1)

    v_lims, h_lims = 1, 1
    height, width = frame.shape[:2]
    w_max = np.int32(min(x, width - x - w))
    h_max = np.int32(min(y, height - y - h))

    cv2.namedWindow("Crop")
    cv2.resizeWindow("Crop", 700, 0)
    cv2.createTrackbar("Vodorovne", "Crop", np.int32(1), w_max, get_width)
    cv2.createTrackbar("Svisle", "Crop", np.int32(1), h_max, get_height)
    cv2.resizeWindow("Crop", 350, 40)

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", np.int32(0.7 * width - (2 * v_lims)), np.int32(0.7 * height - (2 * h_lims)))
    cv2.imshow("Frame", frame[h_lims:-h_lims, v_lims:-v_lims])

    print("\n\tProces ukončíte pomocí klávesy 'ESC'")

    while True:
        if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        if cv2.getWindowProperty("Crop", cv2.WND_PROP_VISIBLE) < 1:
            cv2.namedWindow("Crop")
            cv2.resizeWindow("Crop", 700, 0)
            cv2.createTrackbar("Vodorovne", "Crop", np.int32(1), w_max, get_width)
            cv2.createTrackbar("Svisle", "Crop", np.int32(1), h_max, get_height)
            cv2.resizeWindow("Crop", 350, 40)

        cv2.resizeWindow("Frame", np.int32(0.7 * width - (2 * v_lims)), np.int32(0.7 * height - (2 * h_lims)))
        cv2.imshow("Frame", frame[h_lims:-h_lims, v_lims:-v_lims])

        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()
    return v_lims, h_lims


def make_measurement(camera_index, port, output_folder, x_limit=1, y_limit=1, distance=0.02, speed=0.05, time_span=10,
                     measuring_area=None):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)  # CAP_ANY // CAP_MSMF

    if not cap.isOpened():
        exit("Chyba: Webová kamera není dostupná.")
    ret, frame = cap.read()  # Načtení snímku z kamery
    if not ret:  # Kontrola, zda je webová kamera správně otevřena
        exit("Chyba: Snímek nebyl pořízen.")
    if measuring_area is None:
        decoded_object = decode(frame)
        if not decoded_object:
            exit("Chyba: Nebyl nalezen QR kód.")

        temp_x, temp_y, temp_w, temp_h = 4 * [None]
        for obj in decoded_object:
            measurement_name = obj.data.decode('utf-8')
            temp_x, temp_y, temp_w, temp_h = cv2.boundingRect(np.array([point for point in obj.polygon],
                                                                       dtype=np.int32))
        temp_x, temp_y = temp_x + np.int32(temp_w * 0.15), temp_y + np.int32(temp_h * 0.15)
        temp_w, temp_h = np.int32(temp_w * 0.85), np.int32(temp_h * 0.85)
    else:
        measuring_area = np.int32(measuring_area)
        temp_x, temp_y = measuring_area[0, :]
        temp_w, temp_h = measuring_area[1, :] - temp_x, temp_y

    template = frame[temp_y:temp_y + temp_h, temp_x:temp_x + temp_w]

    cam_width = 3840  # int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = 2160  # int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = 60  # cap.get(cv2.CAP_PROP_FPS)
    # print(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    cap.set(cv2.CAP_PROP_FPS, cam_fps)

    counter = 0
    while True:
        photos = []
        start_time = time.time()
        while True:
            _, frame = cap.read()  # Načtení snímku z kamery
            photos.append(frame[y_limit:-y_limit, x_limit:-x_limit])

            if start_time + time_span <= time.time():
                break

        match = cv2.matchTemplate(photos[-1], template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(match)

        if max_val < 0.65:
            break
        else:
            time.sleep(1)
            counter += 1
            execute_command(port, f"M {distance} {speed}")
            print("\t")
            time.sleep(2)

    # Uvolnění kamery a uložení fotek
    cap.release()
    execute_command(port, f"M {distance * -counter} {speed}")
    output_folder = os.path.join(output_folder, f"photos_output_{measurement_name}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    [cv2.imwrite(os.path.join(output_folder, f"cam_frame_{i + 1: 03d}.png"), frame) for i, frame in enumerate(photos)]


def capture_webcam_photo(camera_index, filename="photo.jpg", save_photo=False):
    # Otevření kamery
    capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        print("Kamera nenalezena!")
        return None

    # Zjistit maximální podporované rozlišení
    width = 1920  # Změňte na požadovanou šířku
    height = 1080  # Změňte na požadovanou výšku
    cam_fps = 60

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, cam_fps)

    # Získání snímku z kamery
    ret, frame = capture.read()

    # Uzavření kamery
    capture.release()

    if not ret:
        print("Nepodařilo se získat snímek z kamery!")
        return None

    if frame is not None and save_photo:
        # Uložení snímku do souboru
        cv2.imwrite(filename, frame)
        print(f"Snímek uložen: [ {filename} ]")

        """cv2.imshow("Webcam", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""

    return frame


def main():
    cv2.setLogLevel(0)

    available_ports = [ports.device for ports in list_ports.comports()]

    if not available_ports:
        exit("Nebyly nalezeny žádné dostupné COM porty.")

    print("Dostupné COM porty:")
    for i, port in enumerate(available_ports):
        print(f"\t{i}: {port}")

    if len(available_ports) == 1:
        selected_port_index = 0
    else:
        while True:
            selected_port_index = input("Vyber číslo COM portu: ")
            try:
                selected_port_index = int(selected_port_index)
                if selected_port_index < 0 or selected_port_index >= len(available_ports):
                    print("Neplatná volba COM portu!")
                else:
                    break
            except ValueError:
                print("Neplatná volba COM portu!")

    selected_port = available_ports[selected_port_index]
    print(f"Vybrán port: [ {selected_port} ]")

    try:
        ser = serial.Serial(selected_port, baudrate=9600, timeout=1)
        time.sleep(2)
        data, empty_count = None, 0
        print("\n")
        while data != "Testing pin: 21 ON OFF":
            data = ser.readline().decode().strip()
            if data != "":
                print(data)
            elif empty_count == 5:
                break
            else:
                empty_count += 1

        del empty_count, data
    except ValueError as ve:
        exit("Chyba serial port", ve)
    print("\n")

    cameras = get_available_cameras()

    if not cameras:
        exit("Žádná kamera nenalezena!")

    print("Dostupné kamery:")
    show_available_cameras(cameras)

    while True:
        selected_camera_index = input("Vyber číslo kamery: ")
        try:
            selected_camera_index = int(selected_camera_index) - 1
            if selected_camera_index < 0 or selected_camera_index >= len(cameras):
                print("Neplatná volba kamery!")
            else:
                break
        except ValueError:
            print("Neplatná volba kamery!")

    while True:
        photo = capture_webcam_photo(selected_camera_index, save_photo=False)

        if photo is None:
            exit("Nebyla pořízena fotografie.")

        # Načtení QR kódů z obrázku
        decoded_objects = decode(photo)
        decoded_info = None

        if decoded_objects:
            temp_x, temp_y, temp_w, temp_h = 4 * [None]
            for obj in decoded_objects:
                photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
                cv2.polylines(photo, [cv2.convexHull(np.array([point for point in obj.polygon], dtype=np.int32))],
                              True, (0, 255, 0), 2)
                decoded_info = obj.data.decode('utf-8')
                temp_x, temp_y, temp_w, temp_h = cv2.boundingRect(np.array([point for point in obj.polygon],
                                                                           dtype=np.int32))
        else:
            print("Nenalezen QR kód.")
            temp_x, temp_y, temp_w, temp_h = (photo.shape[1] // 2) - 1, (photo.shape[0] // 2) - 1, 0, 0

        x_lim, y_lim = crop_image(temp_x, temp_y, temp_w, temp_h, cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY))

        cv2.namedWindow("Cropped camera", cv2.WINDOW_NORMAL)
        cv2.imshow("Cropped camera", photo[y_lim:-y_lim, x_lim:-x_lim])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if not decoded_objects:
            decoded_info = f"cam_measurement_{time.strftime('%H-%M-%S_%d-%m-%Y', time.localtime(time.time()))}"
            import matplotlib.pyplot as plt
            from matplotlib.widgets import RectangleSelector

            def onselect(p0, p1):
                pass

            figure, axes = plt.subplots()
            plt.title("Označte hledanou oblast")

            axes.imshow(cv2.cvtColor(photo[y_lim:-y_lim, x_lim:-x_lim], cv2.COLOR_BGR2GRAY), cmap='gray')
            style = dict(facecolor="yellowgreen", edgecolor="darkgreen", alpha=0.2, linestyle='dashed', linewidth=1.5)
            area = RectangleSelector(axes, onselect, props=style, useblit=True, button=[1],
                                     minspanx=5, minspany=5, spancoords='pixels', interactive=True)

            plt.tight_layout()
            plt.show()
            area = np.int32(np.round(area.extents)).reshape(2, 2).T
        else:
            area = None

        folder_path = os.path.join(output, decoded_info)

        if os.path.exists(folder_path):
            while True:
                folder_path = folder_path + f"_{time.strftime('%H-%M-%S_%d-%m-%Y', time.localtime(time.time()))}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                    break
        else:
            os.makedirs(folder_path)

        make_measurement(camera_index=selected_camera_index, port=ser, output_folder=folder_path,
                         x_limit=x_lim, y_limit=y_lim, distance=0.01, speed=0.01, time_span=10, measuring_area=area)

        print("\nChcete provést další měření?")
        while True:
            ans = input("\t\tZadejte Y / N: ")
            if ans == "Y":
                print("\n\tZvolena možnost 'Y'")
                break
            elif ans == "N":
                print("\n\tZvolena možnost 'N'")
                break
            else:
                print("\n Zadejte platnou odpověď.")

        if ans == "N":
            break

    ser.close()


if __name__ == "__main__":
    output = r"C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\friction"

    main()
    print("\nKonec.")
