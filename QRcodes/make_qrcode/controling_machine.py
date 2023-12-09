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
                    time.sleep(2)
                else:
                    print("\t", received_data)

                time.sleep(1)
                break


def execute_command_and_measure(port, distance, period, camera, x_lim, y_lim):
    if not port.is_open:
        print(f"{port} nelze otevřít.")
        return None
    else:
        images = []
        _, frame = camera.read()  # Načtení snímku z kamery
        log = []

        images.append(frame[y_lim:-y_lim, x_lim:-x_lim])

        command = f"MMDIC {distance} {period}"
        port.write(command.encode())
        print(f"\033[37mPříkaz odeslán: {command}\033[0m")

        time.sleep(0.5)

        def measure():
            start_time = time.time()
            while True:
                log_output = port.readline().decode().strip()
                log.append(log_output)
                current_time = time.time()
                if start_time + period >= current_time:
                    _, frame_ = camera.read()  # Načtení snímku z kamery
                    images.append(frame_[y_lim:-y_lim, x_lim:-x_lim])
                    start_time = current_time

                if "HOTOVO " in log_output:
                    break

        while True:
            output_line = port.readline().decode().strip()

            if "Photo 0 taken at this point" in output_line:
                measure()
                break

    return images


def get_available_cameras(cam_type=None):
    if cam_type is None:
        cam_type = cv2.CAP_MSMF
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
            end_program("\nKamera nenalezena!")
        ret, frame = cam.read()
        cam.release()

        if not ret:
            end_program("\nNepodařilo se získat snímek z kamery!")

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


def crop_image(frame):
    global v_lims, h_lims

    def get_width(*args):
        global v_lims
        v_lims = max(np.int32(cv2.getTrackbarPos("Vodorovne", "Crop")), 1)

    def get_height(*args):
        global h_lims
        h_lims = max(np.int32(cv2.getTrackbarPos("Svisle", "Crop")), 1)

    v_lims, h_lims = 1, 1
    height, width = frame.shape[:2]
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", np.int32(0.7 * width - (2 * v_lims)), np.int32(0.7 * height - (2 * h_lims)))
    cv2.imshow("Frame", frame[h_lims:-h_lims, v_lims:-v_lims])

    cv2.namedWindow("Crop")
    cv2.resizeWindow("Crop", 700, 0)
    cv2.createTrackbar("Vodorovne", "Crop", np.int32(1), np.int32((width - 1) // 2), get_width)
    cv2.createTrackbar("Svisle", "Crop", np.int32(1), np.int32((height - 1) // 2), get_height)
    cv2.resizeWindow("Crop", 350, 40)

    print("\n\tProces ukončíte pomocí klávesy:\033[32;1m 'ESC' \033[0m\n")

    while True:
        if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        if cv2.getWindowProperty("Crop", cv2.WND_PROP_VISIBLE) < 1:
            cv2.namedWindow("Crop")
            cv2.resizeWindow("Crop", 700, 0)
            cv2.createTrackbar("Vodorovne", "Crop", v_lims, np.int32((width - 1) // 2), get_width)
            cv2.createTrackbar("Svisle", "Crop", h_lims, np.int32((height - 1) // 2), get_height)
            cv2.resizeWindow("Crop", 350, 40)

        cv2.resizeWindow("Frame", np.int32(0.7 * width - (2 * v_lims)), np.int32(0.7 * height - (2 * h_lims)))
        cv2.imshow("Frame", frame[h_lims:-h_lims, v_lims:-v_lims])

        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()
    return v_lims, h_lims


def remove_folder(folder):
    # Seznam všech souborů a složek včetně jejich cest
    all_files = [os.path.join(folder, file) for file in os.listdir(folder)]
    # Smazání všech souborů
    [os.remove(file) for file in all_files if os.path.isfile(file)]
    # Rekurzivní smazání obsahu všech složek
    [remove_folder(file) for file in all_files if os.path.isdir(file)]
    os.rmdir(folder)


def make_measurement(camera_index, port, output_folder, x_limit=1, y_limit=1, command_input1=0, command_input2=0,
                     cam_width=1920, cam_height=1080, cam_fps=60):
    cap = cv2.VideoCapture(camera_index)  # CAP_ANY // CAP_MSMF

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    cap.set(cv2.CAP_PROP_FPS, cam_fps)

    if not cap.isOpened():
        try:
            remove_folder(output_folder)
        except PermissionError as e:
            print(f"PermissionError: {e}")
        print("\n\033[31;1mChyba: Webová kamera není dostupná.\033[0m")
        return
    ret, frame = cap.read()  # Načtení snímku z kamery
    if not ret:  # Kontrola, zda je webová kamera správně otevřena
        try:
            remove_folder(output_folder)
        except PermissionError as e:
            print(f"PermissionError: {e}")
        print("\n\033[31;1mChyba: Snímek nebyl pořízen.\033[0m")
        return

    photos = execute_command_and_measure(port, command_input1, command_input2, cap, x_limit, y_limit)

    [cv2.imwrite(os.path.join(output_folder, f"Frame_{i + 1: 03d}.png"), frame) for i, frame in enumerate(photos)]

    print(f"\n\tMěření: \033[34;1m{os.path.basename(output_folder)}\033[0m\n")

    cap.release()


def capture_webcam_photo(camera_index=None, filename="photo.jpg", save_photo=False, width=1920, height=1080, cam_fps=60,
                         camera=None):
    # Otevření kamery
    if camera is None:
        capture = cv2.VideoCapture(camera_index)
    else:
        capture = camera

    if not capture.isOpened():
        print("\n\033[31;1mKamera nenalezena!\033[0m")
        return None

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, cam_fps)

    # Získání snímku z kamery
    ret, frame = capture.read()

    # Uzavření kamery
    if camera is None:
        capture.release()

    if not ret:
        print("\n\033[31;1mNepodařilo se získat snímek z kamery!\033[0m")
        return None

    if frame is not None and save_photo:
        # Uložení snímku do souboru
        cv2.imwrite(filename, frame)
        print(f"\033[32m\nSnímek uložen: [ {filename} ]\033[0m")

        """cv2.imshow("Webcam", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""

    return frame


def live_webcam(camera_index=None, width=1920, height=1080, cam_fps=60, camera=None):
    # Otevřít video nebo kamery
    if camera is None:
        cap = cv2.VideoCapture(camera_index)
    else:
        cap = camera

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, cam_fps)

    ret, frame = cap.read()
    if not ret:
        end_program("\nNebyla pořízeno video z webkamery.")

    h, w = frame.shape[:2]

    if w != width or h != height:
        print("\n\033[31;1mNesouhlasí zadaný formát videa a pořízenou fotografií.\033[0m"
              f"\n\t\033[31mZadání: {width} x {height}, Kamera: {w} x {h}\033[0m")
    else:
        print(f"\nNastavení kamery:\n\tObraz: {w} x {h}\n\tFPS: {int(cap.get(cv2.CAP_PROP_FPS))}")

    cv2.namedWindow("WebCam", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("WebCam", np.int32(0.7 * w), np.int32(0.7 * h))
    cv2.imshow("WebCam", frame)

    while True:
        if cv2.getWindowProperty("WebCam", cv2.WND_PROP_VISIBLE) < 1:
            cv2.namedWindow("WebCam", cv2.WINDOW_NORMAL)

        cv2.resizeWindow("WebCam", np.int32(0.7 * w), np.int32(0.7 * h))
        _, frame = cap.read()
        cv2.imshow('WebCam', frame)

        key = cv2.waitKey(1)  # Čekat na klávesu po dobu 1 ms
        if key == 27:
            break  # Pokud byla stisknuta klávesa ESC, ukončete cyklus

    # Uvolnit video capture a zavřít okno
    if camera is None:
        cap.release()
    cv2.destroyAllWindows()


def manage_ports():
    available_ports = [ports.device for ports in list_ports.comports()]

    if not available_ports:
        end_program("\nNebyly nalezeny žádné dostupné COM porty.")

    print("\nDostupné COM porty:")
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

    port = available_ports[selected_port_index]
    print(f"Vybrán port: [ {port} ]")

    return port


def manage_cameras(port):
    ser_port = None
    try:
        ser_port = Serial(port, baudrate=9600, timeout=1)
        time.sleep(2)
        data, empty_count = None, 0
        print("\n")
        while data != "Testing pin: 21 ON OFF":
            data = ser_port.readline().decode().strip()
            if data != "":
                print(data)
            elif empty_count == 5:
                break
            else:
                empty_count += 1

        del empty_count, data
    except (ValueError, SerialException, Exception) as e:
        end_program(f"\n\033[31;1mChyba serial port, {e}\033[0m")
    print("\n")

    if speed_mode:
        print("Auto výběr kamery číslo: 1")
        camera_index = 0
    else:
        cameras = get_available_cameras()

        if not cameras:
            end_program("\n\033[31;1mŽádná kamera nenalezena!\033[0m")

        print("Dostupné kamery:")
        show_available_cameras(cameras)

        while True:
            camera_index = input("Vyber číslo kamery: ")
            try:
                camera_index = int(camera_index) - 1
                if camera_index < 0 or camera_index >= len(cameras):
                    print("Neplatná volba kamery!")
                else:
                    break
            except ValueError:
                print("Neplatná volba kamery!")

    return ser_port, camera_index


def end_program(text=None):
    # původní stav
    windll.kernel32.SetThreadExecutionState(0x80000000)
    exit(text)


def main():
    # Zabránění spánku a vypnutí obrazovky
    windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    cv2.setLogLevel(0)

    ser, selected_camera_index = manage_cameras(manage_ports())
    print("\n")

    capture = None  # cv2.VideoCapture(selected_camera_index)

    live_webcam(camera_index=selected_camera_index, camera=capture, width=camera_width, height=camera_height,
                cam_fps=camera_fps)

    crop_photo = True
    x_lim, y_lim = None, None
    while True:
        photo = capture_webcam_photo(camera_index=selected_camera_index, camera=capture, save_photo=False,
                                     width=camera_width, height=camera_height, cam_fps=camera_fps)

        if capture is not None:
            capture.release()

        if photo is None:
            end_program("\nNebyla pořízena fotografie.")

        if crop_photo:
            x_lim, y_lim = crop_image(photo)

            cv2.namedWindow("Cropped camera", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Cropped camera", np.int32(0.7 * photo[y_lim:-y_lim, x_lim:-x_lim].shape[1]),
                             np.int32(0.7 * photo[y_lim:-y_lim, x_lim:-x_lim].shape[0]))
            cv2.imshow("Cropped camera", photo[y_lim:-y_lim, x_lim:-x_lim])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            crop_photo = False
        del photo

        while True:
            name = input("\nZadejte jméno měření:  ")
            print(f"\tJe název: \033[35;1m{name}\033[0m v pořádku?")

            ans = input("\t\tZadejte Y / N: ")
            if ans == "Y":
                print("\n\tZvolena možnost 'Y'\n")
                break
            elif ans == "N":
                print("\n\tZvolena možnost 'N'\n")
            else:
                print("\n Zadejte platnou odpověď.")

        folder_path = os.path.join(output, name)

        if os.path.exists(folder_path):
            while True:
                folder_path = folder_path + f"_{time.strftime('%H-%M-%S_%d-%m-%Y', time.localtime(time.time()))}"
                folder_path = folder_path
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                    break
        else:
            os.makedirs(folder_path)

        make_measurement(camera_index=selected_camera_index, port=ser, output_folder=folder_path,
                         command_input1=measurement_distance, command_input2=measurement_periods, x_limit=x_lim,
                         y_limit=y_lim, cam_width=camera_width, cam_height=camera_height, cam_fps=camera_fps)

        execute_command(ser, f"M -{measurement_distance} 2")

        print("\n\033[34;1mChcete provést další měření?\033[0m")
        while True:
            ans_end = input("\t\tZadejte Y / N: ")
            if ans_end == "Y":
                print("\n\tZvolena možnost 'Y'\n")

                print("\n\033[34;1mChcete provést posun?\033[0m")
                while True:
                    ans = input("\t\tZadejte Y / N: ")
                    if ans == "Y":
                        print("\n\tZvolena možnost 'Y'\n")
                        move = input("\n\tZadejte číslo:  ")
                        try:
                            move = float(move)
                            execute_command(ser, f"M {move} 1")

                            print("Chcete provést další posun?")
                            ans = input("\t\tZadejte Y / N: ")
                            if ans == "Y":
                                print("\n\tZvolena možnost 'Y'\n")
                            elif ans == "N":
                                print("\n\tZvolena možnost 'N'\n")
                                break
                            else:
                                print("\n Zadejte platnou odpověď.")

                        except (ValueError, Exception):
                            print("Špatně zadané číslo.")

                    elif ans == "N":
                        print("\n\tZvolena možnost 'N'\n")
                        break
                    else:
                        print("\n Zadejte platnou odpověď.")

                break
            elif ans_end == "N":
                print("\n\tZvolena možnost 'N'\n")
                break
            else:
                print("\n Zadejte platnou odpověď.")
        if ans_end == "N":
            break

    ser.close()
    # původní stav
    windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    import os
    import cv2
    import time
    # import threading
    import numpy as np
    from ctypes import windll
    from serial.tools import list_ports
    from serial import Serial, SerialException

    output = r"C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos"

    camera_width = 1920  # cam_width = 3840
    camera_height = 1080  # cam_height = 2160
    camera_fps = 60  # 100

    measurement_distance = 10
    measurement_periods = 10

    speed_mode = True

    main()
    print("\n\033[35;1mKonec.\033[0m")
