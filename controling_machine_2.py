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


def make_measurement(camera_index, output_folder, txt_path, x_limit=1, y_limit=1, command_distance=0, command_period=0,
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

    images = 1
    cycler = True

    _, frame = cap.read()  # Načtení snímku z kamery

    _, frame = cap.read()  # Načtení snímku z kamery

    # images.append(frame[y_limit:-y_limit, x_limit:-x_limit])
    cv2.imwrite(os.path.join(output_folder, f"Frame_{0:03d}.jpg"), frame[y_limit:-y_limit, x_limit:-x_limit])

    command = f"MMDIC2 {command_distance} {command_period}"

    print("Dejte myš na dané místo.")
    time.sleep(15)
    print("Start")

    print(f"\033[37mPříkaz k odeslání: {command}\033[0m")

    # Pauza na případné otevření cílové aplikace nebo okna
    time.sleep(1)

    # Kliknutí na střed obrazovky
    pyautogui.click(pyautogui.size()[0] // 2, pyautogui.size()[1] // 2)

    # Stisk kláves Ctrl + R
    pyautogui.hotkey('ctrl', 'r')

    # Pauza pro načtení a zpracování stránky (přizpůsobte podle potřeby)
    time.sleep(1)

    # Vložení textu a stisk Enter
    pyautogui.typewrite(txt_path)
    # pyautogui.write(txt_path)
    pyautogui.press('enter')

    # Pauza mezi operacemi
    time.sleep(0.5)

    # Stisk kláves Ctrl + T
    pyautogui.hotkey('ctrl', 't')

    # Pauza pro načtení a zpracování nové stránky (přizpůsobte podle potřeby)
    time.sleep(0.5)

    # Přejde na specifické souřadnice, klikne a vloží text
    # pyautogui.click(*(940, 300))
    # pyautogui.hotkey('ctrl', 'a')
    pyautogui.typewrite(command)

    # Pauza mezi operacemi
    time.sleep(0.5)

    # Přejde na další souřadnice a klikne
    # pyautogui.click(*(1180, 230))
    # Simulace stisknutí klávesy Tab + Shift
    pyautogui.hotkey('shift', 'tab')
    pyautogui.press('enter')

    # Stisk kláves Ctrl + W
    pyautogui.hotkey('ctrl', 'w')

    start_time = time.time()

    while cycler:
        current_time = time.time()
        if start_time + command_period <= current_time:
            # if "taken photo" in log_output:
            _, frame_ = cap.read()  # Načtení snímku z kamery
            cv2.imwrite(os.path.join(output_folder, f"Frame_{images + 1:03d}.jpg"),
                        frame[y_limit:-y_limit, x_limit:-x_limit])
            images += 1
            # images.append(frame_[y_limit:-y_limit, x_limit:-x_limit])
            start_time = current_time
        elif start_time + command_period * 0.5 <= current_time <= start_time + command_period * 0.55:
            with open(txt_path, 'r') as file:
                # f = file.readlines()[-1].strip()
                # print("čtení souboru:", f)
                if "HOTOVO" == file.readlines()[-1].strip():
                    cycler = False
                    break

            # if os.path.getsize(txt_path) > 0:
            #    break
            # Otevření souboru v režimu čtení ('r' znamená čtení)
            # with open(file_path, 'r') as file:
            #    if 1 == int(file.readline().strip()):
            #        break

    # Pauza mezi operacemi
    time.sleep(0.5)

    # Stisk kláves Ctrl + Shift + R
    pyautogui.hotkey('ctrl', 'shift', 'r')

    # [cv2.imwrite(os.path.join(output_folder, f"Frame_{i + 1:03d}.jpg"), frame) for i, frame in enumerate(images)]

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


def manage_cameras():
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

    return camera_index


def end_program(text=None):
    # původní stav
    windll.kernel32.SetThreadExecutionState(0x80000000)
    exit(text)


def main():
    # Zabránění spánku a vypnutí obrazovky
    windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    cv2.setLogLevel(0)

    selected_camera_index = manage_cameras()
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

        folder_path_photos = os.path.join(output_photos, name, "detail_original")
        file_path_txt = os.path.join(output_folder_txt, name + ".txt")

        if os.path.exists(folder_path_photos):
            while True:
                folder_path_photos = (folder_path_photos +
                                      f"_{time.strftime('%H-%M-%S_%d-%m-%Y', time.localtime(time.time()))}")
                if not os.path.exists(folder_path_photos):
                    os.makedirs(folder_path_photos)
                    break
        else:
            os.makedirs(folder_path_photos)

        """while True:
            if os.path.exists(file_path_txt):
                break
            else:
                print(f"\nSoubor \033[36;1m{name + '.txt'} neexistuje."
                      "\n\t\033[0m \033[34;1mZadejte 'Y' až bude vytvořen.\033[0m")
                while True:
                    ans_end = input("\t\tZadejte Y / N: ")
                    if ans_end == "Y":
                        print("\n\tZvolena možnost 'Y'\n")
                        break
                    elif ans_end == "N":
                        print("\n\tZvolena možnost 'N'\nSoubor nevytvořen.")
                    else:
                        print("\n Zadejte platnou odpověď.")"""

        make_measurement(camera_index=selected_camera_index, output_folder=folder_path_photos, txt_path=file_path_txt,
                         command_distance=measurement_distance, command_period=measurement_periods, x_limit=x_lim,
                         y_limit=y_lim, cam_width=camera_width, cam_height=camera_height, cam_fps=camera_fps)

        print("\n\033[34;1mChcete provést další měření?\033[0m")
        while True:
            ans_end = input("\t\tZadejte Y / N: ")
            if ans_end == "Y":
                print("\n\tZvolena možnost 'Y'\n")
                break
            elif ans_end == "N":
                print("\n\tZvolena možnost 'N'\n")
                break
            else:
                print("\n Zadejte platnou odpověď.")
        if ans_end == "N":
            break

    # původní stav
    windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    import os
    import cv2
    import time
    import pyautogui
    import numpy as np
    from ctypes import windll

    output_photos = r"C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos"
    output_folder_txt = r"C:\Users\matej\Desktop\mereni"

    camera_width = 4032  # cam_width = 3840
    camera_height = 3040  # cam_height = 2160
    camera_fps = 10  # 100

    # 4032×3040@10fps; 3840×2160@20fps; 2592×1944@30fps; 2560×1440@30fps; 1920×1080@60fps; 1600×1200@50fps;
    # 1280×960@100fps; 1280×760@100fps; 640×480@80fps

    measurement_distance = 5
    measurement_periods = 5

    speed_mode = True

    main()
    print("\n\033[35;1mKonec.\033[0m")
