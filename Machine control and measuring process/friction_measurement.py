def execute_command(port, command):
    if not port.is_open:
        print(f"{port} nelze otevřít.")
    else:
        port.write(command.encode())
        print("Příkaz odeslán:", command)
        time.sleep(2)
        start_time, received_data = time.time(), None
        while True:
            received_data = port.readline().decode().strip()
            if received_data != command and received_data != "" or command == "P":
                if command == "P":
                    print("\t", "Photo taken")
                    time.sleep(2)
                else:
                    print("\t", received_data)

                time.sleep(0.5)
                break
            elif start_time + command_time_limit < time.time():
                print(f"\tChyba: nepřijetí odpovědi příkazu s časovým limitem {command_time_limit // 60} minut.")
                return


def execute_command_and_measure(port, command, camera, x_lim, y_lim):
    if not port.is_open:
        print(f"{port} nelze otevřít.")
        return None, None
    else:
        images = []
        cap_stamps = []
        received_data = port.readline().decode().strip()
        _, frame = camera.read()
        cap_time: float = time.time()

        start_time = time.time()
        port.write(command.encode())
        print(f"\033[37mPříkaz odeslán: {command}\033[0m")
        time.sleep(0.2)
        while True:
            cap_time = time.time()
            _, frame = camera.read()  # Načtení snímku z kamery
            images.append(frame[y_lim:-y_lim, x_lim:-x_lim])
            cap_stamps.append(cap_time)
            received_data = port.readline().decode().strip()

            if received_data != command and received_data != "":
                print(f"\t\033[37m{received_data}\033[0m")
                break
            elif start_time + command_time_limit < cap_time:
                print("\nChyba: nepřijetí odpovědi posunu s měřením,"
                      f" s časovým limitem {command_time_limit // 60} minut.")
                return None, None
    return images, cap_stamps


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

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", np.int32(0.7 * width - (2 * v_lims)), np.int32(0.7 * height - (2 * h_lims)))
    cv2.imshow("Frame", frame[h_lims:-h_lims, v_lims:-v_lims])

    cv2.namedWindow("Crop")
    cv2.resizeWindow("Crop", 700, 0)
    cv2.createTrackbar("Vodorovne", "Crop", np.int32(1), w_max, get_width)
    cv2.createTrackbar("Svisle", "Crop", np.int32(1), h_max, get_height)
    cv2.resizeWindow("Crop", 350, 40)

    print("\n\tProces ukončíte pomocí klávesy:\033[32;1m 'ESC' \033[0m\n")

    while True:
        if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        if cv2.getWindowProperty("Crop", cv2.WND_PROP_VISIBLE) < 1:
            cv2.namedWindow("Crop")
            cv2.resizeWindow("Crop", 700, 0)
            cv2.createTrackbar("Vodorovne", "Crop", v_lims, w_max, get_width)
            cv2.createTrackbar("Svisle", "Crop", h_lims, h_max, get_height)
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


def make_measurement(camera_index, port, output_folder, x_limit=1, y_limit=1, distance=0.02, speed=0.005, time_span=10,
                     measuring_area=None, cam_width=1920, cam_height=1080, cam_fps=60):
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
    frame = frame[y_limit:-y_limit, x_limit:-x_limit]

    """cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", frame.shape[1] // 2, frame.shape[0] // 2)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    if measuring_area is None:
        decoded_object = decode(frame)
        if not decoded_object:
            print("\n\033[31;1mChyba: Nebyl nalezen QR kód.\033[0m\n")
            try:
                remove_folder(output_folder)
            except PermissionError as e:
                print(f"PermissionError: {e}")
            return
        if len(decoded_object) > 1:
            try:
                remove_folder(output_folder)
            except PermissionError as e:
                print(f"PermissionError: {e}")
            print("\n\033[33;1mPozor: Bylo detekováné více QR kódů.\033[0m")
            for obj in decoded_object:
                photo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.polylines(photo, [cv2.convexHull(np.array([point for point in obj.polygon], dtype=np.int32))],
                              True, (0, 255, 0), 2)
                cv2.namedWindow("QR Codes", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("QR Codes", photo.shape[1] // 2, photo.shape[0] // 2)
                cv2.imshow("QR Codes", photo)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return

        temp_x, temp_y, temp_w, temp_h = 4 * [None]
        for obj in decoded_object:
            temp_x, temp_y, temp_w, temp_h = cv2.boundingRect(np.array([point for point in obj.polygon],
                                                                       dtype=np.int32))
        temp_x, temp_y = np.int32(temp_x + temp_w * 0.2), np.int32(temp_y + temp_h * 0.2)
        temp_w, temp_h = np.int32(temp_x + temp_w * 0.6), np.int32(temp_y + temp_h * 0.6)
    else:
        measuring_area = np.int32(measuring_area).reshape(2, 2)
        temp_x, temp_y = measuring_area[0, :]
        temp_w, temp_h = measuring_area[1, :]

    template = frame[temp_y:temp_h, temp_x:temp_w]

    """cv2.namedWindow("Template", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Template", template.shape[1] // 2, template.shape[0] // 2)
    cv2.imshow("Template", template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    print("\n\033[32;1mZahájení sledování.\033[0m\n")

    counter, moment = 0, None
    total_time = time.time()
    start_time = time.time()
    cur_time: float = time.time()
    while True:
        photos = []
        cap_times = []
        start_time = time.time()
        while True:
            cur_time = time.time()
            _, frame = cap.read()  # Načtení snímku z kamery
            photos.append(frame[y_limit:-y_limit, x_limit:-x_limit])
            cap_times.append(cur_time)

            if start_time + time_span <= cur_time:
                break

        match = cv2.matchTemplate(photos[-1], template, cv2.TM_CCOEFF_NORMED)
        moment = "Waiting"
        # _, max_val, _, _ = cv2.minMaxLoc(match)

        if np.max(match) < 0.75:
            break
        else:
            time.sleep(0.1)
            print(f"\033[37mPosun číslo: \033[0m{counter}")
            counter += 1
            photos, cap_times = execute_command_and_measure(port, f"M {distance} {speed}", cap, x_limit, y_limit)

            if photos is None:
                print("\n\033[31;1mChyba pořízeny fotografie během měření.\033[0m")
                return
            elif photos is []:
                print("\n\033[31;1mNebyly pořízeny fotografie během měření.\033[0m")
                execute_command(port, f"M -{distance * counter} 1")
                return

            match = cv2.matchTemplate(photos[-1], template, cv2.TM_CCOEFF_NORMED)
            moment = "Moving"
            # _, max_val, _, _ = cv2.minMaxLoc(match)

            if np.max(match) < 0.75:
                break
            print("\t")

    # Uvolnění kamery a uložení fotek
    execute_command(port, f"M -{distance * counter} 1")

    """output_folder = os.path.join(output_folder, f"photos_output_{measurement_name}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)"""

    [cv2.imwrite(os.path.join(output_folder, f"cam_frame_{i + 1:03d}.png"), frame) for i, frame in enumerate(photos)]
    if len(cap_times) == len(photos):
        [os.utime(os.path.join(output_folder, f"cam_frame_{i + 1:03d}.png"), (c_time, c_time)) for i, c_time in
         enumerate(cap_times)]
    else:
        print("\n\033[31;1mChyba počtu snímků a časů.\033[0m")

    information = dict(moment=str(moment),
                       width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                       fps=int(cap.get(cv2.CAP_PROP_FPS)), time=int(cur_time - start_time),
                       total_time=int(time.time() - total_time), seconds=float(time_span),
                       photo_herizontal_crop=int(x_limit), photo_vertical_crop=int(y_limit),
                       movemnt_distance=float(distance), movemnt_speed=float(speed))
    with open(os.path.join(output_folder, 'settings.json'), 'w') as file:
        try:
            json.dump(information, file)
        except (TypeError, Exception) as e:
            print(f"\n\033[31;1mChyba uložení popisných dat.\n\t{e}\033[0m")
            print(information)
            print("\n")
    file.close()

    print(f"\n\033[33;1mCelkový čas měření:\033[0m \033[36;1m{information['total_time']} s\033[0m"
          f"\n\tMěření: \033[34;1m{os.path.basename(output_folder)}\033[0m\n")

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
        time.sleep(1)
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
    global first_movement

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
        frame = photo.copy()

        # Načtení QR kódů z obrázku
        decoded_objects = decode(photo)
        decoded_info = None

        photo = cv2.cvtColor(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGRA)
        if decoded_objects:
            if len(decoded_objects) > 1:
                for obj in decoded_objects:
                    cv2.polylines(photo, [cv2.convexHull(np.array([point for point in obj.polygon],
                                                                  dtype=np.int32))], True, (0, 255, 0), 2)
                    cv2.namedWindow("QR Codes", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("QR Codes", photo.shape[1] // 2, photo.shape[0] // 2)
                    cv2.imshow("QR Codes", photo)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                end_program("\nBylo detekováné více QR kódů.")

            temp_x, temp_y, temp_w, temp_h = 4 * [None]
            for obj in decoded_objects:
                cv2.polylines(photo, [cv2.convexHull(np.array([point for point in obj.polygon], dtype=np.int32))],
                              True, (0, 255, 0), 2)
                decoded_info = obj.data.decode('utf-8')
                temp_x, temp_y, temp_w, temp_h = cv2.boundingRect(np.array([point for point in obj.polygon],
                                                                           dtype=np.int32))
        else:
            print("\n\033[33;1mNenalezen QR kód.\033[0m")
            temp_x, temp_y, temp_w, temp_h = (photo.shape[1] // 2) - 1, (photo.shape[0] // 2) - 1, 0, 0

        if crop_photo:
            x_lim, y_lim = crop_image(temp_x, temp_y, temp_w, temp_h, photo)

            cv2.namedWindow("Cropped camera", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Cropped camera", np.int32(0.7 * photo[y_lim:-y_lim, x_lim:-x_lim].shape[1]),
                             np.int32(0.7 * photo[y_lim:-y_lim, x_lim:-x_lim].shape[0]))
            cv2.imshow("Cropped camera", photo[y_lim:-y_lim, x_lim:-x_lim])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            crop_photo = False

        if not decoded_objects:
            print("Zvolte název měření.")
            while True:
                decoded_info = str(input("\tZadejte jméno: "))
                print(f"\t\tZvolené jméno je: \033[34;1m{decoded_info}\033[0m\nJe název v pořádku?")
                ans = str(input("\n\t\tZadejte Y / N: "))
                if ans == "Y":
                    print("\n\tZvolena možnost 'Y'\n")
                    break
                elif ans == "N":
                    print("\n\tZvolena možnost 'N'\n")
                else:
                    print("\n Zadejte platnou odpověď.")

            if os.path.exists(os.path.join(output, decoded_info)):
                decoded_info = f"{decoded_info}_{time.strftime('%H-%M-%S_%d-%m-%Y', time.localtime(time.time()))}"

            import matplotlib.pyplot as plt
            from matplotlib.widgets import RectangleSelector

            def onselect(p0, p1):
                return

            figure, axes = plt.subplots(num="Označení hledané oblasti")
            plt.title("Označte hledanou oblast")

            axes.imshow(cv2.cvtColor(photo[y_lim:-y_lim, x_lim:-x_lim], cv2.COLOR_BGR2GRAY), cmap='gray')
            style = dict(facecolor="yellowgreen", edgecolor="darkgreen", alpha=0.3, linestyle='dashed', linewidth=1.5)
            area = RectangleSelector(axes, onselect, props=style, useblit=True, button=[1],
                                     minspanx=5, minspany=5, spancoords='pixels', interactive=True)

            plt.tight_layout()
            plt.show()
            area = np.int32(np.round(area.extents)).reshape(2, 2).T
        else:
            area = None

            print(f"Zvolte přívlastek názvu k měření: \033[32m{decoded_info}\033[0m.")
            while True:
                add_on = input("\tZadejte přívlastek: ")
                if add_on == "":
                    add_on = decoded_info
                else:
                    add_on = decoded_info + f"_{add_on}"
                print(f"\nZvolené jméno je: \033[34;1m{add_on}\033[0m\n\tJe název v pořádku?")
                ans = str(input("\n\t\tZadejte Y / N: "))
                if ans == "Y":
                    print("\n\tZvolena možnost 'Y'\n")
                    decoded_info = add_on
                    del add_on
                    break
                elif ans == "N":
                    print("\n\tZvolena možnost 'N'\n")
                else:
                    print("\n Zadejte platnou odpověď.")

        del photo, ans

        folder_path = os.path.join(output, decoded_info + "_o")

        if os.path.exists(folder_path):
            while True:
                # folder_path = folder_path + f"_{time.strftime('%H-%M-%S_%d-%m-%Y', time.localtime(time.time()))}"
                folder_path = folder_path + "o"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                    break
        else:
            os.makedirs(folder_path)
        cv2.imwrite(os.path.join(folder_path, f"_first_photo.png"), frame[y_lim:-y_lim, x_lim:-x_lim])
        del frame

        execute_command(ser, f"M {first_movement * 0.75} 0.75")
        execute_command(ser, f"M {first_movement * 0.15} 0.15")
        execute_command(ser, f"M {first_movement * 0.1} 0.1")

        make_measurement(camera_index=selected_camera_index, port=ser, output_folder=folder_path,
                         x_limit=x_lim, y_limit=y_lim, distance=measure_dist, speed=measure_speed, measuring_area=area,
                         time_span=waiting_time, cam_width=camera_width, cam_height=camera_height, cam_fps=camera_fps)

        execute_command(ser, f"M -{first_movement} 2")

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

                print(f"\n\033[34;1mChcete změnit první posun {first_movement}?\033[0m")
                while True:
                    ans = input("\t\tZadejte Y / N: ")
                    if ans == "Y":
                        print("\n\tZvolena možnost 'Y'\n")

                        try:
                            ans = input("\n\tZadejte číslo:  ").replace(",", ".")
                            first_movement = float(ans)

                        except (ValueError, Exception):
                            print("Špatně zadané číslo, znovu:.")
                        break
                    elif ans == "N":
                        print("\n\tZvolena možnost 'N'\n")
                        break
                    else:
                        print("\n Zadejte platnou odpověď.")

                print('\nPříprava nového testu:')
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
    import json
    # import threading
    import numpy as np
    from ctypes import windll
    from pyzbar.pyzbar import decode
    from serial.tools import list_ports
    from serial import Serial, SerialException

    output = r"C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\Friction_photos"

    camera_width = 1280  # cam_width = 3840 // 1920
    camera_height = 960  # cam_height = 2160 // 1080
    camera_fps = 100  # 60
    measure_dist = 0.25  # 0.1
    measure_speed = 0.1  # 0.005
    waiting_time = 10  # 10

    first_movement = 70  # 65 (75)

    speed_mode = True

    command_time_limit = 1800  # seconds

    # start cca 60.5 mm / 59.5
    # angle cca 20.7 degree
    # angle end cca 138 mm

    main()
    print("\n\033[35;1mKonec.\033[0m")
