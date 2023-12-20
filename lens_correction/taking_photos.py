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


def live_webcam(camera_index=None, width=1920, height=1080, cam_fps=60, camera=None):
    # Otevřít video nebo kamery
    if camera is None:
        if isinstance(camera_index, int):
            cap = cv2.VideoCapture(camera_index)  # CAP_ANY // CAP_MSMF
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, cam_fps)
        else:
            end_program("Špatný index kamery.")
    else:
        cap = camera

    ret, frame_ = cap.read()
    if not ret:
        end_program("\nNebyla pořízeno video z webkamery.")

    h, w = frame_.shape[:2]

    if w != width or h != height:
        print("\n\033[31;1mNesouhlasí zadaný formát videa a pořízenou fotografií.\033[0m"
              f"\n\t\033[31mZadání: {width} x {height}, Kamera: {w} x {h}\033[0m")
    else:
        print(f"\nNastavení kamery:\n\tObraz: {w} x {h}\n\tFPS: {int(cap.get(cv2.CAP_PROP_FPS))}")

    # frame_ = frame_[500:550, 1200:1250]
    # h, w = frame_.shape[0] * 20, frame_.shape[1] * 20
    h, w = frame_.shape[:2]

    cv2.namedWindow("WebCam", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("WebCam", np.int32(0.7 * w), np.int32(0.7 * h))
    cv2.imshow("WebCam", frame_)

    while True:
        if cv2.getWindowProperty("WebCam", cv2.WND_PROP_VISIBLE) < 1:
            cv2.namedWindow("WebCam", cv2.WINDOW_NORMAL)

        cv2.resizeWindow("WebCam", np.int32(0.7 * w), np.int32(0.7 * h))
        _, frame_ = cap.read()
        cv2.imshow('WebCam', frame_)
        # cv2.imshow('WebCam', frame_[500:550, 1200:1250])

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
    exit(text)


def main():
    cv2.setLogLevel(0)

    selected_camera_index = manage_cameras()
    print("\n")

    live_webcam(camera_index=selected_camera_index, camera=None, width=camera_width, height=camera_height,
                cam_fps=camera_fps)

    capture = cv2.VideoCapture(selected_camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    capture.set(cv2.CAP_PROP_FPS, camera_fps)
    i = 1

    while True:
        live_webcam(camera_index=selected_camera_index, camera=capture, width=camera_width, height=camera_height,
                    cam_fps=camera_fps)

        _, frame = capture.read()
        # cv2.imwrite(f"1920x1080/WebCam_{i:03d}.jpg", frame)
        print("Uložení fotky číslo:", i)
        i += 1


if __name__ == "__main__":
    import cv2
    import numpy as np

    camera_width = 1920  # cam_width = 3840
    camera_height = 1080  # cam_height = 2160
    camera_fps = 60  # 100

    speed_mode = True

    # 4032×3040@10 fps; 3840×2160@20 fps; 2592×1944@30 fps; 2560×1440@30 fps; 1920×1080@60 fps; 1600×1200@50 fps;
    # 1280×960@100 fps; 1280×760@100 fps; 640×480@80 fps

    main()
    print("\n\033[35;1mKonec.\033[0m")
