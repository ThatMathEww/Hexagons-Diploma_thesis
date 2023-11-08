import os
import time
import cv2
import serial
from serial.tools import list_ports


def process_command(command):
    print(command)
    if command == "take_photo":
        print("Pořizování fotografie...")
        #
        return False
    elif command == "start_recording":
        print("Zahájení nahrávání...")
        #
        return False
    elif command == "stop_recording":
        print("Zastavení nahrávání...")
        #
        return True
    else:
        # print("Neznámý příkaz:", command)
        return False


def receive_commands():
    while True:
        received_data = ser.readline().decode().strip()
        if received_data:
            should_break = process_command(received_data)
            if should_break:
                break


def get_available_ports():
    return [ports.device for ports in list_ports.comports()]


def send_command(command):
    if not ser.is_open:
        print(f"Port {selected_port} nelze otevřít.")
    else:
        ser.write(command.encode())
        print("Příkaz odeslán:", command)
        time.sleep(1)


def execute_command(command):
    if not ser.is_open:
        print(f"Port {selected_port} nelze otevřít.")
    else:
        ser.write(command.encode())
        print("Příkaz odeslán:", command)
        time.sleep(1)

        while True:
            received_data = ser.readline().decode().strip()
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


def are_valid_image_paths(paths):
    for path in paths:
        if not os.path.exists(path):
            return False
    return True


def add_image_labels(images):
    labeled_images = []
    for j, image in enumerate(images, start=1):
        labeled_image = image.copy()
        label = f"WEBCAM {j}"

        # Přidání popisu v levém horním rohu
        cv2.putText(labeled_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        labeled_images.append(labeled_image)

    return labeled_images


def show_images_side_by_side(images):
    if are_valid_image_paths(images):
        images = [cv2.imread(path) for path in images]

    combined_image = cv2.hconcat(images)
    cv2.namedWindow("Combined Images", cv2.WINDOW_NORMAL)
    cv2.imshow("Combined Images", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_available_cameras(cams):
    photos = []
    for n, camera in enumerate(cams):
        print(f"{n + 1}: {camera}")
        photos.append(get_webcam_photo(n))

    labeled_photos = add_image_labels(photos)
    show_images_side_by_side(labeled_photos)


def get_webcam_photo(camera_index):
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        print("Kamera nenalezena!")
        return None
    ret, frame = cam.read()
    cam.release()
    if not ret:
        print("Nepodařilo se získat snímek z kamery!")
        return None
    return frame


def capture_and_save_webcam_photo(camera_index, filename):
    # Otevření kamery
    capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        print("Kamera nenalezena!")
        return None

    # Získání snímku z kamery
    ret, frame = capture.read()

    # Uzavření kamery
    capture.release()

    if not ret:
        print("Nepodařilo se získat snímek z kamery!")
        return None

    # Uložení snímku do souboru
    # cv2.imwrite(output_filename, frame)
    cv2.imshow("Webcam", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Snímek uložen: [ {filename} ]")
    return frame


if __name__ == "__main__":
    selected_port, selected_port_index = None, None
    count = 25
    captured_photos = []
    cv2.setLogLevel(0)

    available_ports = get_available_ports()

    if not available_ports:
        print("Nebyly nalezeny žádné dostupné COM porty.")
    else:
        print("Dostupné COM porty:")
        for i, port in enumerate(available_ports):
            print(f"{i}: {port}")

        selected_port_index = int(input("Vyber číslo COM portu: "))
        if selected_port_index < 0 or selected_port_index >= len(available_ports):
            print("Neplatná volba COM portu!")
        else:
            selected_port = available_ports[selected_port_index]
            print(f"Vybrán port: [ {selected_port} ]")

            try:
                with serial.Serial(selected_port, baudrate=9600, timeout=1) as ser:
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
                    print("\n")

                    for _ in range(3):
                        execute_command("M 5 1")
                        execute_command("P")
                    execute_command("M 15 1")

                    cameras = get_available_cameras()

                    if not cameras:
                        print("Žádná kamera nenalezena!")
                    else:
                        print("Dostupné kamery:")
                        show_available_cameras(cameras)

                        selected_camera_index = int(input("Vyber číslo kamery: ")) - 1
                        if selected_camera_index < 0 or selected_camera_index >= len(cameras):
                            print("Neplatná volba kamery!")
                        else:
                            output_filename = "webcam_photo"  # -------------------------------------------------
                            photo = capture_and_save_webcam_photo(selected_camera_index,
                                                                  f"output_filename_{count:03d}.jpg")

                            if photo is not None:
                                captured_photos.append(photo)
                                try:
                                    command_to_send = "M -5 1"  # ------------------------------------------------
                                    send_command(command_to_send)  # photo.tobytes()
                                except ValueError as ve:
                                    print("Chyba odeslání příkazu:", ve)
                                    pass
                            else:
                                pass

                    ser.close()
            except ValueError:
                pass
