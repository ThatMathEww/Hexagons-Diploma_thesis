import cv2
import os
import numpy as np


def create_video_from_images(image_folder, output_video_path, fps=24, frame_width=1920, frame_height=1080,
                             video_length=None, codec='none'):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith((".jpg", ".jpeg", ".png", ".JPG"))])

    if video_length is not None:
        fps = len(images) / video_length

    # Získáme rozměry prvního obrázku (předpokládáme, že všechny mají stejné rozměry)
    image_height, image_width = cv2.imread(os.path.join(image_folder, images[0]), 0).shape[:2]

    if output_video_path.lower().endswith(".mp4"):
        if codec == 'none':
            codec = 'H265'
        elif codec not in ('H264', 'X264', 'H265', 'VP90', 'mp4v', 'DIVX', 'XVID', 'FMP4', 'avc1'):
            print("Nepodporovaný codec pro mp4.")
            return
    elif output_video_path.lower().endswith(".avi"):
        if codec == 'none':
            codec = 'DIVX'
        elif codec not in ('DIVX', 'XVID', 'MJPG', 'WMV1', 'WMV2', 'mpg1', 'I420', 'IYUV', 'H264'):
            print("Nepodporovaný codec pro mp4.")
            return
    else:
        print("Nepodporovaný nebo nezadaný formát.")
        return

    # Nastavení kodeku a vytvoření objektu pro video zápis
    #                                   VideoWriter objekt s nekomprimovaným kodekem (VYUY)
    # fourcc = cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V') # - nefunkční
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=True)

    # Výpočet poměru šířky a výšky načteného obrázku
    image_ratio = image_width / image_height

    # Výpočet poměru šířky a výšky požadovaného pole
    desired_ratio = frame_width / frame_height

    # Rozhodnutí o změně velikosti obrázku
    if image_ratio > desired_ratio:
        new_width = frame_width
        new_height = int(frame_width / image_ratio)
    else:
        new_width = int(frame_height * image_ratio)
        new_height = frame_height

    # Umístění obrázku do pole, aby zabíralo maximální plochu
    x_offset = (frame_width - new_width) // 2
    y_offset = (frame_height - new_height) // 2

    for image_name in images:
        image = cv2.imread(os.path.join(image_folder, image_name), 1)

        # Změna velikosti obrázku
        resized_image = cv2.resize(image, (new_width, new_height))

        # Vytvoření prázdného pole
        output_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # Umístěte fotografii na střed obrazu
        output_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
        out.write(output_image)

    # Uzavření objektu pro video zápis
    out.release()

    print("hotovo")


def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    close = False

    while True:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Video', 1920 // 2, 1080 // 2)
            cv2.imshow('Video', frame)

            if cv2.waitKey(25) & 0xFF == 27:  # Klávesa "Esc" pro ukončení
                close = True
                break

        # Posuneme časový ukazatel zpět na začátek videa
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if close:  # Klávesa "Esc" pro ukončení
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "photos"
    output_path = "output_video.mp4"

    create_video_from_images(image_path, output_path, video_length=60, codec='mp4v',
                             frame_width=3840, frame_height=2160)

    if os.path.exists(output_path):
        play_video(output_path)
    else:
        create_video_from_images(image_path, output_path)
        play_video(output_path)

    """print("\nChcete video smazat?")
    while True:
        ans = input("\tZadejte Y / N: ")
        if ans == "Y":
            os.remove(output_video_path)
            break
        elif ans == "N":
            break
        else:
            print("Zadejte platnou odpověď.")"""
