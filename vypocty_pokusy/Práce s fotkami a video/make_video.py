import cv2
import time
# import numpy as np

# Vytvoření objektu pro přístup k webové kameře
cap = cv2.VideoCapture(0, cv2.CAP_ANY)  # CAP_ANY // CAP_MSMF

# 3840, 2160  * 4 // 1280, 720
cam_width = 3840  # int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = 2160  # int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cam_fps = 120  # cap.get(cv2.CAP_PROP_FPS)
# print(cap.get(cv2.CAP_PROP_FPS))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
cap.set(cv2.CAP_PROP_FPS, cam_fps)

# Nastavení parametrů pro nahrávání videa
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Kodek


# frame_width = int(cap.get(3))  # Šířka snímků z kamery
# frame_height = int(cap.get(4))  # Výška snímků z kamery
# _w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# _h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# _fps = cap.get(cv2.CAP_PROP_FPS)
# _b = cap.get(cv2.CAP_PROP_BRIGHTNESS)
# _c = cap.get(cv2.CAP_PROP_CONTRAST)
# _s = cap.get(cv2.CAP_PROP_SATURATION)
# _e = cap.get(cv2.CAP_PROP_EXPOSURE)
# _f = cap.get(cv2.CAP_PROP_FOURCC)


# out = cv2.VideoWriter('output.avi', fourcc, cam_fps, (cam_width, cam_height))

# Kontrola, zda je webová kamera správně otevřena
if not cap.isOpened():
    print("Chyba: Webová kamera není dostupná.")
    exit()
else:
    # count = 1
    counter = 1
    outer_counter = 0
    photos = []
    start_time = time.time()
    cv2.namedWindow('Web Camera', cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()  # Načtení snímku z kamery
        # frame = frame[:, 1300:frame.shape[1]-1300]
        """if not ret:
            break  # Konec nahrávání, pokud není dostupný žádný snímek"""

        # Zde můžete provádět operace s každým snímkem, pokud je to nutné

        # Zápis snímku do výstupního videa
        # out.write(frame)

        # cv2.imwrite(f"photos/photo_cam_{count: 03d}.jpg", frame)
        photos.append(frame)
        counter += 1
        # count += 1

        # Zobrazení snímku z kamery (lze zakomentovat, pokud nechcete zobrazovat náhled)
        cv2.imshow('Web Camera', frame)

        if counter == 200:
            [cv2.imwrite(f"photos/photo_cam_{num + 1: 03d}.jpg", frame) for num, frame in enumerate(photos)]
            photos, counter = [], 1
            outer_counter += 1

        if start_time + 2 <= time.time():
            break
        # Ukončení nahrávání a aplikace stiskem klávesy 'q'
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Uvolnění prostředků a zavření okna
[cv2.imwrite(f"photos/photo_cam_{num + 1 + (outer_counter * 200): 03d}.jpg", frame) for num, frame in enumerate(photos)]
# cv2.imwrite(f"photo_cam_.jpg", frame)
cap.release()
# out.release()
cv2.destroyAllWindows()
