import os
from shutil import copy2, move
import cv2
from pyzbar.pyzbar import decode as qr_detect

#############################################################
#############################################################
#############################################################

# Název souboru s fotografií obsahující QR kód
input_path = r"C:\Users\matej\Desktop\mereni\Fotky"
output_path = r"C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos"

function = move  # #### Druh manipulace: copy2 - kopírování // move - přesun

#############################################################
#############################################################
#############################################################

if not os.path.isdir(input_path):
    exit()

photos = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))
          and f.lower().endswith((".jpg", ".jpeg", ".JPG", ".png"))]

for photo in photos:
    photo_path = os.path.join(input_path, photo)
    # Načtení QR kódů z obrázku
    decoded_objects = qr_detect(cv2.cvtColor(cv2.imread(photo_path, 1), cv2.COLOR_BGR2GRAY)[2800:, 1700:-1700])

    """if len(decoded_objects) != 1:
        if len(decoded_objects) > 1:
            print(f"WARRNING:\n\tFotografie obsahuje více QR-kódů:  {photo}")
            continue
        elif len(decoded_objects) == 0:
            continue
        else:
            continue"""

    if not decoded_objects:
        print(f"WARRNING:\n\tFotografie neobsahuje QR-kódy:  {photo}")
        continue

    name = None
    # Procházení nalezených QR kódů
    for obj in decoded_objects:
        name = obj.data.decode('utf-8')
        if name.startswith(".*CP*.") or ".*CP*." in name:
            name = None
            continue

        path = os.path.join(output_path, name)
        if not os.path.isdir(path):
            os.makedirs(path)

        path = os.path.join(path, 'original')
        if not os.path.isdir(path):
            os.makedirs(path)

        # Kopírování souboru a zachování metadata
        function(photo_path, os.path.join(path, photo))

    print(f"\nFotografie {name} přesunuty")
