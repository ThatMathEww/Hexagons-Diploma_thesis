import os
from shutil import copy2, move
import cv2
from pyzbar.pyzbar import decode as qr_detect

# Název souboru s fotografií obsahující QR kód
input_path = r""
output_path = r""

function = copy2  # #### Druh manipulace: copy2 - kopírování // move - přesun

if not os.path.isdir(input_path):
    exit()

photos = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))
          and f.lower().endswith((".jpg", ".jpeg", ".JPG", ".png"))]

for photo in photos[-8:]:
    photo_path = os.path.join(input_path, photo)
    # Načtení QR kódů z obrázku
    decoded_objects = qr_detect(cv2.cvtColor(cv2.imread(photo_path, 1), cv2.COLOR_BGR2RGB))

    if len(decoded_objects) != 1:
        if len(decoded_objects) > 1:
            print(f"WARRNING:\n\tFotografie obsahuje více QR-kódů:  {photo}")
            continue
        elif len(decoded_objects) == 0:
            continue
        else:
            continue

    # Procházení nalezených QR kódů
    for obj in decoded_objects:
        name = obj.data.decode('utf-8')

        path = os.path.join(output_path, name)

        if not os.path.isdir(path):
            os.makedirs(path)

        # Kopírování souboru a zachování metadata
        function(photo_path, os.path.join(path, photo))

print("\nFotografie přesunuty")
