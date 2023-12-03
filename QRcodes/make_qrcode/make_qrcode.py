import qrcode
from PIL import Image
import json
import yaml
import numpy as np
import cv2
import os

# Text nebo data, která chcete zakódovat do QR kódu
# data = ".*CP*._N#1"  # "H02_01_12s"  # ".*CP*._N#3"
# name = "calibration_point01_L"  # "Measurement_" + data  # "calibration_point05"

make_frame = True
square_size = 10

output_folder = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\Hexagons-Diploma_thesis\QRcodes'

file_type = "YAML"

"""matrix = np.random.rand(3, 3)
binary_data = matrix.tobytes()
# str_data = np.array2string(matrix, separator=';')
data = {
    'type': file_type,
    'name': name,
    'data':
        {
            'list_data': [1, 2, 3],
            'dict_data': {'name': name, 'age': 30},
            'num': 45,
            'numpy_data': {'data': binary_data, 'width': matrix.shape[0], 'height': matrix.shape[1]},
        }
}"""
"""data = {
    'type': file_type,
    'name': name,
    'data': None
}

if file_type == 'YAML':
    # Uložení dat do YAML formátu
    data = yaml.dump(data)
elif file_type == 'JSON':
    # Uložení dat do JSON formátu
    data = json.dumps(data)
else:
    exit()"""

"""path = r''
photo = cv2.imread(path, 0).tolist()
name = 'IMG_0385.JPG'
additional_information = (os.path.getmtime(path), os.path.getctime(path))

data = ((name, photo), (os.path.getmtime(path), os.path.getctime(path)))"""

use_logo = False

# Vytvoření QR kódu s určenou úrovní korekce chyb
error_correction = qrcode.constants.ERROR_CORRECT_Q

for i in range(3, 5):
    data = f".*CP*._N#{i}"
    name = f"calibration_point{i:02d}"

    # Vytvoření QR kódu
    qr = qrcode.QRCode(
        version=1,  # Velikost QR kódu (1-40, čím vyšší číslo, tím větší QR kód)
        error_correction=error_correction,  # Oprava chyb: L (Low), M (Medium), Q (Quartile), H (High)
        box_size=square_size,  # Velikost jednoho bloku QR kódu
        border=0,  # Šířka okraje QR kódu
    )

    qr.add_data(data)
    qr.make(fit=True)

    # Vytvoření obrázku QR kódu
    img = qr.make_image(fill_color="black", back_color="white")

    if make_frame:
        img = np.array(img)
        h, w = img.shape
        framed_img = np.ones((h + square_size * 6, w + square_size * 6), dtype=bool)
        framed_img[:, :square_size] = framed_img[:, -square_size:] = framed_img[:square_size,
                                                                     :] = framed_img[-square_size:, :] = False
        framed_img[square_size * 3:-square_size * 3, square_size * 3:-square_size * 3] = img
        img = Image.fromarray(framed_img)

    if error_correction == qrcode.constants.ERROR_CORRECT_H and use_logo:
        try:
            # Otevření logo obrázku
            logo = Image.open("logo.png")

            # Výpočet pozice, kam umístit logo (zde upravte podle svých potřeb)
            logo_size = (img.size[0] // 4, img.size[1] // 4)
            logo_position = ((img.size[0] - logo_size[0]) // 2, (img.size[1] - logo_size[1]) // 2)

            # Přidání loga do QR kódu
            img.paste(logo.resize(logo_size), logo_position)
        except FileNotFoundError:
            pass

    # img.show()

    img_name = f"qr_code_{name}.png"
    if os.path.isdir(output_folder):
        img_name = os.path.join(output_folder, img_name)

    # Uložení obrázku QR kódu s logem do souboru
    img.save(img_name)
