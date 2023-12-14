import os
import qrcode
import numpy as np
from PIL import Image
# import json
# import yaml

output_folder = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\obr'

for i in range(1, 7):
    # Text nebo data, která chcete zakódovat do QR kódu
    data = f"H02_{i:02d}-I_10s_n"   # f"F04_{i:03d}"  # f"T02_{i:02d}-I_1s"  # "H02_{i:02d}_10s"  # ".*CP*._N#3"
    name = "qr_Hex_test_" + data  # "Measurement_" + data  # "calibration_point05" # "qr_Friction_test_"
    short_name = data

    use_logo = False
    make_frame = True
    add_short_name = True

    square_size = 10

    # Vytvoření QR kódu s určenou úrovní korekce chyb
    error_correction = qrcode.constants.ERROR_CORRECT_M

    """file_type = "YAML"
    
    matrix = np.random.rand(3, 3)
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

    if make_frame or add_short_name:
        blank_space = 6

        if add_short_name:
            addition = blank_space
        else:
            addition = 0

        img = np.array(img, dtype=np.uint8) * 255
        h, w = img.shape
        blank_space = round(blank_space / 2) * 2
        framed_img = np.ones((h + square_size * (blank_space + addition), w + square_size * blank_space),
                             dtype=np.uint8) * 255
        framed_img[:, :square_size] = framed_img[:, -square_size:] = framed_img[:square_size, :] = framed_img[
                                                                                                   -square_size:, :] = 0
        (framed_img[
         square_size * blank_space // 2:-square_size * (blank_space // 2 + addition),
         square_size * blank_space // 2:-square_size * blank_space // 2]) = img

        if add_short_name:
            import cv2

            font_size = square_size // 2
            while True:
                (text_width, text_height), _ = cv2.getTextSize(short_name, cv2.FONT_HERSHEY_DUPLEX, font_size,
                                                               square_size // 5)

                if (text_height > (square_size * (blank_space // 2 + addition) / 2) or
                        text_width > (framed_img.shape[1] - (6 * square_size))):
                    font_size -= 0.25
                    if font_size < 0.1:
                        font_size = 0.1
                        break
                else:
                    break

            cv2.putText(framed_img, short_name,
                        ((framed_img.shape[1] - text_width) // 2, np.int32(h + (square_size * blank_space) * 1.6)),
                        cv2.FONT_HERSHEY_DUPLEX, font_size, (0, 0, 0), square_size // 5, cv2.LINE_AA)

            """from PIL import ImageDraw, ImageFont
    
            position_x = img.shape[1] // 2  # Zde zadejte x-ovou pozici středu textu
            position_y = np.int32(h + blank_space * 10)  # Zde zadejte y-ovou pozici středu textu
            max_width = np.int32(w + blank_space * 2 - 4 * square_size)  # Zde zadejte maximální šířku okna
    
            img = Image.fromarray(framed_img)
            draw = ImageDraw.Draw(img)
    
            # Nakreslete text na obrázek
            draw.text((position_x // 2, position_y), short_name, font=ImageFont.load_default())
        else:
            img = Image.fromarray(framed_img)"""
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

    img_name = f"{name}.png"
    if os.path.isdir(output_folder):
        img_name = os.path.join(output_folder, img_name)

    # Uložení obrázku QR kódu s logem do souboru
    img.save(img_name)
