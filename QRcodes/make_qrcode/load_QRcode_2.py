import cv2
from pyzbar.pyzbar import decode
import numpy as np
import json
import yaml
import re

# Název souboru s fotografií obsahující QR kód
image_path = "qr_code.png"

# Načtení obrázku
image = cv2.imread(image_path)

# Převod obrázku do černobílého formátu (grayscale)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Načtení QR kódů z obrázku
decoded_objects = decode(gray)

numpy_array = False

# Procházení nalezených QR kódů
for obj in decoded_objects:
    data = obj.data.decode('utf-8')
    if "JSON" in data:
        data = json.loads(data)
    elif "YAML" in data:
        data = yaml.safe_load(data)
        data['data']['numpy_data'] = {key: np.frombuffer(value).reshape(data['data']['numpy_data']['width'],
                                                                        data['data']['numpy_data']['height'])
        if isinstance(value, bytes) else value for key, value in data['data']['numpy_data'].items()}

        """data['numpy_data'] = {key: np.fromstring(value, sep=';').reshape(data['numpy_data']['width'],
                                                                          data['numpy_data']['height'])
        if isinstance(value, str) else value for key, value in data['numpy_data'].items()}"""
    elif numpy_array:
        a, b = data.split("'")[1:]
        b = b[2:]
        b, c = b.split("), (")
        c, d = c[:-2].split(", ")

        match = re.search(r'\[([^]]+)\]', b)

        if match:
            # Získáme obsah uvnitř závorek
            array_content = match.group(1)

            # Použijeme numpy k vytvoření pole
            array = np.array(eval(array_content))

        a = a
        # b = np.array(eval(b))
        c = float(c)
        d = float(d)

        # a, b = data.split("), (")
        # a, c = a.split(" array")
        # a = a.split("'")[0]
    else:
        pass


    print(f"QR Code data: {data}")

    # Vykreslení obdélníku kolem QR kódu
    points = obj.polygon
    if len(points) > 3:
        hull = cv2.convexHull(np.array([point for point in points], dtype=np.int32))
        cv2.polylines(image, [hull], True, (0, 255, 0), 2)

# Zobrazení obrázku s vyznačenými QR kódy
cv2.namedWindow("QR Codes", cv2.WINDOW_NORMAL)
cv2.resizeWindow("QR Codes", image.shape[1]//2, image.shape[0]//2)
cv2.imshow("QR Codes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
