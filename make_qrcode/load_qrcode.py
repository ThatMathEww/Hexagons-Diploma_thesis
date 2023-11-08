import cv2

# Načtěte obrázek
image = cv2.imread('Snimek obrazovky 2023-09-22 180952.png')

# Převeďte obrázek na černobílý
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Inicializujte detektor QR kódů
qr_decoder = cv2.QRCodeDetector()

# Najděte QR kódy v obraze
success, decoded_info, points, _ = qr_decoder.detectAndDecodeMulti(gray)

# Pokud byl QR kód nalezen
if success:
    for i in range(len(decoded_info)):
        print(f"QR kód {i + 1}: {decoded_info[i]}")
        # Nakreslete obdelník kolem QR kódu
        rect_points = points[i].astype(int)
        cv2.polylines(image, [rect_points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Zobrazte obrázek s označenými QR kódy
    cv2.imshow('Detekované QR kódy', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("QR kódy nebyly nalezeny.")
