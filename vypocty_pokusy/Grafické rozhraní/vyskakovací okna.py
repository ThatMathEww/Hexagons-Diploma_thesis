import cv2
import numpy as np
import tkinter as tk

style = "tk"

if style == "cv2":
    # Vytvoření prázdného černého obrazu
    image = np.zeros((400, 400, 3), dtype=np.uint8)

    # Nastavení textu
    text = "Vyskakovaci okno"
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 1
    color = (255, 255, 255)  # Bílá barva
    thickness = 2

    # Získání velikosti textu
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Výpočet pozice textu
    x = int((image.shape[1] - text_width) / 2)
    y = int((image.shape[0] + text_height) / 2)

    # Vložení textu do obrazu
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness)

    # Vytvoření a zobrazení vyskakovacího okna
    cv2.namedWindow("Vyskakovaci okno", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vyskakovaci okno", 200, 200)
    cv2.imshow("Vyskakovaci okno", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif style == "tk":
    # Vytvoření instance okna
    window = tk.Tk()

    # Funkce pro zobrazení vyskakovacího okna s textem
    def show_popup():
        popup = tk.Toplevel(window)
        popup.title("Vyskakovací okno")

        # Vytvoření popisku s textem
        label = tk.Label(popup, text="Vyskakovací okno s textem")
        label.pack(padx=20, pady=20)

        # Tlačítko pro zavření okna
        button = tk.Button(popup, text="Zavřít", command=popup.destroy)
        button.pack(pady=10)


    # Tlačítko pro zobrazení vyskakovacího okna
    button_show_popup = tk.Button(window, text="Zobrazit vyskakovací okno", command=show_popup)
    button_show_popup.pack(padx=20, pady=20)

    # Spuštění hlavní smyčky pro zobrazení okna
    window.mainloop()

else:
    print("Žádná ze zvolených možností")
