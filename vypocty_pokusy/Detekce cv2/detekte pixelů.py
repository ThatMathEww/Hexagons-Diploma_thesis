import cv2
import numpy as np


def show_area(x1, y1, x2, y2):
    ###################################################################
    # zobrazení černé plochy a výřezu

    # Vytvoření masky (maskovací matice) pro definování oblasti
    mask = np.zeros(image1.shape[:2], dtype=np.uint8)
    region_of_interest = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]],
                                  dtype=np.int32)  # Definice oblasti (čtyřúhelník)
    cv2.fillPoly(mask, [region_of_interest], 255)  # Nastavení hodnoty 255 v oblasti definované polygonem

    # Aplikace masky na první fotografii
    masked_image = cv2.bitwise_and(image1, image1, mask=mask)

    height, width, _ = image1.shape
    # Zobrazení výsledku
    cv2.namedWindow('Oblast definovana maskou', cv2.WINDOW_NORMAL)
    cv2.imshow('Oblast definovana maskou', masked_image)
    cv2.resizeWindow('Oblast definovana maskou', int(0.25 * width), int(0.25 * height))
    ###################################################################

    # zobrazení obdélníku
    height, width, _ = image1.shape
    cv2.rectangle(image1, (x1, y1), (x2, y2), (255, 255, 255), 7)
    cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Preview', int(0.25 * width), int(0.25 * height))
    cv2.imshow('Preview', image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    while True:
        print("Souhlasíte se zvolenou oblastí?")
        answer = input("Zadejte Y nebo N: ")

        if answer == "Y":
            print("Byla zvolena odpověď Y.")
            break

        elif answer == "N":
            print("Byla zvolena odpověď N.")
            print("\nPůvodní souřanice:")
            print("\t\tx1:", x1, "y1:", y1)
            print("\t\tx2:", x2, "y2:", y2)
            print("\nZadejte nové souřanice:")

            print("Zadejte čtyři čísla souřadnic 'x1 y1 x2 y2': ")
            x1 = float(input("\tZadejte x1: "))
            y1 = float(input("\tZadejte y1: "))
            x2 = float(input("\tZadejte x2: "))
            y2 = float(input("\tZadejte y2: "))
            show_area(x1, y1, x2, y2)
            break  # ukončení smyčky po správné odpovědi4

        else:
            print("Neplatná odpověď. Zadejte pouze Y nebo N.")
            # smyčka se opakuje, dokud uživatel nezadá platný vstup

    return x1, y1, x2, y2


mask = None

x1, y1 = 2650, 400
x2, y2 = 3450, 650

# Načtení první a druhé fotografie
image1 = cv2.imread('photos/IMG_0385.JPG')
image2 = cv2.imread('photos/IMG_0417.JPG')

# Převod obou fotografií na šedotónový formát
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

area_ans = True
if area_ans:
    show_area(x1, y1, x2, y2)

# Naleznete rozměry první fotografie
height, width = gray1.shape

template = gray1[y1:y2, x1:x2]

cv2.namedWindow('Original area', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original area', x2 - x1, y2 - y1)
cv2.imshow('Original area', gray1[y1:y2, x1:x2])

# Porovnejte šablonu s druhou fotografií pomocí metody šablony
result = cv2.matchTemplate(gray2, template, cv2.TM_CCOEFF_NORMED)

# Nastavte práh pro výběr dobrých shod
threshold = 0.5

# Naleznete polohy, kde je shoda vyšší než prah
locations = np.where(result >= threshold)

found_coordinates = np.array(locations).T
found_coordinates[:, [0, 1]] = found_coordinates[:, [1, 0]]

"""
# Získání souřadnic první nalezené oblasti
if len(found_coordinates) > 0:
    x, y, w, h = found_coordinates[0]
    print("Nalezená oblast: x = {}, y = {}, šířka = {}, výška = {}".format(x, y, w, h))
    cv2.rectangle(image2, (x, y), (w, h), (0, 255, 0), 2)
else:
    print("Nenalezena žádná oblast.")
"""

"""
x_new = []
y_new = []
# Vypište souřadnice nalezených oblastí
for i in range(len(found_coordinates)):
    x, y = found_coordinates[i]
    cv2.line(image2, (x1, y1), (x1, y + y2 - y1), (255, 0, 0), 5)
    cv2.line(image2, (x2, y1), (x2, y + y2 - y1), (255, 0, 0), 5)
    cv2.rectangle(image2, (x1, y1), (x2, y2), (255, 255, 255), 7)
    cv2.rectangle(image2, (x, y), (x + x2 - x1, y + y2 - y1), (0, 255, 0), 5)
    cv2.circle(image2, (x, y), 5, (0, 0, 255), 15)

    x_new.append(x)
    y_new.append(y)
# print("Nalezená oblast: x = {}, y = {}".format(x, y))
"""

dist1 = 100
dist2 = 220
dist1_x, dist1_y = 1521, 1457
dist2_x, dist2_y = 4338, 1468

x_f_mean = np.mean(found_coordinates[:, 0])
y_f_mean = np.mean(found_coordinates[:, 1])

dif_x = abs(x1 - x_f_mean)
dif_y = abs(y1 - y_f_mean)

dist_mm = dist2 - dist1
dist_px = np.sqrt((dist2_x - dist1_x) ** 2 + (dist2_y - dist1_y) ** 2)

final_dist_x_mm = dist_mm / dist_px * dif_x
final_dist_y_mm = dist_mm / dist_px * dif_y

print("\nVelikost fotografie:\t\t", height, "x", width, "px")
print("Průměrný výškový rozdíl:\t\t", round(dif_y, 5), "px")
print("Průměrný vodorovný rozdíl:\t\t", round(dif_x, 5), "px")

print("\nPrůměrný výškový rozdíl:\t\t", round(final_dist_y_mm, 5), "mm")
print("Průměrný vodorovný rozdíl:\t\t", round(final_dist_x_mm, 5), "mm")

cv2.line(image2, (x1, y1), (x1, int(y_f_mean + y2 - y1)), (255, 0, 0), 5)
cv2.line(image2, (x2, y1), (x2, int(y_f_mean + y2 - y1)), (255, 0, 0), 5)
cv2.rectangle(image2, (x1, y1), (x2, y2), (255, 255, 255), 7)
cv2.rectangle(image2, (int(x_f_mean), int(y_f_mean)), (int(x_f_mean + x2 - x1), int(y_f_mean + y2 - y1)), (0, 255, 0),
              5)
cv2.circle(image2, (int(x_f_mean), int(y_f_mean)), 5, (0, 0, 255), 15)

cv2.namedWindow('Found area', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Found area', x2 - x1, y2 - y1)
cv2.imshow('Found area', gray2[int(y_f_mean):int(y_f_mean + y2 - y1), int(x_f_mean):int(x_f_mean + x2 - x1)])

cv2.namedWindow('Hledání oblasti', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hledání oblasti', int(0.25 * width), int(0.25 * height))
# Zobrazení výsledku
cv2.imshow('Hledání oblasti', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
