import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


##########################################
# Hledání pomocí "matchTemplate"
# mezi dvěma fotkami
# Oblast pixelů
##########################################


def mark_points_on_canvas(num):
    marked_points = []
    h, w = int(0.0017 * height), int(0.0017 * width)

    # Vytvoření figure a osy v matplotlib
    fig, ax = plt.subplots(figsize=(w, h))

    # Funkce pro označení bodů
    def mark_points(event):
        if len(marked_points) == num:
            plt.close()
        elif event.key == 'shift':
            x = event.xdata
            y = event.ydata
            ax.plt(x, y, 'ro')
            fig.canvas.draw()
            marked_points.append((x, y))
        elif event.key == 'enter':
            plt.close()

    # Připojení události ke funkci mark_points
    cid = fig.canvas.mpl_connect('key_press_event', mark_points)

    # Zobrazení obrázku v matplotlib
    ax.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

    # Zobrazení textu v levém horním rohu
    text = "Označte 2 body pomocí Shift\nUkončení pomocí Enter"
    ax.text(0.01, 0.99, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

    # Zobrazení obrázku s označenými body
    plt.tight_layout()
    plt.show()

    return marked_points


def def_area(marked_points=None):
    # Seznam pro ukládání bodů
    if marked_points is None:
        marked_points = []

    marked_points = mark_points_on_canvas(4)

    marked_points = np.array(marked_points)

    # Výpis označených bodů
    print("Označené body:")
    for i, point in enumerate(marked_points):
        print("Bod {}: x = {:.2f}, y = {:.2f}".format(i + 1, point[0], point[1]))

    print("Press Enter or Space to close the window.\n")

    cv2.rectangle(image1, (np.int32(marked_points[0, :])), (np.int32(marked_points[1, :])), (0, 0, 255), 5)
    cv2.namedWindow('Marked area', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Marked area', int(0.25 * width), int(0.25 * height))
    cv2.imshow('Marked area', image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Souhlasíte se zvolenou oblastí?")

    while True:
        answer = input("\tZadejte Y nebo N: ")

        if answer == "Y":
            print("Byla zvolena odpověď Y.\n")
            break

        elif answer == "N":
            print("\nZadejte nové souřanice:\n")
            marked_points = []
            marked_points = def_area(marked_points)
            break  # ukončení smyčky po správné odpovědi

        else:
            print("Neplatná odpověď. Zadejte pouze Y nebo N.\n")
            # smyčka se opakuje, dokud uživatel nezadá platný vstup

    return marked_points


# Načtení první a druhé fotografie
image1 = cv2.imread('photos/IMG_0385.JPG')
image2 = cv2.imread('photos/IMG_0417.JPG')

# Převod obou fotografií na šedotónový formát
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Naleznete rozměry první fotografie
height, width = gray1.shape

area_ans = True
if area_ans:
    points = def_area()
    x1, y1 = np.int32(points[0, :])
    x2, y2 = np.int32(points[1, :])
    del points
else:
    # x1, y1 = 2650, 400
    # x2, y2 = 3450, 650

    x1, y1 = 3080, 505
    x2, y2 = 3150, 575

numbers = [(x1, y1), (x2, y2)]
sorted_numbers = sorted(numbers, key=lambda num: num[0])
x1, y1 = sorted_numbers[0]
x2, y2 = sorted_numbers[1]
del numbers, sorted_numbers

template = gray1[y1:y2, x1:x2]

cv2.namedWindow('Original area', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original area', x2 - x1, y2 - y1)
cv2.imshow('Original area', gray1[y1:y2, x1:x2])

# Porovnejte šablonu s druhou fotografií pomocí metody šablony
result = cv2.matchTemplate(gray2, template, cv2.TM_CCOEFF_NORMED)

# Práh pro výběr dobrých shod
threshold = 1

# Cyklus pro nalezení oblasti snižováním hranice přesnosti
while True:
    # Naleznete polohy, kde je shoda vyšší než prah
    locations = np.where(result >= threshold)

    found_coordinates = np.array(locations).T
    found_coordinates[:, [0, 1]] = found_coordinates[:, [1, 0]]
    if len(found_coordinates) != 0:
        break
    else:
        threshold = threshold - 0.01
        if threshold < 0.1:
            print("Oblast nenalezena.")
            sys.exit()

print("\nPřesnost:", threshold * 100, "%\n")


def do_scale():
    dist = mark_points_on_canvas(2)
    d1_x, d1_y = dist[0]
    d2_x, d2_y = dist[1]
    print("Zadej vzdáolenost v bodů [mm]")
    d1 = 0
    while True:
        d2 = input("\tVzádlenost: ").replace(",", ".")
        try:
            d2 = abs(float(d2))  # pokus o převod na číslo
            break
        except ValueError:
            print("Vzdálenost ve špatném formátu, zadajete ji znovu.")
            pass
    return d1_x, d1_y, d2_x, d2_y, d1, d2


making_scale = False
if making_scale:
    print("Chce zadat měřítko?")
    while True:
        make_scale = input("\tZadejte Y nebo N: ")
        if make_scale == "Y":
            dist1_x, dist1_y, dist2_x, dist2_y, dist1, dist2 = do_scale()
            while True:
                print("\nChcete proces zopakovat?")
                rerun = input("\tZadejte Y nebo N: ")
                if rerun == "Y":
                    dist1_x, dist1_y, dist2_x, dist2_y, dist1, dist2 = do_scale()
                elif rerun == "N":
                    break
                else:
                    print("\nNeplatná odpověď. Zadejte pouze Y nebo N.")
            del rerun
            break

        elif make_scale == "N":
            break
        else:
            print("\nNeplatná odpověď. Zadejte pouze Y nebo N.")
else:
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

print("\nScale:", round((dist_mm / dist_px), 3))

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

cv2.namedWindow('Found image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Found image', int(0.25 * width), int(0.25 * height))

print("\n\tPress Enter or Space to close the windows.")

# Zobrazení výsledku
cv2.imshow('Found image', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
