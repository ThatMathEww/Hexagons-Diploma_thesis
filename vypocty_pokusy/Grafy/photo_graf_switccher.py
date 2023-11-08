from matplotlib.widgets import Slider, Button
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2
import matplotlib.image as mpimg


def load_photos(folder, color=0, file_list=None, img_types=(".jpg", ".jpeg", ".JPG", ".png", ".tiff", ".tif")):
    if any(item not in (".jpg", ".jpeg", ".jpe", ".JPG", ".jp2", ".png", ".bmp", ".dib", ".webp",
                        ".avif", ".pbm", ".pgm", ".ppm", ".pxm", ".pnm", ".pfm", ".sr", ".ras",
                        ".tiff", ".tif", ".exr", ".hdr", ".pic") for item in img_types):
        exit(f"\n\033[31;1;21mERROR\033[0m"
             f"\n\tNepodporovaný typ fotografie.\n\t\033[41;30m➤ Ukončení programu.\033[0m")
    else:
        folders = [f for f in
                   (os.listdir(folder) if file_list is None else file_list)
                   if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(img_types)][1:35]
        first_type = os.path.splitext(folders[0])[1]
        if all(os.path.splitext(f)[1] == first_type for f in folders):  # kontrola jestli josu všechny fotky stejné
            return [mpimg.imread(os.path.join(folder, f)) for f in folders]
        else:
            exit(f"\n\033[31;1;21mERROR\033[0m"
                 f"\n\tNačtené fotografie jsou různého typu.\n\t\033[41;30m➤ Ukončení programu.\033[0m")


def release(event):
    if event.button == 1:  # Tlačítko myši 1 odpovídá uvolnění slideru
        print(slider.val)


def move_left(val):
    if val.button == 1:
        val = int(max(slider.val - 1, 1))
        if slider.val != val:
            slider.set_val(val)


def move_right(val):
    if val.button == 1:
        val = int(min(slider.val + 1, max_photos))
        if slider.val != val:
            slider.set_val(val)


def get_slider(val):
    # axs.imshow(photos[val - 1])
    img.set_data(photos[val - 1])
    fig.canvas.draw()
    # value = slider.val


def reset(val):
    if val.button == 1:
        slider.reset()


if __name__ == '__main__':

    # Nastavení cesty k složce s obrázky
    image_folder = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos\__test_1\modified'

    if not os.path.isdir(image_folder):
        print(f"Načtená hlavní složka neexituje: [{image_folder}]")
        while not os.path.isdir(image_folder):
            image_folder = input("\tZadejte novou cestu ke složce: ")

    # set up a plt
    photos = load_photos(image_folder)
    max_photos = len(photos)

    ratio = (photos[0].shape[1] / photos[0].shape[0])

    fig_size = 5.5

    fig, axs = plt.subplots(figsize=(fig_size * ratio + 0.5, fig_size), sharex=True, num="Figure photo slider")
    box_width = 0.2

    print("Fotky načteny.")

    fig.subplots_adjust(right=1 - box_width - 0.03, left=0.1, top=0.95, bottom=0.1)
    ax = plt.axes((fig.subplotpars.left, fig.subplotpars.bottom + (1 - fig.subplotpars.top) + 0.01,
                   fig.subplotpars.right - fig.subplotpars.left, fig.subplotpars.top - fig.subplotpars.bottom))
    ax_b = plt.axes(((fig.subplotpars.right + 0.03), fig.subplotpars.top - 0.05, box_width * 0.6, 0.05))
    ax_s = plt.axes(((fig.subplotpars.right + 0.03), fig.subplotpars.top - 0.175, box_width * 0.6, 0.075))
    ax_sb_l = plt.axes(((fig.subplotpars.right + 0.03), fig.subplotpars.top - 0.225, box_width * 0.25, 0.05))
    ax_sb_r = plt.axes(((fig.subplotpars.right + 0.03 + (box_width * 0.6 - box_width * 0.25)),
                        fig.subplotpars.top - 0.225, box_width * 0.25, 0.05))

    img = ax.imshow(photos[0], cmap='gray')
    ax.autoscale(True)
    axs.axis('off')

    # Vytvoření slideru
    slider = Slider(ax=ax_s, label='', valmin=1, valmax=max_photos, valinit=0, valstep=1, dragging=False)
    button = Button(ax_b, 'Reset', color='royalblue', hovercolor='skyblue', useblit=True)
    btn_l = Button(ax_sb_l, '<', color="orange", hovercolor='wheat', useblit=True)
    btn_r = Button(ax_sb_r, '>', color="orange", hovercolor='wheat', useblit=True)
    # '#1E90FF'
    slider.on_changed(get_slider)
    button.on_clicked(reset)
    btn_l.on_clicked(move_left)
    btn_r.on_clicked(move_right)

    # fig.canvas.mpl_connect('button_release_event', release)

    plt.show()
