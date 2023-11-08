import numpy as np
from matplotlib import pyplot as plt
import matplotlib.widgets
import matplotlib.patches
import mpl_toolkits.axes_grid1
import os
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
                   if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(img_types)]
        first_type = os.path.splitext(folders[0])[1]
        if all(os.path.splitext(f)[1] == first_type for f in folders):  # kontrola jestli josu všechny fotky stejné
            return [mpimg.imread(os.path.join(folder, f)) for f in folders]
        else:
            exit(f"\n\033[31;1;21mERROR\033[0m"
                 f"\n\tNačtené fotografie jsou různého typu.\n\t\033[41;30m➤ Ukončení programu.\033[0m")


class PageSlider(matplotlib.widgets.Slider):
    def __init__(self, ax, label, numpages=10, valinit=0, valfmt='%1d',
                 closedmin=True, closedmax=True,
                 dragging=True, **kwargs):

        self.facecolor = kwargs.get('facecolor', "w")
        self.activecolor = kwargs.pop('activecolor', "b")
        self.fontsize = kwargs.pop('fontsize', 10)
        self.numpages = numpages

        super(PageSlider, self).__init__(ax, label, 0, numpages,
                                         valinit=valinit, valfmt=valfmt, **kwargs)

        self.poly.set_visible(False)
        self.vline.set_visible(False)
        self.pageRects = []
        for i in range(numpages):
            facecolor = self.activecolor if i == valinit else self.facecolor
            r = matplotlib.patches.Rectangle((float(i) / numpages, 0), 1. / numpages, 1,
                                             transform=ax.transAxes, facecolor=facecolor)
            ax.add_artist(r)
            self.pageRects.append(r)
            ax.text(float(i) / numpages + 0.5 / numpages, 0.5, str(i + 1),
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=self.fontsize)
        self.valtext.set_visible(False)

        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        bax = divider.append_axes("right", size="5%", pad=0.05)
        fax = divider.append_axes("right", size="5%", pad=0.05)
        self.button_back = matplotlib.widgets.Button(bax, label='<',
                                                     color=self.facecolor, hovercolor=self.activecolor)
        self.button_forward = matplotlib.widgets.Button(fax, label='>',
                                                        color=self.facecolor, hovercolor=self.activecolor)
        self.button_back.label.set_fontsize(self.fontsize)
        self.button_forward.label.set_fontsize(self.fontsize)
        self.button_back.on_clicked(self.backward)
        self.button_forward.on_clicked(self.forward)

    def _update(self, event):
        super(PageSlider, self)._update(event)
        i = int(self.val)
        if i >= self.valmax:
            return
        self._colorize(i)

    def _colorize(self, i):
        for j in range(self.numpages):
            self.pageRects[j].set_facecolor(self.facecolor)
        self.pageRects[i].set_facecolor(self.activecolor)

    def forward(self, event):
        current_i = int(self.val)
        i = current_i + 1
        if (i < self.valmin) or (i >= self.valmax):
            return
        self.set_val(i)
        self._colorize(i)

    def backward(self, event):
        current_i = int(self.val)
        i = current_i - 1
        if (i < self.valmin) or (i >= self.valmax):
            return
        self.set_val(i)
        self._colorize(i)


if __name__ == "__main__":

    image_folder = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos\__test_1\original'

    photos = load_photos(image_folder)
    print("Fotky načteny.")

    num_pages = len(photos)

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.18)

    ax_slider = fig.add_axes([0.1, 0.05, 0.8, 0.04])
    slider = PageSlider(ax_slider, 'Page', num_pages, activecolor="orange")

    if not os.path.isdir(image_folder):
        print(f"Načtená hlavní složka neexituje: [{image_folder}]")
        while not os.path.isdir(image_folder):
            image_folder = input("\tZadejte novou cestu ke složce: ")

    img = ax.imshow(photos[0])
    ax.autoscale(True)


    def update(val):
        i = min(max(int(val), 0), num_pages - 1)
        img.set_data(photos[i])
        fig.canvas.draw()


    slider.on_changed(update)

    plt.show()
