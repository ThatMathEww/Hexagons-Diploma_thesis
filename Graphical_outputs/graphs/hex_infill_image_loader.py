import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os


def place_image(path=None, image=None, scale=1.1):
    if image is None and path is not None:
        img_d = mpimg.imread(path)
    elif image is not None:
        img_d = image.copy()
    else:
        return None

    max_size = int(np.max(img_d.shape[:2]) * scale)
    img_m = np.ones((max_size, max_size, 4), dtype=np.uint8)

    # Výpočet pozice pro vložení fotografie do středu prázdné matice
    start_x = (max_size - img_d.shape[1]) // 2
    start_y = (max_size - img_d.shape[0]) // 2

    # Vložení fotografie do prázdné matice
    img_m[start_y:start_y + img_d.shape[0], start_x:start_x + img_d.shape[1], :] = img_d * 255
    return img_m


def place_image2(path=None, image=None, scale=1.1):
    if image is None and path is not None:
        img_d = mpimg.imread(path)
    elif image is not None:
        img_d = image.copy()
    else:
        return None

    max_size = int(np.max(img_d.shape[:2]) * scale)
    img_m = np.ones((max_size, max_size, 4), dtype=np.uint8) * 255

    # Výpočet pozice pro vložení fotografie do středu prázdné matice
    start_x = (max_size - img_d.shape[1]) // 2
    start_y = (max_size - img_d.shape[0]) // 2

    # Vložení fotografie do prázdné matice
    img_m[start_y:start_y + img_d.shape[0], start_x:start_x + img_d.shape[1], :] = img_d * 255
    return img_m


load_scans = False

# Načtení PNG obrázku s transparentním pozadím
images_path = [i for i in os.listdir("./") if i.endswith(".png") and i.startswith("hexagon cantilever_")]

for i, image_path in enumerate(images_path, start=1):
    img = mpimg.imread(image_path)

    centers = ((img.shape[1] // 2, 230), (3630, 1130), (3630, 2960))
    positions = ((1.05, 0.85), (1.05, 0.4), (1.05, -0.05))
    # positions = ((1.05, 0.8), (1.05, 0.2))
    spacing = 100

    if image_path.endswith('_n.png'):
        paths = ('map3_edit.png', None, 'map2_edit.png')
    else:
        paths = ('map1_edit.png', None, None)

    # Vytvoření obrázku Matplotlib
    fig, ax = plt.subplots()

    # Zobrazení hlavního obrázku
    im = ax.imshow(img)

    for p, (sx, sy), pos in zip(paths, centers, positions):
        x1, x2, y1, y2 = sx - spacing, sx + spacing, sy - spacing, sy + spacing
        ax_ins = ax.inset_axes([*pos, 0.4, 0.4], xticks=[], xticklabels=[], yticks=[], yticklabels=[])
        # axins.invert_yaxis()

        img_det = place_image(image=img[y1: y2, x1: x2, :])

        ax_ins.imshow(img_det, extent=(x1, x2, y1, y2), origin="upper")

        # Nastavení stylu
        ax.indicate_inset_zoom(ax_ins, edgecolor="black", linewidth=1.05, alpha=1)
        [ax_ins.spines[axis].set_linewidth(1.05) for axis in ['top', 'bottom', 'left', 'right']]

        if load_scans:
            img_map = place_image2(path=p)
        else:
            img_map = None

        if img_map is not None:
            ax_ins.spines['right'].set_visible(False)

            axi_ns_insert = ax_ins.inset_axes([1, 0, 1, 1], xticks=[], xticklabels=[], yticks=[], yticklabels=[])
            axi_ns_insert.imshow(img_map, origin="upper")
            [axi_ns_insert.spines[axis].set_linewidth(1.05) for axis in ['top', 'bottom', 'left', 'right']]
            axi_ns_insert.spines['left'].set_visible(False)

    ax.axis('off')

    fig.savefig(f"hex{i}.pdf", format="pdf", bbox_inches='tight', dpi=1500)
    plt.tight_layout()
plt.show()
