from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import cv2
import sys
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Data pro graf
x = np.linspace(0, 10, 100)  # Vytvoření dat pro osu x
y1 = np.sin(x)  # První sada dat
y2 = np.cos(x)  # Druhá sada dat

make_line_graph = True

# Data pro barevné mapování
subregion_values = [1, 2, 3]

# Vytvoření obrázku pro zobrazení
image_data = cv2.imread("book.jpg")[:1000, :]

figures = 2

ratio = (image_data.shape[1] / image_data.shape[0])
if figures * 5 * ratio > 16:
    size = 16 / (figures * ratio)
else:
    size = 5

# LaTeX setup
# plt.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['text.usetex'] = True

# Vytvoření většího grafu s dvěma podgrafy (1x2)
fig, axs = plt.subplots(1, figures, figsize=(figures * size * ratio, size), gridspec_kw={'width_ratios': [2, 2]})
if not isinstance(axs, np.ndarray):
    axs = [axs]

plt.suptitle("MAIN big graph", y=0.98, fontweight='bold', wrap=True)

# Použití for loopu pro generování dvou podgrafů
for ax in axs:
    if make_line_graph and ax == axs[0]:
        # První subplot
        ax.set_title('Graf sin(x)\ngraf_new line', pad=5, wrap=True)
        ax.set_axis_off()

        inner_axes = ax.inset_axes([0.07, 0.2, 0.93, 0.65])  # left, down, right, up
        # Nastavení z-indexu pro vložený graf tak, aby byl nad hlavním grafem

        inner_axes.plt(x[:40], y1[:40], color='dodgerblue', label='sin(x)')
        inner_axes.plt(x, y1, color='dodgerblue', alpha=0.3, lw=0.75)

        inner_axes.fill_between(x[:40], y1[:40], color='dodgerblue', alpha=0.25, label='Vybarvená oblast', zorder=7)
        inner_axes.fill_between(x, y1, color='skyblue', alpha=0.25, label='Vybarvená oblast 2', zorder=7)

        inner_axes.scatter(x[[2, 6, 20, 25, 32]], y1[[2, 6, 20, 25, 32]], zorder=12,
                           facecolor='dodgerblue', edgecolor='white')

        xticks = np.arange(np.min(x), np.max(x) + 1, 1)  # Nastavení značek na ose x na 0, 2, 4, 6, 8, 10
        inner_axes.set_xticks(xticks)
        yticks = np.linspace(np.min(y1), np.max(y1), 7)  # Nastavení značek na ose x na 0, 2, 4, 6, 8, 10
        inner_axes.set_yticks(yticks)

        # Získání rozsahů os x a y na začátku
        x_range_start, x_range_end = inner_axes.get_xlim()
        y_range_start, y_range_end = inner_axes.get_ylim()

        # Spočítání původního poměru os
        initial_aspect_ratio = (x_range_end - x_range_start) / (y_range_end - y_range_start)

        # Nastavení stejného poměru os x a y při přibližování
        # inner_axes.set_aspect(initial_aspect_ratio / 3)

        print(inner_axes.get_aspect())

        xticks = inner_axes.get_xticks().tolist()  # Získání hodnot na ose x
        yticks = inner_axes.get_yticks().tolist()  # Získání hodnot na ose y

        tick_labels = [f"{label:.2f}".replace('.', ',') for label in sorted(xticks)]
        inner_axes.set_xticklabels(tick_labels)
        tick_labels = [f"{label:.2f}".replace('.', ',') for label in sorted(yticks)]
        inner_axes.set_yticklabels(tick_labels)

        inner_axes.set_xlabel('x', labelpad=1)
        inner_axes.set_ylabel('y', labelpad=1)
        inner_axes.grid()
        inner_axes.axhline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.3)

        inner_axes.plt(0, 0, "x")

        x_ = np.arange(2)

        markers = ["o", "s", "d", "^"]
        labels = ["Total", "Very very very long label", "Component 1", "Component 3"]

        for i in range(4):
            inner_axes.plt(x_ + i, x_ * 0.1 * i, marker=markers[i], label=labels[i])

        # Vytvoření druhého grafu s druhou y-osou
        inner_axes2 = inner_axes.twinx()
        inner_axes3 = inner_axes.twiny()

        # inner_axes.set_aspect(initial_aspect_ratio/3)
        # inner_axes2.set_yticks()
        inner_axes2.axis('off')
        inner_axes3.axis('off')
        # inner_axes2.plt(x, y2, 'r-')
        # inner_axes2.set_ylabel('Druhá Y osa', color='r')

        # Vytvoření druhé x-osy s vlastním měřítkem
        secax = inner_axes.secondary_xaxis('top')
        secax.set_xlabel('Time [s]')
        secax.set_xlim(-1, 30)  # Nastavení začátku a konce druhé x-osi

        secax.xaxis.set_minor_locator(plt.MultipleLocator(base=0.2))

        rozsa = np.array(range(0, 33))
        a = np.array([""] * len(rozsa))
        l = np.linspace(0, 32, 9, dtype=int)
        a[l] = rozsa[l]
        a.tolist()
        secax.set_xticks(range(0, 33))
        new_labels = ['A', 'B', 'C', '12', 'E']  # Nové popisky pro značky
        secax.set_xticklabels(a, fontsize=5)

        secax.tick_params(which='major', axis='x', length=6, width=0.8, color='black', direction='out')  # 'inout'

        # Nastavení stylu čáry pro minor ticky
        secax.tick_params(which='minor', axis='x', length=3.5, width=0.8, color='gray', direction='out')

        # Nastavení stylu čar pro horní hranu grafu (spiny)
        for side in ('top', 'bottom', 'left', 'right'):
            inner_axes.spines[side].set_color('red')
            inner_axes.spines[side].set_linewidth(2)
            inner_axes.spines[side].set_linestyle(':')
        secax.spines['top'].set_linewidth(0)

        h, l = inner_axes.get_legend_handles_labels()
        kw = dict(ncol=5, loc="lower center", frameon=False)
        s = len(h) // 2
        leg1 = inner_axes.legend(h[:s], l[:s], bbox_to_anchor=[0.5, -0.4], **kw)
        leg2 = inner_axes.legend(h[s:], l[s:], bbox_to_anchor=(0, -0.55, 1, 0.2), **kw)  # (x, y, width, height)
        inner_axes.add_artist(leg1)

        # inner_axes.set_xlim(np.min(x)-1, np.max(x)+1)
        # inner_axes.set_ylim(np.min(y1)-1, np.max(y1)+1)

        # inner_axes.axis('equal')
    else:
        # Přidání obrázku do podgrafu pomocí imshow

        arrow_length = min(image_data.shape[0] * 0.2, image_data.shape[1] * 0.2)
        # Určete body polygonu pro šipku
        points = np.array([
            (0, 0), (arrow_length, 0), (arrow_length * 0.6, arrow_length * 0.3),
            (arrow_length * 0.6, arrow_length * 0.15), (arrow_length * 0.15, arrow_length * 0.15),
            (0, arrow_length * 0.15)], dtype=np.int32)
        points += 2

        result_matrix = np.vstack((points, points[:, [1, 0]]))

        # Vykreslete šipku jako polygon
        ax.add_patch(Polygon(points, closed=True, facecolor='black', alpha=0.7))
        ax.plt(points[:-1, 0], points[:-1, 1], color='white', linewidth=2, alpha=0.7)
        points[:, [0, 1]] = points[:, [1, 0]]
        ax.add_patch(Polygon(points, closed=True, facecolor='black', alpha=0.7))
        ax.plt(points[:-1, 0], points[:-1, 1], color='white', linewidth=2, alpha=0.7)

        ax.text(points[1, 1] * 0.95, arrow_length * 0.45, 'x', fontsize=12, fontweight='bold',
                color='white', bbox=dict(facecolor='black', edgecolor='white', boxstyle='round', pad=0.2, alpha=0.5))
        ax.text(arrow_length * 0.3, points[1, 1] * 1.1, 'y', fontsize=12, fontweight='bold',
                color='white', bbox=dict(facecolor='black', edgecolor='white', boxstyle='round', pad=0.2, alpha=0.5))

        image = ax.imshow(image_data, cmap='gray', aspect='equal')  # Změňte cmap podle potřeby

        # Přidání Polygonu
        polygon = Polygon([(200, 500), (300, 800), (400, 700)], closed=True, facecolor='lightblue', edgecolor='blue')
        ax.add_patch(polygon)

        ax.set_title('Jednoduchý graf', pad=5, wrap=True)

        """text = ("Toto je velmi dlouhý text, který se zalamuje, pokud je delší než 500 znaků. Toto je velmi dlouhý text,"
                " který se zalamuje, pokud je delší než 500 znaků.")
    
        # Nastavení maximální šířky řádku na 500
        wrapped_text = textwrap.fill(text, width=40)
        ax.text(image_data.shape[1]//2, -400, wrapped_text, ha='center', va='center', fontsize=12, wrap=True)"""

        ax.set_facecolor('none')
        ax.axis('equal')
        ax.axis('off')

        # Přidání barevného mapování a colorbaru
        if ax == axs[-1]:
            subregion_values[0] += 50

        scalar_map = plt.cm.ScalarMappable(cmap='seismic')
        # scalar_map.set_array(subregion_values)

        vmin, vmax = np.min(subregion_values), np.max(subregion_values)
        scalar_map.set_clim(vmin=vmin)
        scalar_map.set_clim(vmax=vmax)

        spacing = np.linspace(vmin, vmax, 9).tolist()
        spacing += [5.12554526562]
        spacing.sort()

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", 0.2, pad=0.25)

        # Přidání colorbaru k prvnímu podgrafu (axs[0])
        cbar = plt.colorbar(scalar_map, cax=cax, ticks=spacing)  # , drawedges=False, pad=0.03, shrink=0.9,
        # aspect=15)  # format=FuncFormatter(lambda x, _: f"{x:.2f}".replace('.', ','))

        # Změna formátu pro první číslo
        tick_labels = cbar.get_ticks().tolist()
        tick_labels = [f"{label:.2f}".replace('.', ',') for label in sorted(tick_labels)]
        # tick_labels[-1] = f"[mm] →↑\n\n{tick_labels[-1]}\n\n"
        cbar.set_ticklabels(tick_labels)

        cbar.set_label('[mm]', labelpad=-10, y=1.08, rotation=0)

        value = 15
        cbar.ax.plot([0, 0.25], [value, value], color='black', lw=0.6, alpha=0.5)
        cbar.ax.plot([0.75, 1], [value, value], color='black', lw=0.6, alpha=0.5)
        cbar.ax.annotate(f'{value}', xy=(0, value), xytext=(-15, 0),
                         textcoords='offset points', color='black', fontsize=10)

        ax.set_xlim(xmin=0, xmax=image_data.shape[1])  # Omezení osy x
        ax.set_ylim(ymin=image_data.shape[0], ymax=0)  # Omezení osy y
        # ax.set_ybound(lower=0, upper=10)

        # ax.text(image_data.shape[1], -100, '[MPa]', ha='right', fontsize=11)

# plt.tight_layout()
plt.subplots_adjust(top=0.8,
                    bottom=0.1,
                    left=0.01,
                    right=1.0,
                    hspace=0.0,
                    wspace=0)

"""# Změna velikosti prvního subgrafu
left, bottom, width, height = axs[0].get_position().bounds
axs[0].set_position([left+0.04, bottom + 0.1, width-0.1, height - 0.2])"""

# Zobrazení většího grafu
# form = "jpg"
# plt.savefig(f"graf_pokus2.{form}", format=f"{form}", dpi=300)
# plt.savefig('graf.pgf', format='pgf', bbox_extra_artists=[plt.xlabel, plt.ylabel, plt.title, plt.legend], dpi=300)

# Save the full figure...


"""import matplotlib.transforms as mtransforms
from matplotlib.backends.backend_agg import FigureCanvasAgg

fig.suptitle("")

left, bottom, width, _ = axs[0].get_position().bounds
left2, _, width2, _ = axs[1].get_position().bounds

figure = fig

figure.savefig(
    "a.raw",
    bbox_inches=mtransforms.Bbox([[0, 0], [0.0009, 0.003]]).transformed(
        figure.transFigure - figure.dpi_scale_trans
    ),
)

figure.savefig(
    "bottom.png",
    # we need a bounding box in inches
    bbox_inches=mtransforms.Bbox(
        # This is in "figure fraction" for the bottom half
        # input in [[xmin, ymin], [xmax, ymax]]
        [[0, 0], [width + left * 2, 1 - bottom]]
    ).transformed(
        # this take data from figure fraction -> inches
        #    transFigrue goes from figure fraction -> pixels
        #    dpi_scale_trans goes from inches -> pixels
        (figure.transFigure - figure.dpi_scale_trans)
    ),
)


figure.savefig(
    "a.raw",
    bbox_inches=mtransforms.Bbox([[0, 0], [0.0009, 0.003]]).transformed(
        figure.transFigure - figure.dpi_scale_trans
    ),
)

figure.savefig(
    "top.png",
    bbox_inches=mtransforms.Bbox([[width + left + (left2 - width2), 0], [1, 1 - bottom]]).transformed(
        figure.transFigure - figure.dpi_scale_trans
    ),
)

os.remove("a.raw")"""

plt.show()

"""for i in ('eps', 'jpeg', 'jpg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff', 'webp'):
    fig.savefig(
        f"a.{i}",  # eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff, webp
        bbox_inches=mtransforms.Bbox([[0, 0], [0.0009, 0.003]]).transformed(
            fig.transFigure - fig.dpi_scale_trans
        ),
    )"""
