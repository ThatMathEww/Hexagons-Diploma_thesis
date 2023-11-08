import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors


def gradient_fill(data_x=None, data_y=None, fill_color=None, ax=None, line=None, up_alpha=None, down_alpha=0.05,
                  z_order=None, gradient_type: str | int = 0, detail_multi=5):
    if line is None and data_x is None and data_y is None:
        return
    if not (isinstance(fill_color, str) or isinstance(fill_color, tuple) or fill_color is None):
        return
    if all((gradient_type != 0, gradient_type != 1, gradient_type != "constant", gradient_type != "smooth")):
        print("\n\033[33;1;21mWARRNING\033[0m\n\tŠpatně zvolen typ gradientu.")
        gradient_type = 0

    if ax is None:
        ax = plt.gca()
    if line is None and not (data_x is None and data_y is None):
        max_x, min_x, max_y, min_y = np.max(data_x), np.min(data_x), np.max(data_y), np.min(data_y)
        line = ax.plot(data_x, data_y, color=fill_color)[0]
    else:
        data_x = line.get_xdata()
        data_y = line.get_ydata()
        max_x, min_x, max_y, min_y = np.max(data_x), np.min(data_x), np.max(data_y), np.min(data_y)
    if fill_color is None:
        fill_color = line.get_color()
    if z_order is None:
        z_order = line.get_zorder()

    line_alpha = line.get_alpha()
    up_alpha = 1 if up_alpha is None and line_alpha is None else line_alpha if up_alpha is None else up_alpha
    if up_alpha > 1:
        print("\n\033[37;1;21mWARRNING\033[0m"
              "\033[37m\n\tMaximální průhlednost byla nastavena na hodnotu větší než 1.\033[0m")
    if down_alpha < 0:
        print("\n\033[37;1;21mWARRNING\033[0m"
              "\033[37m\n\tMinimální průhlednost byla nastavena na hodnotu menší než 0.\033[0m")

    detail_multi = max(1, detail_multi)

    data_shape = data_y.shape[0]
    new_length = data_shape * detail_multi
    rgb = plt.cm.colors.to_rgba(fill_color)[:3]

    if gradient_type == 1 or gradient_type == "smooth":
        quality = data_shape * detail_multi
        smoothed_data = [(0.5 * np.median(data_y[max(0, i - detail_multi):min(data_shape, i + detail_multi + 1)]) +
                          1.5 * np.mean(data_y[max(0, i - detail_multi):min(data_shape, i + detail_multi + 1)])) / 2
                         for i in range(data_shape)]

        # Vytvoření nového pole s interpolovanými hodnotami
        output_array = np.interp(np.linspace(0, data_shape - 1, new_length), np.arange(data_shape), smoothed_data)

        z = np.ones((new_length, quality, 4), dtype=np.float32)
        z[:, :, :3] = rgb
        for p in range(new_length):
            pos = np.int32(np.ceil((output_array[p] - min_y) / (max_y - min_y) * quality))
            z[p, :pos, -1] = np.linspace(0, 1, pos)
        z = np.transpose(z, axes=(1, 0, 2))
    else:
        # z = np.atleast_2d(np.linspace(0, 1, 256, dtype=np.float32)).T
        z = np.ones((new_length, 1, 4), dtype=np.float32)
        z[:, :, :3] = rgb
        z[:, :, -1] = np.linspace(0, 1, new_length)[:, None]

    min_val = np.min(z[:, :, -1])
    z[:, :, -1] = down_alpha + ((z[:, :, -1] - min_val) / (np.max(z[:, :, -1]) - min_val)) * (up_alpha - down_alpha)
    z[:, :, -1] = np.clip(z[:, :, -1], a_min=0, a_max=1)

    c_map = plt.get_cmap('Blues')

    fill = ax.imshow(z, aspect='auto', extent=[min_x, max_x, min_y, max_y], origin='lower', zorder=z_order, cmap=c_map)
    xy = np.vstack([[min_x, min_y], np.column_stack([data_x, data_y]), [max_x, min_y], [min_x, min_y]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    fill.set_clip_path(clip_path)
    ax.autoscale(True)

    def gradient_bars(bars):
        from matplotlib.colors import ListedColormap

        # viridis = plt.get_cmap('Blues').resampled(256)
        # newcolors = viridis(np.linspace(0, 1, 256))
        newcolors = np.empty((256, 4))
        # col = np.array([0 / 256, 53 / 256, 256 / 256, 1])
        newcolors[:, :-1] = rgb
        newcolors[:, -1] = np.linspace(0, 1, 256)
        newcmp = ListedColormap(newcolors)

        grad = np.atleast_2d(np.linspace(1, 0, 256)).T
        ax = bars[0].axes
        lim = ax.get_xlim() + ax.get_ylim()
        for bar in bars:
            bar.set_zorder(1)
            bar.set_facecolor("none")
            x, y = bar.get_xy()
            w, h = bar.get_width(), bar.get_height()
            ax.imshow(grad, extent=[x, x + w, y, y + h], aspect="auto", zorder=0, cmap=newcmp)
        ax.axis(lim)

    fig, ax = plt.subplots()
    bar = ax.bar([1, 2, 3, 4, 5, 6], [4, 5, 6, 3, 7, 5])
    gradient_bars(bar)
    fig.tight_layout()


fig, axs = plt.subplots()
x = np.linspace(0, 100, 100)
np.random.seed(1977)
y = np.random.normal(0, 0.5, 100).cumsum()

gradient_fill(ax=axs, data_x=x, data_y=y, up_alpha=0.85, down_alpha=0.1, gradient_type=0, fill_color='#1F77B4')
fig.tight_layout()

# Vytvoření vlastního gradientu barev
colors = ['lightgray', 'royalblue', 'dodgerblue', 'navy', '#061C2B']  # Přechod mezi červenou, zelenou a modrou

list_1 = ['#CCCCCC', '#99E6FF', '#6ECBF7', '#00A9F4', '#0679C3', '#084D91', '#061C2B']
list_2 = ['#E6E6E6', '#82A6C9', '#386694', '#1B456E', '#051C2C']
list_3 = ['#E6E6E6', '#99E6FF', '#6ECBF7', '#34B4F4', '#0291DC']
list_4 = ['#A2E6FE', '#18AFF3', '#3761FE', '#1D3485']
list_5 = ['#00A9F4', '#051C2C']
list_6 = ['#B3C1CA', '#7DBEC7', '#7FCCF0', '#4EACE0']

list_6_2 = ['#4EA0AB', '#B3C1CA', '#4EACE0']
list_6_2_light = ['#BCDEE1', '#E5EAED', '#BDE6F7']

list_7 = ['#3C96B4', '#AAE6F0', '#2251FF', '#00A9F4', '#051C2C']
list_8 = ['#D9E0E5', '#4EACE0']

base_blue = '#4EACE0'
sec_blue = '#2251FF'
third_blue = '#00A9F4'
dark_blue = '#051C2C'
background = '#F0F0F0'

tot_list = ['white', 'lightgray', '#CCCCCC', '#A2E6FE', '#99E6FF', '#6ECBF7', '#34B4F4', '#18AFF3', '#00A9F4',
            '#0291DC', '#0679C3', '#084D91', '#1D3485', '#00003E', '#061C2B', '#051C2C']

my_list = ['#CCCCCC', '#7DBEC7', '#3C96B4', '#34B4F4', '#0679C3', '#1D3485', '#051C2C']
my_list_blues = ['#CCCCCC', '#99E6FF', '#34B4F4', '#0679C3', '#1D3485', '#051C2C']

list_red = ['#FF96BA', '#E5395F', '#C8142F', '#941741', '#920422', '#320015']

list_rb_1 = ['#335789', '#EA494F']
list_rb_2 = ['#4A001E', '#731331', '#9F2945', '#CC415A', '#E06E85', '#ED9AB0', '#F8C3D9',
             '#FAF0FF',
             '#C6D0F2', '#92B2DE', '#5D94CB', '#2F74B3', '#265191', '#163670', '#0B194C'] # --> plt: "RdBu" // "RdYlBu"


my_blue = '0073E6'
my_blue2 = '#1E90FF'

cmap = mcolors.LinearSegmentedColormap.from_list('custom', my_list, N=256)

# Vytvoření dat pro zobrazení
data = np.random.rand(10, 10)

plt.figure()
# Použití vlastní colormapy
plt.imshow(data, cmap=cmap)
plt.colorbar()
plt.tight_layout()

plt.show()
