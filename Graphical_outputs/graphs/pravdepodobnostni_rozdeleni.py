import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator


def gradient_bars(bars, x_data, y_data, up_alpha=0.9, down_alpha=0.15, bin_alpha=0.9, line_color='tab:blue',
                  grad_num=256, line_alpha=0.85, line_width=1.1, multiple_colors=True, multi_grad_colors=True,
                  equal_grad_color=True, c_map: str = None):
    grad = np.clip(np.atleast_2d(np.linspace(up_alpha, down_alpha, 256)).T, a_min=0, a_max=1)

    if multiple_colors:
        if equal_grad_color:
            center = x_data[np.argmax(y_data)]
            # Najít největší hodnotu
            diff = np.max(np.abs(x_data - center))
            min_diff = center - diff
            max_diff = center + diff
        else:
            min_diff = np.min(x_data)
            max_diff = np.max(x_data)

        if c_map is not None:
            color_map = plt.cm.get_cmap(c_map)  # 'Blues'
        else:
            colors = ['#b9e5fb', '#8ed8f8', '#6dcff6', '#20c4f4', '#00b9f2', '#00aeef', '#00a1e4', '#0095da', '#0083ca']
            color_map = LinearSegmentedColormap.from_list("custom", colors + colors[::-1][1:],
                                                          N=max(grad_num, len(colors) * 2))

        sm = plt.cm.ScalarMappable(cmap=color_map)  # Inicializace ScalarMappable
        sm.set_clim(min_diff, max_diff)  # Nastavení rozsahu hodnot pro ScalarMappable
    else:
        # color = np.atleast_2d(np.linspace(1, 0, 256)).T
        color = np.tile(plt.cm.colors.to_rgba('dodgerblue'), (grad.shape[0], 1, 1))
        color[:, :, -1] = grad



    bar_ax = bars[0].axes
    lim = bar_ax.get_xlim() + bar_ax.get_ylim()
    ax.axis(lim)
    for b in bars:
        # b.set_zorder(1)
        # b.set_facecolor("none")
        b.set_visible(False)
        x_b, y_b = b.get_xy()
        w_b, h_b = b.get_width(), b.get_height()

        if multiple_colors:
            if multi_grad_colors:
                # Převedení hodnot na barvy pomocí ScalarMappable
                color_gradient = sm.to_rgba(np.linspace(x_b, x_b + w_b, 256))
                color = np.tile(color_gradient.T[:, :, np.newaxis], grad.shape[0]).T  # barevný přechod v rámci binu
            else:
                color = np.tile(sm.to_rgba((x_b + (x_b + w_b)) / 2), (grad.shape[0], 1, 1))  # barvy pro danou hodnotu
            color[:, :, -1] = grad

        bar_ax.imshow(color, extent=[x_b, x_b + w_b, y_b, y_b + h_b], aspect="auto", zorder=3, alpha=bin_alpha)
        # cmap='Blues'
        bar_ax.add_patch(plt.Rectangle((x_b, y_b), w_b, h_b, edgecolor=line_color, linewidth=line_width,
                                       facecolor='none', alpha=line_alpha, zorder=4))


# Nastavení parametrů gausovského rozdělení
mean_dev = 9  # Střední hodnota
std_dev = 1  # Směrodatná odchylka
plt.rcParams['font.family'] = 'Times New Roman'
# Generování dat z gausovského rozdělení
# data = np.random.normal(mean_dev, std_dev, 1000)  # Generuje 1000 vzorků
# data = [15, 15.5, 14.9, 10.3, 18.9, 15.1, 12, 23]
data = np.array([27.12, 21.44, 27.09, 23.85, 27.03, 21.74, 24.88, 21.28, 18.28, 23.64, 23.43, 23.14, 20.3, 22.88, 22.79,
                 28.3, 27.08, 20.67, 24.46, 37.74, 21.87, 20.84, 21.02, 23.92, 18.99, 18.76, 24.64, 16.99, 21.48, 20.9,
                 20.34])
std = np.std(data)
mean = np.mean(data)

fig, ax = plt.subplots()

# Vykreslení histogramu dat
bar = plt.hist(data, bins=15, density=False)

# Vykreslení teoretického gausovského rozdělení
# x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
x = np.linspace(np.min(data), np.max(data), 1000)
pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
# pdf = (((pdf - np.min(pdf)) / (np.max(pdf) - np.min(pdf)))) * np.max(bar[0])
pdf *= (np.max(bar[0]) / np.max(pdf))

gradient_bars(bar[2], x, pdf)

max_value = x[np.argmax(pdf)]
median_value = np.median(data)
mean_value = np.mean(data)
std_value = np.std(data)
print(f"\nStřední hodnota křivky: {max_value:.4f}\n"
      f"Medián dat: {median_value:.4f}\n"
      f"STD dat: {std_value:.4f}\n"
      f"Mean dat: {mean_value:.4f}\n")

ax.plot(x, pdf, '#28418C', label='Probability', zorder=6)

ax.vlines(mean_value, 0, np.max(pdf), colors='black', linestyles='dashed', label='Mean', zorder=5)
# Vybarvení oblasti pod křivkou od 0.5 do 1
plt.fill_between(x, pdf, where=[(mean_value - std_value <= i <= mean_value + std_value) for i in x],
                 color='darkcyan', alpha=0.5, label='Std', zorder=1)

condition = mean_value - std_value <= x
ax.vlines(x[condition][0], 0, pdf[condition][0], colors='teal', linewidth=0.8, zorder=5)
condition = mean_value + std_value >= x
ax.vlines(x[condition][-1], 0, pdf[condition][-1], colors='teal', linewidth=0.8, zorder=5)

# Přidání popisků
# plt.title('Gausovské pravděpodobnostní rozdělení')
plt.xlabel('Angle [°]')
plt.ylabel('Frequency')  # 'Hustota pravděpodobnosti'
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])

h, n = ax.get_legend_handles_labels()
# h[0] = plt.Rectangle((0, 0), 1, 1, fc="tab:blue", alpha=0.7)  # Vytvoření obdélníku pro značku
# plt.Line2D((0, 0), (1, 1), color="tab:blue", alpha=0.75)
# Put a legend below current axis
# ax.legend(h, n, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)

ax.set_aspect('auto', adjustable='box')
ax.relim()
ax.autoscale(True)

min_ax_x, max_ax_x = ax.get_xlim()
min_ax_y, max_ax_y = ax.get_ylim()
max_x, min_x = max(np.max(x), max_ax_x), min(np.min(x), min_ax_x)
max_y, min_y = max(np.max(pdf), max_ax_y), min(np.min(pdf), min_ax_y)
addition_x = (max_x - min_x) * 0.05
addition_y = (max_y - min_y) * 0.05
ax.set_xlim(min_x - addition_x, max_x + addition_x)
ax.set_ylim(0 if min_y == 0 else min_y - addition_y, max_y + addition_y)
plt.yticks(np.arange(min_ax_y, max_ax_y + 1, 1.0))

for axis in ['bottom', 'left', 'right']:
    plt.gca().spines[axis].set_linewidth(0.5)
    plt.gca().spines[axis].set_color('gray')

plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['top'].set_color('gray')
plt.gca().spines['top'].set_alpha(0.5)

plt.gca().tick_params(axis='x', which='major', direction='out', width=1, length=5, zorder=5, color="gray")
plt.gca().tick_params(axis='y', which='both', length=0)

for axis in ['left', 'right']:
    plt.gca().spines[axis].set_visible(False)

# Nastavení gridu pro major a minor ticks
plt.grid(axis='y', which='major', linestyle='-', linewidth='1.5', color='gray', alpha=0.5)
plt.grid(axis='y', which='minor', linestyle='-', linewidth='0.75', color='lightgray', alpha=0.8)

if plt.gca().get_ylim()[1] % plt.gca().get_yticks()[-1] != 0:
    step_ticks_y = np.mean(np.diff(plt.gca().get_yticks()))
    y_max = np.ceil(plt.gca().get_ylim()[1] / step_ticks_y) * step_ticks_y
    plt.gca().set_ylim(plt.gca().get_ylim()[0], y_max)

# ax.set(axisbelow=True)
ax.grid(axis='y', zorder=0, alpha=0.5)

plt.savefig("hist.pdf", format="pdf", bbox_inches='tight')
# Zobrazení grafu
plt.show()
