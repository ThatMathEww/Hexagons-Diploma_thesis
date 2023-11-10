import numpy as np
import matplotlib.pyplot as plt


def gradient_bars(bars):
    # color = np.atleast_2d(np.linspace(1, 0, 256)).T
    grad = np.clip(np.atleast_2d(np.linspace(0.85, 0.15, 256)).T, a_min=0, a_max=1)
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
        bar_ax.imshow(color, extent=[x_b, x_b + w_b, y_b, y_b + h_b], aspect="auto", zorder=0, alpha=0.5)
        # cmap='Blues'
        bar_ax.add_patch(plt.Rectangle((x_b, y_b), w_b, h_b, edgecolor='tab:blue', facecolor='none', alpha=0.85))


# Nastavení parametrů gausovského rozdělení
mean_dev = 9  # Střední hodnota
std_dev = 1  # Směrodatná odchylka

# Generování dat z gausovského rozdělení
# data = np.random.normal(mean_dev, std_dev, 1000)  # Generuje 1000 vzorků
# data = [15, 15.5, 14.9, 10.3, 18.9, 15.1, 12, 23]
data = np.array([27.12, 21.44, 27.09, 23.85, 27.03, 21.74, 24.88, 21.28, 18.28, 23.64, 23.43, 23.14, 20.3, 22.88, 22.79,
                 28.3, 27.08, 20.67, 24.46, 37.74, 21.87])
std = np.std(data)
mean = np.mean(data)

fig, ax = plt.subplots()

# Vykreslení histogramu dat
bar = plt.hist(data, bins=30, density=False, label='Hustota pravděpodobnosti')
gradient_bars(bar[2])

# Vykreslení teoretického gausovského rozdělení
# x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
x = np.linspace(np.min(data), np.max(data), 1000)
pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
# pdf = (((pdf - np.min(pdf)) / (np.max(pdf) - np.min(pdf)))) * np.max(bar[0])
pdf *= (np.max(bar[0]) / np.max(pdf))

median_value = x[np.argmax(pdf)]
print(f"\nStřední hodnota křivky: {median_value:.4f}\n"
      f"Medián dat: {np.median(data):.4f}")

plt.plot(x, pdf, '#28418C', label='Teoretická hustota pravděpodobnosti')

# Přidání popisků
plt.title('Gausovské pravděpodobnostní rozdělení')
plt.xlabel('Hodnota')
plt.ylabel('Četnost')  # 'Hustota pravděpodobnosti'
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])

h, n = ax.get_legend_handles_labels()
h[0] = plt.Rectangle((0, 0), 1, 1, fc="tab:blue", alpha=0.7)  # Vytvoření obdélníku pro značku
# plt.Line2D((0, 0), (1, 1), color="tab:blue", alpha=0.75)
# Put a legend below current axis
ax.legend(h, n, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)

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

# Zobrazení grafu
plt.show()
