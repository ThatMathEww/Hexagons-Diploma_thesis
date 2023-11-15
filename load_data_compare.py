import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Načtení dat ze souboru CSV
path = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\data\data_csv'

file_names = np.array([p for p in os.listdir(path) if os.path.isfile(os.path.join(path, p))
                       and p.lower().endswith(".csv") and p.lower().startswith("h01")])

if str(file_names[0]).lower().startswith("h01"):
    measurements1 = [4, 9, 14, 19, 24, 29]  # konzola, normal
    measurements2 = [30, 31, 32, 33, 34, 35, 36]  # konzola, snaped
    measurements3 = np.array(([3, 8, 13, 18, 23, 28],  # I
                              [2, 7, 12, 17, 22, 27],  # II
                              [1, 6, 11, 16, 21, 26],  # III
                              [0, 5, 10, 15, 20, 25])  # MAX
                             ).T
elif str(file_names[0]).lower().startswith("s01"):
    measurements1 = [4, 9, 14, 19, 24, 29]  # konzola, normal
    measurements2 = [30, 31, 32, 33, 34, 35, 36]  # konzola, snaped
    measurements3 = np.array(([0, 6, 12, 18],  # I
                              [2, 8, 14, 20],  # II
                              [4, 10, 16, 22],  # III
                              [1, 7, 13, 19],  # MAX I
                              [3, 9, 15, 21],  # MAX II
                              [5, 11, 17, 23],  # MAX III
                              )).T
    measurements4 = np.array(([0, 6, 12, 18, 24, 27, 30, 33, 36, 39],  # I
                              [2, 8, 14, 20, 25, 28, 31, 34, 37, 40],  # II
                              [4, 10, 16, 22, 26, 29, 32, 35, 38, 41],  # III
                              )).T
    measurements5 = np.array(([1, 7, 13, 19],  # MAX I
                              [3, 9, 15, 21],  # MAX II
                              [5, 11, 17, 23]  # MAX III
                              )).T

measurement = measurements3[2, :]
# measurements3[:-1, 4] , measurements3[:, :]

file_names = file_names[measurement.flatten() if isinstance(measurement, np.ndarray) else measurement]  # [-1:]

if isinstance(measurement, np.ndarray) and len(measurement.shape) > 1:
    tab_colors = [list(plt.cm.colors.to_rgba(color)[:3]) for color in list(plt.cm.colors.TABLEAU_COLORS.values())]
    colors = ('dodgerblue', 'orange', 'limegreen', 'gray', 'red', 'aqua', 'magenta')
    tab_colors = [list(plt.cm.colors.to_rgba(color)[:3]) for color in colors]
    colors = []
    for row in range(measurement.shape[0]):
        for col in range(measurement.shape[1]):
            colors.append([c * (1 - col * 0.1) for c in tab_colors[row]])
else:
    colors = [list(plt.cm.colors.to_rgba(color)[:3]) for color in list(plt.cm.colors.TABLEAU_COLORS.values())]
    while len(colors) < len(file_names):
        colors.extend(colors)
plt.figure(figsize=(10, 5))
plt.xlabel('Distance [mm]')
plt.ylabel('Force [N]')
plt.grid()

line_x, line_y = [], []
for file in file_names:
    df = pd.read_csv(os.path.join(path, file))  # DATAFRAME
    zr = 5
    d_len = df.shape[0]
    zr = max(min(zr, d_len // 3), 3)
    z2 = max(zr // 2, 1)
    # Vytvoření grafu
    x_data = df.iloc[:, 0].values  # První sloupec jako osa x - posun
    y_data = -((df.iloc[:, 1].values - df.iloc[:zr, 1].mean()) + (df.iloc[:, 2].values - df.iloc[:zr, 2].mean()))
    photo_indexes = df[df['Photos'].notna()].index

    # Najdi indexy, kde je okno rovno `pocet_podminka`
    start_index = np.max(np.where(y_data[:max(min(int(np.where(np.convolve(np.where(np.abs(np.array(
        [np.mean(y_data[s - z2:s + z2]) for s in range(z2, d_len // 3)])[zr:] - np.median(
        y_data[1:zr * 2 + 1])) >= np.mean(
        np.array([np.std(y_data[s - z2:s + z2]) for s in range(z2, d_len // 300)])[zr:]
        * 1.5), 1, 0), np.ones(int(np.ceil(d_len * 0.1))), mode='valid') == np.ceil(d_len * 0.1))[0][0]) - zr, d_len),
                                              0) + 1] <= 0)[0])

    # Najdi indexy, kde je rozdíl menší než -5
    snap_index = np.where(np.diff(y_data) <= -5)[0]

    x_data = x_data[start_index:] - x_data[start_index]
    y_data = y_data[start_index:]

    line_x.append(x_data)
    line_y.append(y_data)

[plt.plot(x, y, label=f'{n}', color=c) for x, y, n, c in zip(line_x, line_y, file_names, colors)]

e = [(y[2700] - y[2200]) / (x[2700] - x[2200]) for x, y in zip(line_x, line_y)]
[print(f"{(n + ':').ljust(31)} {val: .5f}") for val, n in zip(e, file_names)]

std = np.std(e)
e_av = np.mean(e)
e_med = np.median(e)

print(f"\n{'Směrodatná odchylka:'.ljust(25)} {std: .5f}\n{'Průměr:'.ljust(25)} {e_av: .4f}"
      f"\n{'Medián:'.ljust(25)} {e_med: .4f}")

mean_x = [np.median([x[i] for x in line_x]) for i in range(np.min([x.shape[0] for x in line_x]))]
mean_y = [np.median([y[i] for y in line_y]) for i in range(np.min([y.shape[0] for y in line_y]))]

plt.plot(mean_x, mean_y, label=f'MEAN', lw=3, ls='--', c="black")
plt.scatter(line_x[0][[2200, 2700]], line_y[0][[2200, 2700]], c="black", marker='X', s=100, zorder=4)

plt.plot([0, line_x[0][-1]], [0, line_x[0][-1] * e_av], color="black", lw=1)
x_range_start, x_range_end = plt.gca().get_xlim()  # rozsah x
y_range_start, y_range_end = plt.gca().get_ylim()  # rozsah y
plt.gca().set_aspect(((x_range_end - x_range_start) / (y_range_end - y_range_start)) / 2, adjustable='box')
plt.gca().autoscale(True)
plt.legend(fontsize=8, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
plt.tight_layout(pad=3)
plt.show()
