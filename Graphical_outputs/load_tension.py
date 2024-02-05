import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Načtení dat ze souboru CSV
path = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\data\data_csv'

file_names = np.array([p for p in os.listdir(path) if os.path.isfile(os.path.join(path, p))
                       and p.lower().endswith(".csv") and p.startswith("T01")])
"""for f in file_names:
    n = os.path.join(path, f)
    df = pd.read_csv(n)  # DATAFRAME
    df = df.replace(',', '.', regex=True)
    names = df.axes[1]
    values = df.values
    values[:, 1] /= 1000
    df = pd.DataFrame(dtype=float)
    df[names] = values
    # df.to_csv(n, index=False)"""

# p = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos'
"""
tt = {}
for f in file_names:
    f = f.replace(".csv", "")
    n = os.path.join(p, f, "original")
    t = [os.path.getmtime(os.path.join(n, p)) for p in os.listdir(n) if
         p.lower().endswith(".jpg") and not p.startswith("0")][2:]
    t = [int(t[i + 1] - t[i]) for i in range(len(t) - 1)]
    tt[f] = t
    print(f, ":", t[:])"""

measurement1 = np.array(([2, 5, 8, 11, 14, 17, 20, 23, 26, 29],  # I
                         [1, 4, 7, 10, 13, 16, 19, 22, 25, 28],  # II
                         [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]  # III
                         )).T

measurement2 = np.array(([32, 35, 38, 41, 44, 47, 50],  # MAX I
                         [31, 34, 37, 40, 43, 46, 49],  # MAX II
                         [30, 33, 36, 39, 42, 45, 48]  # MAX III
                         )).T

measurement3 = np.array(([2, 5, 8, 11, 14, 17, 20],  # I
                         [1, 4, 7, 10, 13, 16, 19],  # II
                         [0, 3, 6, 9, 12, 15, 18],  # III
                         [32, 35, 38, 41, 44, 47, 50],  # MAX I
                         [31, 34, 37, 40, 43, 46, 49],  # MAX II
                         [30, 33, 36, 39, 42, 45, 48]  # MAX III
                         )).T

measurement = measurement3

file_names = file_names[measurement.flatten()]  # [-1:]

if isinstance(measurement, np.ndarray) and len(measurement.shape) > 1:
    tab_colors = [list(plt.cm.colors.to_rgba(color)[:3]) for color in list(plt.cm.colors.TABLEAU_COLORS.values())]
    colors = ('dodgerblue', 'orange', 'limegreen', 'gray', 'red', 'aqua', 'magenta')
    tab_colors = [list(plt.cm.colors.to_rgba(color)[:3]) for color in colors]
    colors = []
    for row in range(measurement.shape[1]):
        for col in range(measurement.shape[0]):
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
    y_data = df.iloc[:, 1].values
    x_data -= x_data[0]
    y_data -= y_data[0]

    numeric_values = pd.to_numeric(df.iloc[:, 0], errors='coerce').values

    # Najděte index, kde platí, že rozdíl je roven 100
    end_index = min(np.where(y_data - np.roll(y_data, -10) >= 5)[0][0] + 100, len(y_data))

    line_x.append(x_data[:end_index])
    line_y.append(y_data[:end_index])

[plt.plot(x, y, label=f'{n}', color=c) for x, y, n, c in zip(line_x, line_y, file_names, colors)]
y = line_y[0]
a = y[30]
sm = [(y[30] - y[10]) / (x[30] - x[10]) for x, y in zip(line_x, line_y)]
e = [(((y[30] - y[10]) / (x[30] - x[10])) * (42.5 / (2.64 * 15.13))) for x, y in zip(line_x, line_y)]
[print(f"{(n + ':').ljust(31)} {val: .5f}".replace(".", ",")) for val, n in zip(sm, file_names)]

sm_std = np.std(sm)
sm_av = np.mean(sm)
sm_med = np.median(sm)

e_std = np.std(e)
e_av = np.mean(e)
e_med = np.median(e)

print(f"\nSměrnice:\n\t{'Směrodatná odchylka:'.ljust(25)} {sm_std: .5f}\n\t{'Průměr:'.ljust(25)} {sm_av: .4f}"
      f"\n\t{'Medián:'.ljust(25)} {sm_med: .4f}")

print(f"\nModul pružnosti:\n\t{'Směrodatná odchylka:'.ljust(25)} {e_std: .5f}\n\t{'Průměr:'.ljust(25)} {e_av: .4f}"
      f"\n\t{'Medián:'.ljust(25)} {e_med: .4f}")

mean_x = [np.median([x[i] for x in line_x]) for i in range(np.min([x.shape[0] for x in line_x]))]
mean_y = [np.median([y[i] for y in line_y]) for i in range(np.min([y.shape[0] for y in line_y]))]

plt.plot(mean_x, mean_y, label=f'MEAN', lw=3, ls='--', c="black")
plt.scatter(line_x[0][[30, 10]], line_y[0][[30, 10]], c="black", marker='X', s=100, zorder=4)

plt.plot([0, line_x[0][-1]], [0, line_x[0][-1] * sm_av], color="black", lw=1)
x_range_start, x_range_end = plt.gca().get_xlim()  # rozsah x
y_range_start, y_range_end = plt.gca().get_ylim()  # rozsah y
plt.gca().set_aspect(((x_range_end - x_range_start) / (y_range_end - y_range_start)) / 2, adjustable='box')
plt.gca().autoscale(True)
plt.legend(fontsize=8, ncol=2, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
plt.tight_layout(pad=3)
plt.show()
