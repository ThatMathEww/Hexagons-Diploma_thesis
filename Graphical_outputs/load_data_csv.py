import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

"""from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression"""

file_type = "jpg"
out_dpi = 600

# H01_04_12s.csv
# H01_10_12s_p.csv
# H01_03-I_10s.csv / H01_03-II_10s.csv / H01_03-III_10s.csv   //   H01_03-II-max_12s.csv

# Načtení dat ze souboru CSV
file_path = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\data\data_csv\M01_glued_10s.csv'
df = pd.read_csv(file_path)  # DATAFRAME

zr = 5

d_len = df.shape[0]
zr = max(min(zr, d_len // 3), 3)
z2 = max(zr // 2, 1)

# Odečtení průměru od všech následujících hodnot v 2. a 3. sloupci
y_data1 = -(df.iloc[:, 1].values - df.iloc[:zr, 1].mean())
y_data2 = -(df.iloc[:, 2].values - df.iloc[:zr, 2].mean())
# Vytvoření grafu
x_data = df.iloc[:, 0].values  # První sloupec jako osa x - posun
y1 = -np.array(df.iloc[:, 1])  # Druhý sloupec jako osa y - síla1
y2 = -np.array(df.iloc[:, 2])  # Třetí sloupec jako osa y - síla2"""
y_data = y_data1 + y_data2  # - celková síla
photo_indexes = df[df['Photos'].notna()].index

start_value_y = np.median(y_data[1:zr * 2 + 1])
mean_values = np.array([np.mean(y_data[s - 3:s + 3]) for s in range(zr, d_len)])

# Najdi indexy, kde je okno rovno `pocet_podminka` - pro všechna data mít stejný počáteční index
s = [np.max(np.where(d[:max(min(int(np.where(np.convolve(np.where(np.abs(np.array(
    [np.mean(d[s - z2:s + z2]) for s in range(z2, d_len // 3)])[zr:] - np.median(d[1:zr * 2 + 1])) >= np.mean(
    np.array([np.std(d[s - z2:s + z2]) for s in range(z2, d_len // 300)])[zr:]
    * 1.5), 1, 0), np.ones(int(np.ceil(d_len * 0.1))), mode='valid') == int(np.ceil(d_len * 0.1)))[0][0]) - zr, d_len),
                            0) + 1] <= 0)[0]) for d in [y_data]]  # y_data1, y_data2, y_data

# Najdi indexy, kde je okno rovno `pocet_podminka`
# start_index = round((start_index1 + start_index2) / 2)
start_index = np.min(s)


def poly_function(poly_x, coefficients):
    print("KOEFICIENTY:\n", coefficients)
    print("ROZSAH:\n\t", poly_x[0], " - ", poly_x[-1])
    function = "Function:\n\t y = "
    # Získání počtu koeficientů
    len_c = len(coefficients)
    poly_y = np.zeros_like(poly_x)
    # Výpočet hodnoty y_fit
    for i in range(len_c):
        poly_y += coefficients[i] * poly_x ** (len_c - i - 1)
        if (len_c - i - 1) == 0:
            function += f"{coefficients[i]:.5f}" if -0.00001 < coefficients[i] < 0.00001 else f"{coefficients[i]:.5f}"
        elif (len_c - i - 1) == 1:
            function += f"{coefficients[i]:.5e}*x + " if -0.00001 < coefficients[i] < 0.00001 \
                else f"{coefficients[i]:.5f}*x + "
        else:
            function += f"{coefficients[i]:.5e}*x^{len_c - i - 1} + " if -0.00001 < coefficients[i] < 0.00001 \
                else f"{coefficients[i]:.5f}*x^{len_c - i - 1} + "
    if function.endswith("+ "):
        function = function[:-2]
    print(function)
    return poly_y


# Najdi indexy, kde je rozdíl menší než -30
snap_index = np.where(np.diff(y_data) <= -5)[0]
x_fit = np.linspace(x_data[0], x_data[start_index], start_index)
y_fit = np.zeros_like(x_fit)
if snap_index.size > 0:
    snap_index = snap_index[0]
    mean_values2 = np.array([np.mean(y_data[s:s + 2]) for s in range(snap_index)])
    mean_values2[0] = y_data[0]
    mean_values2[-1] = y_data[snap_index - 1]
    mean_values2[-2] = (mean_values2[-1] + mean_values2[-3]) / 2

    """d = np.array(
        [y_data[s] + (y_data[s + 1] - y_data[s]) / 2 for s in range(snap_index[0]+1, snap_index[1] + 50)])"""
    differ_index = np.where(np.diff(y_data[snap_index:]) <= -0.3)[0][[0, -1]] + snap_index

    d = [np.mean(y_data[s:s + 2]) for s in range(differ_index[0], differ_index[1] + 1)]
    d = np.array(
        [np.mean(d[s:s + 2]) for s in range(len(d))])
    mean_values2 = np.append(mean_values2, d)

    d = [np.mean(y_data[s:s + 2]) for s in range(differ_index[1] + 1, d_len)]
    mean_values2 = np.append(mean_values2, d)

    x_ = x_data[start_index:snap_index - 1]
    y = mean_values2[start_index:snap_index - 1]

    """dim = 7
    A = np.zeros((len(x), dim*2))
    for k in range(0, dim):
        A[:, k] = np.cos(k * np.pi * x)
        A[:, k + dim] = np.sin(k * np.pi * x)"""

    """A = np.vstack([1 ** x, x ** 3, x ** 2, np.cos(x), x, np.ones_like(x), 1 / x]).T
    b = y
    # Použijte metodu nejmenších čtverců k nalezení koeficientů
    coefficients, resid, rank, s = np.linalg.lstsq(A, b, rcond=None)
    a1, a2, a3, a4, a5, a6, a7 = coefficients

    # Vytvořte body pro fitovanou kvadratickou funkci
    y_ = a1 * (1 ** x) + a2 * x ** 3 + a3 * x ** 2 + a4 * np.cos(x) + a5 * x + a6 + a7 / x"""
    """y_ = np.zeros_like(x)
    len_c = len(coefficients)
    for k in range(len_c):
        y_ += np.cos(coefficients[k] * np.pi * x)
        y_ += np.sin(coefficients[k] * np.pi * x)"""

    y_ = poly_function(x_, np.polyfit(x_, y, 5))
    x_fit = np.append(x_fit, x_)
    y_fit = np.append(y_fit, y_)

    # Výpočet kvadratické chyby
    mse = np.mean((y - y_) ** 2)
    print("\nMean Squared Error (MSE):", mse)

    x_ = x_data[snap_index + 1:]
    y = mean_values2[snap_index + 1:]
    y_ = poly_function(x_, np.polyfit(x_, y, 5))
    x_fit = np.append(x_fit, x_)
    y_fit = np.append(y_fit, y_)

    # Výpočet kvadratické chyby
    mse = np.mean((y - y_) ** 2)
    print("Mean Squared Error (MSE):", mse)

    """st = np.where(y_fit[:snap_index] < 0)[0][-1]
    y_fit[:st] = 0"""

else:
    mean_values2 = np.array([np.mean(y_data[s:s + 2]) for s in range(d_len)])
    mean_values2[0] = y_data[0]
    x_fit = np.append(x_fit, x_data[start_index:])
    y_fit = np.append(y_fit, poly_function(x_data[start_index:],
                                           np.polyfit(x_data[start_index:], mean_values2[start_index:], 5)))

    # Výpočet kvadratické chyby
    mse = np.mean((mean_values2[start_index:] - y_fit[start_index:]) ** 2)
    print("\nMean Squared Error (MSE):", mse)

"""print(start_index3)

start_index = min(max(int(np.where(
    np.convolve(
        np.where(np.abs(
            np.array([np.mean(y_data[s - z2:s + z2]) for s in range(zero_stage, data_length // 3)])[zero_stage:] -
            np.median(y_data[:zero_stage * 2 + 1])) >= np.mean(
            np.array([np.std(y_data[s - z2:s + z2]) for s in range(zero_stage, data_length // 30)])[zero_stage:] * 2),
                 1, 0), np.ones(data_length // 30), mode='valid') == (data_length // 30))[0][0]), 0),
                  data_length)"""
print("\nStart index:", start_index)
# start_index = start_index3

"""# y_data -= y_data[:zero_stage].mean()
zero_force = y_data[:zero_stage].mean()

# Definice jádra pro konvoluci (5 po sobě jdoucích hodnot)
kernel = np.ones(5) / 5

# Aplikace konvoluce na signál
gradient = np.convolve(y_data[3:], kernel, mode='valid')
# Nastavení práhu pro začátek stoupajícího trendu
threshold = 0.1
# Indexy, kde derivace překročí práh
start_indices = (np.where(gradient > threshold)[0])[0] + 3

# Vytvoříme si klouzavý průměr s oknem šířky 5
window_size = 3
cumulative_sum = np.cumsum(y_data)
cumulative_sum[window_size:] = cumulative_sum[window_size:] - cumulative_sum[:-window_size]
# Najděte kladná čísla
positive_numbers = y_data[window_size // 2:int(window_size * 1.5)][y_data[window_size // 2:int(window_size * 1.5)] > 0]
min_positive = np.min(positive_numbers)
b = y_data[window_size - 1:window_size]
# Porovnáme průměry 5 po sobě jdoucích čísel s hodnotami na daných pozicích
condition = (cumulative_sum[window_size - 1:] / window_size) < min_positive

# Najdeme pozice, kde podmínka platí
start_position = (np.where(condition)[0] + window_size - 1)[-1]
print("Pozice, kde je průměr 5 po sobě jdoucích čísel větší než 0:", start_position)
x_data = x_data - x_data[start_position]"""

photo_x = x_data[photo_indexes]
photo_y = y_data[photo_indexes]

"""degree = 5  # Stupeň polynomu, můžete zvolit jiný stupeň podle vašich potřeb
coefficients = np.polyfit(x_data, y_data, degree)
poly_model = np.poly1d(coefficients)
x_fit = np.linspace(min(x_data), max(x_data), 1000)
y_fit = poly_model(x_fit)


def model_function(x, a, b, c):
    return a * x ** 2 + b * x + c


popt, _ = curve_fit(model_function, x_data, y_data)
x_fit2 = np.linspace(min(x_data), max(x_data), 1000)  # Pro plynulý průběh křivky
y_fit2 = model_function(x_fit, *popt)

x_data_reshaped = x_data.reshape(-1, 1)  # Potřebujeme dvourozměrný vstup
model = LinearRegression()
model.fit(x_data_reshaped, y_data)

x_fit3 = np.linspace(min(x_data), max(x_data), 1000).reshape(-1, 1)
y_fit3 = model.predict(x_fit3)"""

"""window_size = 1  # Velikost klouzavého okna
# Vytvoření váhového jádra (rozdělení váhového okna)
weights = np.ones(window_size) / window_size
extended_data = np.concatenate((y_data, np.repeat(y_data[-1], window_size)))
smoothed_data = np.convolve(extended_data, weights, mode='valid')"""

plt.figure(figsize=(10, 4))

# plt.scatter(x_data[start_index3], y_data[start_index3], label='Start position', c="red", marker='X', s=100)
plt.scatter(x_data[start_index], y_data[start_index], label='Start position', c="black", marker='X', s=100)
plt.scatter(x_data[snap_index - 1], y_data[snap_index - 1], label='Snap', c="green", marker='X', s=100)
plt.axhline(y=start_value_y, color='firebrick', linestyle='--', zorder=10)
plt.plot(x_data[:len(mean_values2)], mean_values2, color='r', linestyle='--', zorder=10)

plt.plot(x_data, y_data, label='Original data')
# plt.plt(x_data, y1, label=' Strain gage 1', color="orange", linestyle='-.')
# plt.plt(x_data, y2, label=' Strain gage 2', color="coral", linestyle='-.')
"""plt.plt(x_fit, y_fit, label='Fitted Curve - scipy polyfit', color='red')
plt.plt(x_fit2, y_fit2, label='Fitted Curve - scipy curve_fit', color='green')
plt.plt(x_fit3, y_fit3, label='Fitted Curve - sklearn', color='orange')"""
"""plt.plt(x_data, smoothed_data[:-1], label=f'Smoothed\n ({window_size}-point moving average)',
         color='royalblue')"""
plt.scatter(photo_x, photo_y, label='taken photos', c='blue')
# plt.scatter(x_data[start_position], y_data[start_position], label='Start position', c="navy")
# plt.scatter(x_data[start_indices], y_data[start_indices], label='Start position', c="green")
plt.plot(x_data, y_data1, linestyle=':')
plt.plot(x_data, y_data2, linestyle=':')

plt.plot(x_fit, y_fit, linestyle='--', c='black', label='polyfit')

x_range_start, x_range_end = plt.gca().get_xlim()
y_range_start, y_range_end = plt.gca().get_ylim()

# Nastavení stejného poměru os x a y při přibližování
plt.gca().set_aspect(((x_range_end - x_range_start) / (y_range_end - y_range_start)) / 2.5)
plt.xlabel('Distance [mm]')
plt.ylabel('Force [N]')
plt.legend(fontsize=8, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
plt.grid()
# plt.axis('equal')
plt.tight_layout()

fig, ax = plt.subplots(figsize=(6, 3.5))

x_data_plot = x_data - x_data[0]

ax.plot(x_data_plot, y_data1, ls="-", lw=1, c="dodgerblue", zorder=10, alpha=1)
ax.plot(x_data_plot, y_data2, ls="-", lw=1, c="tab:orange", zorder=11, alpha=1)

ax.grid(color="lightgray", linewidth=0.5, zorder=0)
for axis in ['top', 'right']:
    ax.spines[axis].set_linewidth(0.5)
    ax.spines[axis].set_color('lightgray')

if ax.get_xlim()[1] % ax.get_xticks()[-1] == 0:
    ax.spines['right'].set_visible(False)
if ax.get_ylim()[1] % plt.gca().get_yticks()[-1] == 0:
    ax.spines['top'].set_visible(False)

ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())

ax.tick_params(axis='both', which='minor', direction='in', width=0.5, length=2.5, zorder=5, color="black")
ax.tick_params(axis='both', which='major', direction='in', width=0.8, length=5, zorder=5, color="black")

ax.set_xlabel('Displacement [mm]')
ax.set_ylabel('Force [N]')

ax.set_aspect('auto', adjustable='box')
plt.tight_layout()
name = file_path.split("\\")[-1].split(".")[0]
plt.savefig(f'.outputs/support_plot_{name}.{file_type}', format=file_type, dpi=out_dpi, bbox_inches='tight')

plt.show()
