import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Načtení dat ze souboru CSV
file_path = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\data\data_csv\T01_08-I_1s.csv'

photo_path = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos'
df = pd.read_csv(file_path)  # DATAFRAME

# Vytvoření grafu
x_data = np.hstack(([0], df.iloc[:, 0].values))  # První sloupec jako osa x - posun
y_data = np.hstack(([0], df.iloc[:, 1].values))  # - celková síla
times = np.hstack(([0], df.iloc[:, 2].values))

f = os.path.basename(file_path).replace(".csv", "")
n = os.path.join(photo_path, f, "original")
t = [os.path.getmtime(os.path.join(n, p)) for p in os.listdir(n) if
     p.lower().endswith(".jpg") and not p.startswith("0")][2:]
check_t = [int(t[i + 1] - t[i]) for i in range(len(t) - 1)]
print(f, ":", check_t[:])
if 0 in check_t:
    print("byli pořízeny stejné fotografie")
if 2 in check_t:
    print("byla mezera")
t = [int(t[i] - t[0]) for i in range(len(t))]
periods = range(len(t))

# Zjištění indexů nejbližších hodnot z prvního vektoru k hodnotám ve druhém vektoru
closest_indices_t = np.argsort(np.abs(times[:, np.newaxis] - t), axis=0)[0]
closest_indices = np.argsort(np.abs(times[:, np.newaxis] - periods), axis=0)[0]
found_num = times[closest_indices]

plt.figure(figsize=(10, 4))

plt.plot(x_data, y_data, label='Original data', zorder=3)

plt.scatter(x_data[closest_indices], y_data[closest_indices], label='taken photos', c='blue', zorder=2)
plt.scatter(x_data[closest_indices_t], y_data[closest_indices_t], label='taken photos', c='red', zorder=2)

# Nastavení stejného poměru os x a y při přibližování
# #x_range_start, x_range_end = plt.gca().get_xlim()
# #y_range_start, y_range_end = plt.gca().get_ylim()
# plt.gca().set_aspect(((x_range_end - x_range_start) / (y_range_end - y_range_start)) / 2.5)
plt.xlabel('Distance [mm]')
plt.ylabel('Force [N]')
plt.legend(fontsize=8, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
plt.grid()
# plt.axis('equal')
plt.tight_layout()
plt.show()
