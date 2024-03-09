import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time


# Funkce pro aktualizaci grafu
def update(frame):
    global x_data, y_data1, y_data2

    with open("stock.txt", "a") as file:
        new_data_point1 = np.random.uniform(-10, 10) * np.sin(frame / 10) + np.sqrt(frame) + (
                np.random.random() * -200) + 100
        new_data_point2 = np.random.uniform(-10, 10) * np.sin(frame / 10) + np.sqrt(frame) + (
                np.random.random() * -200) + 100
        file.write(f"{time.time() - start_time},{new_data_point1},{new_data_point2}\n")

    # Načtení nových dat ze souboru
    with open("stock.txt", "r") as file:
        input_data = np.float64(file.readlines()[-1].strip().split(","))

    # Přidání nového datového bodu
    x_data.append(input_data[0])
    y_data1.append(input_data[1])
    y_data2.append(input_data[2])

    # Oříznutí dat na posledních 100 hodnot
    x_data = x_data[-150:]
    y_data1 = y_data1[-150:]
    y_data2 = y_data2[-150:]

    # Aktualizace grafu
    line1.set_data(x_data, y_data1)
    line2.set_data(x_data, y_data2)
    # Pokud je aktuální hodnota větší než maximální hodnota na ose y, aktualizuj rozsah osy y
    if not (ax.get_ylim()[0] + 5 < np.max(input_data[1:]) < ax.get_ylim()[1] - 5):
        ax.set_ylim(np.min(np.vstack((y_data1, y_data2))) - 5, np.max(np.vstack((y_data1, y_data2))) + 5)
    # Pokud je aktuální hodnota větší než maximální hodnota na ose x, aktualizuj rozsah osy x
    if not ax.get_xlim()[0] < input_data[0] < ax.get_xlim()[1]:
        ax.set_xlim(np.min(x_data) + 5 if input_data[0] >= x_max_lim else 0, np.max(x_data))
    return line1, line2


# Nastavení grafu
fig, ax = plt.subplots()
x_data = []
y_data1 = []
y_data2 = []

x_max_lim = 20
y_max_lim = 5
y_min_lim = 2

line1, = ax.plot(x_data, y_data1, label='Podporová síla 1', c='dodgerblue')
line2, = ax.plot(x_data, y_data2, label='Podporová síla 2', c='orange')
ax.set_xlim(0, x_max_lim)
ax.set_ylim(y_min_lim, y_max_lim)
ax.set_xlabel('Čas [s]')
ax.set_ylabel('Síla [N]')
ax.legend(loc='center', bbox_to_anchor=(0.5, 1.075), ncol=2)

# Přidání nových dat do souboru
open("stock.txt", "w").close()  # Vymazání obsahu souboru

start_time = time.time()

# Vytvoření animace
ani = FuncAnimation(fig, update, interval=150, cache_frame_data=False)

# Zobrazení grafu
plt.show()
