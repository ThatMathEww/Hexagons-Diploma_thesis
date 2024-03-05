import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


# Funkce pro aktualizaci grafu
def update(frame):
    global x_data, y_data

    # Přidání nových dat do souboru
    with open("stock.txt", "a") as file:
        new_data_point = np.random.uniform(-10, 10) * np.sin(frame / 10) + np.sqrt(frame)
        file.write(f"{frame},{new_data_point}\n")

    # Načtení nových dat ze souboru
    with open("stock.txt", "r") as file:
        timestamp, value = np.float64(file.readlines()[-1].strip().split(","))

    # Přidání nového datového bodu
    x_data.append(timestamp)
    y_data.append(value)

    # Oříznutí dat na posledních 100 hodnot
    x_data = x_data[-100:]
    y_data = y_data[-100:]

    # Aktualizace grafu
    line.set_data(x_data, y_data)
    # Pokud je aktuální hodnota větší než maximální hodnota na ose y, aktualizuj rozsah osy y
    if not ax.get_ylim()[0] < value < ax.get_ylim()[1]:
        ax.set_ylim(np.min(y_data) - 5, np.max(y_data) + 5)
    # Pokud je aktuální hodnota větší než maximální hodnota na ose x, aktualizuj rozsah osy x
    if not ax.get_xlim()[0] < timestamp < ax.get_xlim()[1]:
        ax.set_xlim(np.min(x_data) + 5 if timestamp >= 100 else 0, np.max(x_data) + 5)
    return line,


# Nastavení grafu
fig, ax = plt.subplots()
line, = ax.plot([], [], label='Data')
ax.set_xlim(0, 100)
ax.set_ylim(2, 5)
ax.set_xlabel('Čas')
ax.set_ylabel('Hodnota')
ax.legend()

# Inicializace prázdného grafu
x_data = []
y_data = []
line.set_data(x_data, y_data)

# Přidání nových dat do souboru
open("stock.txt", "w").close()  # Vymazání obsahu souboru

# Vytvoření animace
ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)

# Zobrazení grafu
plt.show()
