import numpy as np
import matplotlib.pyplot as plt

def vytvor_grid(maska, vzdalenost_bodu):
    # Získání souřadnic oblasti s hodnotou 1 v masce
    y, x = np.where(maska == 1)

    # Vytvoření pravidelného gridu bodů s danou vzdáleností
    grid_y, grid_x = np.meshgrid(
        np.arange(y.min(), y.max() + vzdalenost_bodu, vzdalenost_bodu),
        np.arange(x.min(), x.max() + vzdalenost_bodu, vzdalenost_bodu)
    )

    # Vytvoření nové masky pro body vytvořeného gridu
    nova_maska = np.zeros_like(maska)
    nova_maska[grid_y, grid_x] = 1

    return nova_maska

# Příklad použití
# Můžete si vytvořit vlastní matici masky s oblastmi hodnot 0 a 1
# Například:
maska = np.zeros((100, 100))
maska[20:50, 30:70] = 1

# Vytvoření gridu s body vzdálenými 5 jednotek
nova_maska = vytvor_grid(maska, 5)

# Vizualizace původní a nové masky
plt.subplot(1, 2, 1)
plt.imshow(maska, cmap='gray', interpolation='none')
plt.title('Původní Maska')

plt.subplot(1, 2, 2)
plt.imshow(nova_maska, cmap='gray', interpolation='none')
plt.title('Nová Maska s Gridem')

plt.show()
