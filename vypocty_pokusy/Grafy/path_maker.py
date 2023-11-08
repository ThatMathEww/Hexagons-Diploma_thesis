import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
import numpy as np

# Inicializace prázdných seznamů pro ukládání bodů
body_x = []
body_y = []
stred_x = []
stred_y = []


# Funkce pro vykreslení bodu na grafu
def vykresli_bod(event):
    if event.button == 1:  # Levé tlačítko myši
        plt.scatter(event.xdata, event.ydata, c='b')  # Vykreslení bodu
        body_x.append(event.xdata)
        body_y.append(event.ydata)

        # Vykreslení úseček spojujících body
        if len(body_x) > 0:
            plt.plot(body_x[-2:], body_y[-2:], c='b')

        # Výpočet a vykreslení středu úsečky
        if len(body_x) > 1:
            stred_x.append((body_x[-1] + body_x[-2]) / 2)
            stred_y.append((body_y[-1] + body_y[-2]) / 2)
            plt.scatter(stred_x[-1], stred_y[-1], c='r')

        plt.draw()

# Funkce pro smazání bodů
def smaz_body(event):
    if event.key == 'c':
        del body_x[:]
        del body_y[:]
        del stred_x[:]
        del stred_y[:]
        plt.cla()
        plt.draw()
        plt.title("Smazání čar pomocí klávesy 'c'")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)


# Nastavení grafu
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

plt.title("Smazání čar pomocí klávesy 'c'")
# Připojení událostí pro klikání myší a stisk klávesy
fig.canvas.mpl_connect('button_press_event', vykresli_bod)
fig.canvas.mpl_connect('key_press_event', smaz_body)

# Spuštění zobrazení
plt.show()
