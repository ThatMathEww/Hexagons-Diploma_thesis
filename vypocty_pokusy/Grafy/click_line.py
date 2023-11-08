import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Definujeme prázdný graf
fig, ax = plt.subplots()
points, = ax.plot([], [], 'ro')  # Definujeme prázdné body

# Seznam pro uchování souřadnic bodů
coordinates = []

# Seznam pro uchování čar
lines = []

# Index nejbližšího bodu
nearest_index = None


# Funkce pro aktualizaci bodů a čar na grafu
def update_plot():
    x = [coord[0] for coord in coordinates]
    y = [coord[1] for coord in coordinates]
    points.set_data(x, y)

    for line in lines:
        line.remove()

    lines.clear()
    for i in range(len(coordinates) - 1):
        line = Line2D([coordinates[i][0], coordinates[i + 1][0]], [coordinates[i][1], coordinates[i + 1][1]],
                      color='blue')
        lines.append(ax.add_line(line))

    plt.draw()


# Funkce pro přidání bodu na zadané souřadnice
def add_point(x, y):
    coordinates.append((x, y))
    update_plot()


# Funkce pro smazání bodu
def remove_point(index):
    try:
        del coordinates[index]
        update_plot()
    except TypeError:
        pass


# Funkce pro přemístění bodu
def move_point(index, x, y):
    coordinates[index] = (x, y)
    update_plot()


# Funkce, která se volá při kliknutí na graf
def onclick(event):
    if event.button == 1:  # Levé tlačítko myši
        add_point(event.xdata, event.ydata)
    elif event.button == 3 and event.key == 'shift':  # Pravé tlačítko myši + Shift
        index = find_nearest_point_delete(event.xdata, event.ydata)
        remove_point(index)


"""# Funkce, která se volá při pohybu myší
def onmove(event):
    if event.inaxes is ax and event.button == 1:  # Levé tlačítko myši uvnitř grafu
        index = find_nearest_point_move(event.xdata, event.ydata)
        move_point(index, event.xdata, event.ydata)"""


# Funkce pro nalezení nejbližšího bodu k daným souřadnicím
def find_nearest_point_move(x, y):
    distances = [(i, (coord[0] - x) ** 2 + (coord[1] - y) ** 2) for i, coord in enumerate(coordinates)]
    valid_distances = [(i, dist) for i, dist in distances if dist <= 0.075 ** 2]

    min_distance = min(valid_distances, key=lambda d: d[1])
    return min_distance[0]


def find_nearest_point_delete(x, y):
    distances = [(i, (coord[0] - x) ** 2 + (coord[1] - y) ** 2) for i, coord in enumerate(coordinates)]
    valid_distances = [(i, dist) for i, dist in distances if dist <= 0.005 ** 2]
    if valid_distances:
        min_distance = min(valid_distances, key=lambda d: d[1])
        return min_distance[0]
    else:
        return None


# Funkce, která se volá při pohybu myší
def onmove(event):
    global nearest_index
    if event.inaxes is ax:
        if event.button == 1:  # Levé tlačítko myši uvnitř grafu
            index = find_nearest_point_move(event.xdata, event.ydata)
            move_point(index, event.xdata, event.ydata)
        elif event.button == 3 and event.key == 'control':  # Pravé tlačítko myši + Ctrl
            if nearest_index is not None:
                move_point(nearest_index, event.xdata, event.ydata)


# Funkce, která se volá při pohybu myší při stisknuté klávese Ctrl
def onmove_with_ctrl(event):
    global nearest_index
    if event.inaxes is ax and event.button == 3 and event.key == 'control':  # Pravé tlačítko myši + Ctrl
        nearest_index = find_nearest_point_move(event.xdata, event.ydata)


# Připojení událostí k grafu
cid = fig.canvas.mpl_connect('button_press_event', onclick)
cid = fig.canvas.mpl_connect('motion_notify_event', onmove)
cid = fig.canvas.mpl_connect('motion_notify_event', onmove_with_ctrl)

# Zobrazení grafu
plt.show()
