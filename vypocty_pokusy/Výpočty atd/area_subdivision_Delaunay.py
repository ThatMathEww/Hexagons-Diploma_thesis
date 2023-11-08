import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import timeit

# Seznam pro ukládání bodů oblasti
polygon_points = []


def subdivide_triangulation(tri):
    new_points = []
    new_triangles = []

    for triangle in tri.simplices:
        # Získání vrcholů existujícího trojúhelníka
        v1, v2, v3 = triangle

        # Vytvoření nových bodů (např. středy stran)
        mid12 = np.average([tri.points[v1], tri.points[v2]], axis=0)
        mid23 = np.average([tri.points[v2], tri.points[v3]], axis=0)
        mid31 = np.average([tri.points[v3], tri.points[v1]], axis=0)

        # Přidání nových bodů do seznamu vrcholů
        new_points.extend([mid12, mid23, mid31])

        # Vytvoření nových trojúhelníků
        t1 = [v1, len(new_points) - 3, len(new_points) - 1]
        t2 = [len(new_points) - 3, v2, len(new_points) - 2]
        t3 = [len(new_points) - 1, len(new_points) - 2, v3]

        # Přidání nových trojúhelníků do seznamu
        new_triangles.extend([t1, t2, t3])

    # Aktualizace seznamu vrcholů
    new_points = np.vstack((tri.points, new_points))

    # Vytvoření nové triangulace
    new_tri = Delaunay(new_points)

    return new_tri


def subdivide_triangulation2(tri):
    # Vytvoření nových bodů (např. středy stran)
    midpoints = np.average(tri.points[tri.simplices[:, [0, 1, 1, 2, 2, 0]]], axis=1)

    # Přidání nových bodů do seznamu vrcholů
    new_points = np.concatenate((tri.points, midpoints))

    # Vytvoření nové triangulace
    new_tri = Delaunay(new_points)

    return new_tri


# Funkce pro interaktivní zaznamenání bodů na grafu
def onclick(event):
    if event.button == 1:
        x, y = event.xdata, event.ydata
        polygon_points.append([x, y])
        plt.plot(x, y, 'ro')
        plt.draw()


# Vytvoření prázdného grafu
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-10, 10)  # Přizpůsobte rozsah grafu vašim potřebám
ax.set_ylim(-10, 10)  # Přizpůsobte rozsah grafu vašim potřebám
plt.title('Vyberte body oblasti')
x = [-10, 10]
y = [0, 0]
ax.plt(x, y, color='black', linewidth=0.8, linestyle='dashed')
y = [-10, 10]
x = [0, 0]
ax.plt(x, y, color='black', linewidth=0.8, linestyle='dashed')
plt.tight_layout()
plt.grid()

# Přiřazení funkce onclick ke kliknutí myší
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# Zobrazení grafu
plt.show()

# polygon_points = np.array([[0, 0], [7, 0], [7, 9], [4.5, 10], [0, 9]])

# Vytvoření seznamu vrcholů oblasti
polygon_points = np.array(polygon_points)

# Výpočet původní triangulace
tri = Delaunay(polygon_points)

# Počet podrozdělení
num_subdivisions = 2

tri1, tri2 = tri, tri

# Podrozdělení triangulace
for _ in range(num_subdivisions):
    tri1 = subdivide_triangulation(tri1)

for _ in range(num_subdivisions):
    tri2 = subdivide_triangulation2(tri2)

# Vykreslení mnohoúhelníku s trojúhelníkovými elementy
plt.figure()
plt.gca().set_aspect('equal')
plt.triplot(tri1.points[:, 0], tri1.points[:, 1], tri1.simplices.copy())
plt.plot(tri1.points[:, 0], tri1.points[:, 1], 'o')
plt.tight_layout()

plt.figure()
plt.gca().set_aspect('equal')
plt.triplot(tri2.points[:, 0], tri2.points[:, 1], tri2.simplices.copy())
plt.plot(tri2.points[:, 0], tri2.points[:, 1], 'o')
plt.tight_layout()
plt.show()
