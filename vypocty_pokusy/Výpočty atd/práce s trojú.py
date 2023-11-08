import numpy as np
import matplotlib.pyplot as plt


def size_up_triangle(coordinates, k):
    triangle_center = tuple(sum(x) / 3 for x in zip(*coordinates))  # Výpočet středu trojúhelníka

    # Posunutí bodů trojúhelníka tak, aby střed byl v počátku souřadnicového systému
    moved_coordinates = [(x - triangle_center[0], y - triangle_center[1]) for x, y in coordinates]

    # Zvětšení trojúhelníka změnou měřítka
    new_coordinates = [(k * x, k * y) for x, y in moved_coordinates]

    # Vrácení bodů na jejich původní místo (posunutí zpět)
    new_coordinates = [(x + triangle_center[0], y + triangle_center[1]) for x, y in new_coordinates]

    return new_coordinates, triangle_center


# Příklad použití
# Definujeme souřadnice vrcholů trojúhelníku jako seznam tuplů (x, y)
coordinates_triangle = [(-0.5, 0), (2, 0.5), (-1, -3)]

# Zvětšíme trojúhelník o faktor 1.5 kolem jeho středu a spočítáme střed trojúhelníka
new_coordinates_triangle, center = size_up_triangle(coordinates_triangle, 1.5)

print("Původní souřadnice trojúhelníku:", coordinates_triangle)
print("Zvětšený trojúhelník:", new_coordinates_triangle)
print("Střed trojúhelníka:\n", center)

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.scatter(center[0], center[1], zorder=3)
for triangle in np.array(coordinates_triangle).reshape(1, -1, 2):
    plt.plot(np.append(triangle[:, 0], triangle[0, 0]), np.append(triangle[:, 1], triangle[0, 1]),
             marker='o', linestyle='-', color='g')
    plt.fill(triangle[:, 0], triangle[:, 1], 'g', alpha=0.1)
for triangle in np.array(new_coordinates_triangle).reshape(1, -1, 2):
    plt.plot(np.append(triangle[:, 0], triangle[0, 0]), np.append(triangle[:, 1], triangle[0, 1]),
             marker='o', linestyle='-', color='r')
    plt.fill(triangle[:, 0], triangle[:, 1], 'r', alpha=0.1)
plt.tight_layout()

# Příklad definice matic
triangles = np.array([[[0, 0], [1, -1], [4, 3]],  # první trojúhelník,
                      [[7, 3], [10, 10], [11, 3]],  # druhý trojúhelník
                      [(4.5, 0), (8, 0.5), (6, -3)]
                      ])

point = np.array([8, 4])  # x a y souřadnice bodu


def is_inside_triangle(triangle, point):
    v0, v1, v2 = triangle
    x, y = point

    # Výpočet barycentrických souřadnic
    denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    if denom != 0:
        alpha = ((v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])) / denom
        beta = ((v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])) / denom
        gamma = 1 - alpha - beta
    else:
        return False

    # Kontrola, zda bod leží uvnitř trojúhelníku
    if 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1:
        return True
    else:
        return False


true_index = []
# Projděte všechny trojúhelníky a zkontrolujte, zda se bod nachází uvnitř některého z nich
for i, triangle in enumerate(triangles):
    if is_inside_triangle(triangle, point):
        print(f"Bod se nachází uvnitř trojúhelníku {i + 1}.")
        true_index.append(i)
        # break
if not true_index:
    print("Bod se nenachází uvnitř žádného trojúhelníku.")

plt.subplot(122)
plt.scatter(point[0], point[1], c='firebrick', marker='o', zorder=3)
for triangle in triangles:
    plt.plot(np.append(triangle[:, 0], triangle[0, 0]), np.append(triangle[:, 1], triangle[0, 1]),
             marker='.', linestyle='-', color='b')
    plt.fill(triangle[:, 0], triangle[:, 1], 'b', alpha=0.1)
plt.tight_layout()
plt.grid(True)
plt.show()
