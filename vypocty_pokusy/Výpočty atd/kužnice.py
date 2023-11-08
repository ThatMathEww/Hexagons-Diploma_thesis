import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from scipy.optimize import least_squares, minimize
import cv2

"""def get_indexes(vect):
    size = len(vect)
    if size > 5:
        index_1, index_2 = int(size * 0.2), int(size * 0.4)
        index_3, index_4 = int(size * 0.6), int(size * 0.8)

    else:
        index_1 = index_3 = 0
        index_2 = index_4 = -1

    vector_member = [vect[0], vect[index_1], vect[index_2], vect[index_3], vect[index_4], vect[-1]]

    return vector_member


# Příklad použití
vector = np.array([1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11])

middle_element = get_indexes(vector)
print(middle_element)"""


def distance_error(point, distances, known_points):
    x, y = point
    error = 0
    for i in range(len(distances)):
        x_k, y_k = known_points[i]
        d = distances[i]
        error += (np.sqrt((x - x_k) ** 2 + (y - y_k) ** 2) - d) ** 2
    return error


def find_fourth_point(distances, known_points):
    initial_guess = [0, 0]  # Počáteční odhad polohy čtvrtého bodu
    result1 = least_squares(distance_error, initial_guess, args=(distances, known_points))
    # result2 = minimize(distance_error, initial_guess, args=(distances, known_points))
    x1, y1 = result1.x
    # x2, y2 = result2.x
    return x1, y1,  # x2, y2


def find_point(distances, known_points):
    A = np.array([[2 * (known_points[0][0] - known_points[1][0]), 2 * (known_points[0][1] - known_points[1][1])],
                  [2 * (known_points[0][0] - known_points[2][0]), 2 * (known_points[0][1] - known_points[2][1])]])
    b = np.array([distances[1] ** 2 - distances[0] ** 2 + known_points[0][0] ** 2 - known_points[1][0] ** 2 +
                  known_points[0][1] ** 2 - known_points[1][1] ** 2,
                  distances[2] ** 2 - distances[0] ** 2 + known_points[0][0] ** 2 - known_points[2][0] ** 2 +
                  known_points[0][1] ** 2 - known_points[2][1] ** 2])

    x, y = np.linalg.lstsq(A, b, rcond=None)[0]
    return x, y


# Příklad použití:
distances = np.array([40.925389821303206, 40.925389821303206, 42.19765230685435])  # Vzdálenosti k čtvrtému bodu
known_points = np.array([(2110.3388671875, 2117.3388671875), (2117.3388671875, 2114.393798828125),
                         (2114.393798828125, 2183.5908203125)])  # Známé body A, B, C

# print("Poloha minimaze bodu: ({}, {})".format(x2, y2))
# print("Poloha numpy bodu: ({}, {})".format(xn, yn))


def triangulate_point(distances, known_points):
    # Konstrukce triangulační matice
    A = np.zeros((len(distances), 4))
    for i, (x, y) in enumerate(known_points):
        A[i] = [x, y, 1, -distances[i]**2]

    # SVD rozklad matice A
    _, _, V = np.linalg.svd(A)

    # Řešení přes poslední sloupec V
    point_3d = V[-1, :3] / V[-1, 3]

    return point_3d[0], point_3d[1]

"""def circle_intersection(c1, c2, c3):
    # Extrahujte souřadnice středů a poloměry kružnic
    (x1, y1), r1 = c1
    (x2, y2), r2 = c2
    (x3, y3), r3 = c3
    # (x4, y4), r4 = c4
    # (x5, y5), r5 = c5
    # (x6, y6), r6 = c6

    # Vytvořte matici koeficientů soustavy rovnic
    A = np.array([
        [2 * (x2 - x1), 2 * (y2 - y1)],
        [2 * (x3 - x1), 2 * (y3 - y1)],
        # [2 * (x4 - x1), 2 * (y4 - y1)],
        # [2 * (x5 - x1), 2 * (y5 - y1)],
        # [2 * (x6 - x1), 2 * (y6 - y1)],
    ])

    # Vytvořte vektor pravých stran soustavy rovnic
    b = np.array([
        (x2 ** 2 + y2 ** 2 - r2 ** 2) - (x1 ** 2 + y1 ** 2 - r1 ** 2),
        (x3 ** 2 + y3 ** 2 - r3 ** 2) - (x1 ** 2 + y1 ** 2 - r1 ** 2),
        # (x4 ** 2 + y4 ** 2 - r4 ** 2) - (x1 ** 2 + y1 ** 2 - r1 ** 2),
        # (x5 ** 2 + y5 ** 2 - r5 ** 2) - (x1 ** 2 + y1 ** 2 - r1 ** 2),
        # (x6 ** 2 + y5 ** 2 - r6 ** 2) - (x1 ** 2 + y1 ** 2 - r1 ** 2),
    ])

    # Vypočtěte souřadnice průsečíku
    intersection = np.linalg.lstsq(A, b, rcond=None)[0]
    # intersection = np.linalg.solve(A, b)

    return intersection"""

# Definujte kružnice
sou = np.array([known_points[0, 0], known_points[0, 1], known_points[1, 0], known_points[1, 1],
                known_points[2, 0], known_points[2, 1]])
dist = np.array([distances[0], distances[1], distances[2]])

c1 = ((sou[0], sou[1]), dist[0])
c2 = ((sou[2], sou[3]), dist[1])
c3 = ((sou[4], sou[5]), dist[2])
c4 = ((sou[0], sou[1]), dist[0])
c5 = ((sou[2], sou[3]), dist[1])
c6 = ((sou[4], sou[5]), dist[2])

x = sou[0], sou[2], sou[4]
y = sou[1], sou[3], sou[5]

test_x = np.mean(x)
test_y = np.mean(y)

# Vypočítejte průsečík
x1, y1 = find_point(distances, known_points)

if np.linalg.norm(np.array([x1, y1]) - np.array([test_x, test_y])) > 1.5*np.mean(distances):
    x1, y1 = find_fourth_point(distances, known_points)

x2, y2 = triangulate_point(distances, known_points)

prusecik = np.array([x1, y1])

new_distances = np.linalg.norm(known_points - prusecik, axis=1)

print("Poloha square bodu: ({}, {})".format(x1, y1))

print(new_distances)

new_center = np.empty((0, 2))
prus = prusecik.reshape(1, 2)

for _ in range(3):
    new_center = np.append(new_center, prus, axis=0)

# Vykreslení kružnic a průsečíku
fig, ax = plt.subplots()

# Kružnice
circle1 = Circle(sou[0:2], dist[0], fill=False)
circle2 = Circle(sou[2:4], dist[1], fill=False)
circle3 = Circle(sou[4:6], dist[2], fill=False)
# circle4 = Circle(c4[0], c3[1], fill=False)

# Průsečík
# ax.plt(prusecik[0], prusecik[1], 'ro')
ax.plt(sou[0], sou[1], 'gs')
ax.plt(sou[2], sou[3], 'cs')
ax.plt(sou[4], sou[5], 'bs')
ax.plt(test_x, test_y, marker='o', color='black')

ax.plt(x1, y1, marker='o', color='red')
ax.plt(x2, y2, marker='s', color='red')
# ax.plt(xn, yn, marker='d', color='green')

# Přidání kružnic a průsečíku do grafu
ax.add_patch(circle1)
ax.add_patch(circle2)
ax.add_patch(circle3)
# ax.add_patch(circle4)

# Nastavení rozsahu grafu
# ax.set_xlim(-2110, 2120)
# ax.set_ylim(-2175, 2185)

# Zobrazení grafu
# plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.axis('equal')
plt.show()

"""def calculate_angle(point0, points):
    vektor1 = points - point0  # Rozdíl mezi prvním bodem a ostatními body
    uhly = np.arctan2(vektor1[:, 1], vektor1[:, 0])  # Výpočet úhlu vzhledem k ose x
    return uhly


def rotation_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


# Příklad použití
original_points = np.array([[0, 0], [0, 5], [3, 2.5]])
original_center = np.array([[0, 2.5]])
new_points = None

# ####################################################################
uhel_otoceni = np.pi * 0.645  # například otočení o 45 stupňů

new_points = np.dot(original_points, rotation_matrix(uhel_otoceni))

new_points[:, 0], new_points[:, 1] = new_points[:, 0] - 12.5, new_points[:, 1] - 12.65
# ####################################################################

# puvodni_teziste = np.mean(puvodni_body, axis=0).reshape(1,2)


original_angles = calculate_angle(original_points[0], original_points[1:])
new_angles = calculate_angle(new_points[0], new_points[1:])

new_position = [new_points[0] - original_points[0]]

new_center = np.dot(original_center, rotation_matrix(np.mean(original_angles - new_angles))) + new_position


plt.scatter(original_points[:, 0], original_points[:, 1], c='blue', marker='o', label='Původní body')
plt.scatter(new_points[:, 0], new_points[:, 1], c='red', marker='o', label='Původní body')
plt.scatter(original_center[:, 0], original_center[:, 1], c='blue', marker='s', label='Původní těžiště')
plt.scatter(new_center[:, 0], new_center[:, 1], c='red', marker='s', label='Nové těžiště')"""

"""
uhel_otoceni = np.pi / 4  # například otočení o 45 stupňů

matice_otoceni = np.array([[np.cos(uhel_otoceni), -np.sin(uhel_otoceni)],
                           [np.sin(uhel_otoceni), np.cos(uhel_otoceni)]])


nove_body = np.dot(puvodni_body, matice_otoceni)


# nove_body = np.array([[5, 6], [8, 9], [10, 8], [11, 2], [8, 8]])

plt.scatter(nove_body[:, 0], nove_body[:, 1], c='red', label='Nové body')


vzdalenosti = np.linalg.norm(puvodni_body - puvodni_teziste, axis=1)

kruznice = []
for bod, vzdalenost in zip(nove_body, vzdalenosti):
    kruznice.append([np.array(bod), vzdalenost])

# kruznice = np.column_stack((puvodni_body, vzdalenosti))


pruseciky = []
for i in range(len(kruznice)):
    for j in range(i + 1, len(kruznice)):
        bod1, r1 = kruznice[i]
        bod2, r2 = kruznice[j]
        d = np.linalg.norm(bod2 - bod1)
        if d <= r1 + r2:
            a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
            h = np.sqrt(abs(r1 ** 2 - a ** 2))
            x_mid = bod1[0] + a * (bod2[0] - bod1[0]) / d
            y_mid = bod1[1] + a * (bod2[1] - bod1[1]) / d
            x_int1 = x_mid + h * (bod2[1] - bod1[1]) / d
            y_int1 = y_mid - h * (bod2[0] - bod1[0]) / d
            x_int2 = x_mid - h * (bod2[1] - bod1[1]) / d
            y_int2 = y_mid + h * (bod2[0] - bod1[0]) / d
            pruseciky.append(np.array([x_int1, y_int1]))
            pruseciky.append(np.array([x_int2, y_int2]))

nove_teziste = np.mean(pruseciky, axis=0)

print(puvodni_teziste, nove_teziste)

pruseciky_int = pruseciky  # .astype(int)

jedinecne, pocet = np.unique(pruseciky_int, axis=0, return_counts=True)

nejcastejsi_index = np.argmax(pocet)

nejcastejsi_prusecik = jedinecne[nejcastejsi_index]

print(nejcastejsi_prusecik)

plt.axis('equal')
plt.legend()
plt.show()"""
