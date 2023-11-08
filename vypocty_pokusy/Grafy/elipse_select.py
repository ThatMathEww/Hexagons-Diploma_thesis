import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import EllipseSelector
from matplotlib.path import Path


def select_callback(eclick, erelease):
    """
    Callback for line selection.

    *eclick* and *erelease* are the press and release events.
    """
    pass


points_to_test = np.array([[0.1, 0.4], [0.8, 0.3], [0.5, 0.5], [0.1, 0.7], [0.4, 0.6]])

fig = plt.figure(layout='constrained')
axs = fig.subplots(1)

N = 100000  # If N is large one can see improvement by using blitting.
x = np.linspace(0, 10, N)

selectors = []

selector = EllipseSelector(axs, select_callback, useblit=True,
                           button=[1, 3],  # disable middle button
                           minspanx=5, minspany=5,
                           spancoords='pixels',
                           interactive=True)

plt.scatter(points_to_test[:, 0], points_to_test[:, 1], c='orange', marker='o',
            label='Testované body')  # Vykreslení testovaných bodů

plt.gca().set_xlim(0, 1)
plt.gca().set_ylim(0, 1)

plt.show()

center = selector.center
geometry = selector.geometry
extents = selector.extents
corners = selector.corners
edge_cor = selector.edge_centers
print("\nCenter:", center, "\ngeometry:", geometry, "\nextents:", extents,
      "\ncorners:", corners, "\nedge_centers:", edge_cor)

g = np.array((geometry[1], geometry[0]))

plt.figure()
plt.scatter(g[0], g[1])
plt.scatter(center[0], center[1])
plt.scatter(corners[0], corners[1])
plt.scatter(edge_cor[0], edge_cor[1])
plt.fill(geometry[1], geometry[0], facecolor='skyblue', edgecolor='none', alpha=0.5)
plt.show()

polygon_vertices = g.T

polygon_path = Path(polygon_vertices)

points_inside_polygon = []

for point in points_to_test:
    if polygon_path.contains_point(point):
        points_inside_polygon.append(point)

if len(points_inside_polygon) > 0:
    points_inside_polygon = np.array(points_inside_polygon)
    plt.fill(polygon_vertices[:, 0], polygon_vertices[:, 1], color="skyblue")  # Vykreslení polygonu
    plt.scatter(points_to_test[:, 0], points_to_test[:, 1], c='red', marker='o',
                label='Testované body')  # Vykreslení testovaných bodů
    plt.scatter(points_inside_polygon[:, 0], points_inside_polygon[:, 1], c='black', marker='x',
                label='Body uvnitř polygonu')  # Vykreslení bodů uvnitř polygonu
    plt.legend()
    plt.show()
