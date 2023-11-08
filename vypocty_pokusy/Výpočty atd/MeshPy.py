import meshpy.triangle as triangle
import numpy as np
import numpy.linalg as la
from six.moves import range


def round_trip_connect(start, end):
    return [(i, i + 1) for i in range(start, end)] + [(end, start)]


def main():
    points = []
    points.extend(
        (0.3 * np.cos(angle) + 0.5, 0.3 * np.sin(angle)) for angle in np.linspace(0, 2 * np.pi, 30, endpoint=False))
    facets = round_trip_connect(0, len(points) - 1)

    circ_start = len(points)
    points.extend(
        (1 * np.cos(angle), 1 * np.sin(angle))
        for angle in np.linspace(0, 2 * np.pi, 30, endpoint=False))

    facets.extend(round_trip_connect(circ_start, len(points) - 1))

    def needs_refinement(vertices, area):
        bary = np.sum(np.array(vertices), axis=0) / 3
        max_area = 0.001 + (la.norm(bary, np.inf) - 1) * 0.01
        return bool(area > max_area)

    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_holes([(0.7, 0)])
    info.set_facets(facets)

    mesh = triangle.build(info, max_volume=6e-3)  # , refinement_func=needs_refinement)
    # triangle.write_gnuplot_mesh("circle2.dat", mesh)

    mesh_points = np.array(mesh.points)
    mesh_tris = np.array(mesh.elements)
    mesh_facets = np.array(mesh.facets)

    import matplotlib.pyplot as pt
    pt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris)
    pt.show()


if __name__ == "__main__":
    main()

"""import numpy as np
import meshpy.triangle as triangle
import matplotlib.pyplot as plt


def round_trip_connect(start, end):
    result = []
    for i in range(start, end):
        result.append((i, i + 1))
    result.append((end, start))
    return result


# Definice bodů okraje oblasti
points = np.array([
    [0.0, 0.0],
    [0.05, -0.05],
    [1.0, 0.0],
    [0.9, 0.4],
    [1.0, 1.0],
    [0.0, 1.0]
])

# Definice otvoru (negativní oblasti)
hole_points = np.array([
    [0.3, 0.5],
    [0.5, 0.5],
    [0.4, 0.7]
])

combined_points = np.concatenate((points, hole_points))

# Nastavení parametrů pro generování sítě
mesh_info = triangle.MeshInfo()
mesh_info.set_points(points)
mesh_info.set_facets(round_trip_connect(0, len(points) - 1))
# mesh_info.set_holes(hole_points)

# Generování sítě
mesh = triangle.build(mesh_info, max_volume=0.1, min_angle=35, quality_meshing=True, allow_boundary_steiner=True,
                      attributes=True)

mesh_points = np.array(mesh.points)
mesh_tris = np.array(mesh.elements)

# Vykreslení sítě
plt.figure()
plt.gca().set_aspect('equal')
plt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris, linewidth=0.5, color='k')
plt.plt(mesh_points[:, 0], mesh_points[:, 1], 'o', markersize=3, color='r')
plt.tight_layout()
plt.show()
"""
