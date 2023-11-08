import pygmsh
import matplotlib.pyplot as plt
import numpy as np

choice = 0

if choice == 1:
    with pygmsh.occ.Geometry() as geom:
        points = [[-4, -5],
                  [-5, 5],
                  [7, 7],
                  [5, -5],
                  ]

        poly = geom.add_polygon(
            points,  # mesh_size=0.1,
        )

        geom.characteristic_length_max = 0.8
        geom.characteristic_length_min = 0.5

        disks = [
            geom.add_disk([-0.5, -0.25], 1.0),
            geom.add_disk([+0.5, -0.25], 1.0),
            geom.add_disk([0.0, 0.5], 1.0),
        ]

        # geom.boolean_difference(disks[0], disks[1:])
        # geom.boolean_intersection(disks)
        """geom.boolean_difference(
            geom.boolean_union([rectangle, disks[0], disk2]),
            geom.boolean_union([disk3, disk4])"""

        geom.boolean_difference(poly, disks)

        mesh = geom.generate_mesh()

        print("done")
else:
    with pygmsh.geo.Geometry() as geom:
        lcar = 160
        """p1 = geom.add_point([0.0, 0.0], lcar)
        p2 = geom.add_point([1.0, 0.0], lcar)
        p3 = geom.add_point([1.0, 0.5], lcar)
        p4 = geom.add_point([1.0, 1.0], lcar)
        s1 = geom.add_bspline([p1, p2, p3, p4])"""

        """points = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [0.5, 1], [0.3, 0.8], [0.0, 1.0], [0, 0.5]]
        
        point_entities = [geom.add_point(point, lcar) for point in points]

        point_entities.append(point_entities[0])

        s2 = geom.add_spline(point_entities)"""

        """p3 = geom.add_point([0.5, 1.0], lcar)
        p5 = geom.add_point([0.3, 0.8], lcar)
        p6 = geom.add_point([0.0, 1.0], lcar)
        p7 = geom.add_point([0, 0.5], lcar)
        s2 = geom.add_spline([p4, p3, p5, p6, p7, p1])"""

        """ll = geom.add_curve_loop([s2])
        pl = geom.add_plane_surface(ll)"""

        points = np.array([[2492, 1113], [2450, 1150], [1902, 2125], [2492, 3151],
                           [3666, 3151], [4257, 2137], [3666, 1113], [2540, 1113]], np.int32)

        point_entities = [geom.add_point(point, lcar) for point in points]
        point_entities.append(point_entities[0])

        lines = []
        for i in range(len(point_entities) - 1):
            line = geom.add_line(point_entities[i], point_entities[i + 1])
            lines.append(line)

        # Přidání křivky
        curve = geom.add_curve_loop(lines)
        surface = geom.add_plane_surface(curve)

        mesh = geom.generate_mesh(dim=2)

        print("done")

fig = plt.figure()
ax = fig.add_subplot(111)
p = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
plt.scatter(p[:, 0], p[:, 1])
p = np.array([[0.0, 1.0], [0.5, 1.0], [0.3, 0.8], [0, 0.5]])
plt.scatter(p[:, 0], p[:, 1])
ax.triplot(mesh.points[:, 0], mesh.points[:, 1], mesh.get_cells_type("triangle"))  # mesh.cells[1].data
plt.tight_layout()
# plt.show()


triangle_points = mesh.points[mesh.get_cells_type("triangle")[0]]
triangle_cells = [[0, 1, 2]]
# Vykreslení trojúhelníka
plt.triplot(triangle_points[:, 0], triangle_points[:, 1], triangle_cells)

plt.figure()
"""# Vykreslení všech trojúhelníků
for cell in mesh.get_cells_type("triangle"):
    triangle_points = mesh.points[cell]
    plt.triplot(triangle_points[:, 0], triangle_points[:, 1])"""

[plt.triplot(mesh.points[cell][:, 0], mesh.points[cell][:, 1]) for cell in mesh.get_cells_type("triangle")]

plt.tight_layout()
plt.show()
