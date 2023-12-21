import pygmsh
import numpy as np

# Definice oblasti 1
area1 = [
    np.array(((1301, 537),
             (1216, 1255),
             (794, 1128),
             (904, 444))),
    np.array(((1098, 512),
             ( 1098, 968),
             ( 1191, 537)))
]

# Definice oblasti 2
area2 = np.array((( 500, 512),
             ( 500, 968),
             ( 500, 537)))

with pygmsh.occ.Geometry() as geom:
    # for p in area1:
    poly1 = [geom.add_polygon(p) for p in area1]

    if len(area2) > 2:
        [geom.boolean_difference(p, geom.add_polygon(area2)) for p in poly1]

    mesh = geom.generate_mesh(dim=2)

# Pro získání informací o síti můžete použít něco jako:
points, cells, point_data, cell_data, field_data = generate_mesh.extract_cells_from_mesh(mesh)
