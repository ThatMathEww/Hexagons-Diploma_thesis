import gmsh

# Inicializace Gmsh
gmsh.initialize()

# Definice bodů okraje oblasti
p1 = gmsh.model.geo.addPoint(0.0, 0.01, 0.0, meshSize=10)
p2 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, meshSize=10)
p3 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, meshSize=0.1)
p4 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, meshSize=0.1)
p5 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, meshSize=0.1)

# Definice link
l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p5)
l5 = gmsh.model.geo.addLine(p5, p1)

# Definice oblasti
loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5])
surface = gmsh.model.geo.addPlaneSurface([loop])

# Nastavení parametrů sítě
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 5)

# Synchronizace geometrie a sítě
gmsh.model.geo.synchronize()

# Přidání omezující podmínky pro tvorbu sítě
gmsh.model.geo.mesh.setTransfiniteCurve(l1, 1)
gmsh.model.geo.mesh.setTransfiniteCurve(l2, 10)

# Generování sítě
gmsh.model.mesh.generate()

# Získání výsledné sítě
mesh = gmsh.model.mesh.getNodes(), gmsh.model.mesh.getElements()

# Vykreslení sítě
gmsh.fltk.initialize()
gmsh.fltk.run()

# Ukončení Gmsh
gmsh.finalize()
