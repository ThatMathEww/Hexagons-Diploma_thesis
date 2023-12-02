import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import matplotlib.tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Načtení STL souboru
stl_filename = "3.stl"
your_mesh = mesh.Mesh.from_file(stl_filename)

# Získání numpy pole vrcholů
vertices = your_mesh.vectors

# Vykreslení 3D grafu
fig_3D = plt.figure()
ax = fig_3D.add_subplot(111, projection='3d')

x = vertices[:, :, 0].flatten() - np.min(vertices[:, :, 0].flatten())
y = vertices[:, :, 1].flatten() - np.min(vertices[:, :, 1].flatten())
z = vertices[:, :, 2].flatten() - np.min(vertices[:, :, 2].flatten())

# Rozložení trojúhelníků pomocí plot_trisurf
surf = ax.plot_trisurf(x, y, z, cmap='jet', alpha=1, edgecolor='None', antialiased=True)

"""# Převedení trojúhelníkové matice na jednotnou plochu
triangulation = mtri.Triangulation(vertices[:, :, 0].flatten(),
                                   vertices[:, :, 1].flatten())

# Přidání plochy s gradientním zbarvením
surf = ax.plot_trisurf(triangulation, vertices[:, :, 2].flatten(), cmap='jet', alpha=1, edgecolor='None',
                       antialiased=True)"""

"""ax.plot_trisurf(triangulation, vertices[:, :, 2].flatten()-0.1, cmap='jet', alpha=1, edgecolor='None', antialiased=True)
ax.plot_trisurf(triangulation, vertices[:, :, 2].flatten()-0.2, cmap='jet', alpha=1, edgecolor='None', antialiased=True)
ax.plot_trisurf(triangulation, vertices[:, :, 2].flatten()-0.3, cmap='jet', alpha=1, edgecolor='None', antialiased=True)"""

"""poly3d = mplot3d.art3d.Poly3DCollection(your_mesh.vectors, alpha=1)
ax.add_collection3d(poly3d)

# Přizpůsobení parametrů osy pro lepší zobrazení
ax.auto_scale_xyz(your_mesh.x.flatten(), your_mesh.y.flatten(), your_mesh.z.flatten())"""

# Přidání colorbaru
cbar = plt.colorbar(surf, shrink=0.75, aspect=15)

# Nastavení jednotek pro colorbar
# cbar.set_label('[µm]')
cbar.ax.text(1, np.max(z) * 1.025, '[µm]', rotation=0, va='bottom', ha='left')

# Nastavení os
ax.set_xlabel(f'{np.max(x):.3f} µm', labelpad=-10)
ax.set_ylabel(f'{np.max(y):.3f} µm', labelpad=-10)
# ax.set_zlabel(f'{round(np.max(z))}')

# Vypnutí hodnot na ose X a Y
ax.set_xticks([])
ax.set_yticks([])

# ax.axis('off')
ax.axis('equal')

# Nastavení spodního limitu osy Z na nulu
ax.set_zlim(0)

ax.auto_scale_xyz(your_mesh.x.flatten(), your_mesh.y.flatten(), your_mesh.z.flatten())
# Vykreslení 3D grafu
ax.view_init(elev=20, azim=60)
# ax.dist = 50  # Upravte vzdálenost osy podle potřeby
# ax.invert_xaxis()

plt.tight_layout()

plt.show()
