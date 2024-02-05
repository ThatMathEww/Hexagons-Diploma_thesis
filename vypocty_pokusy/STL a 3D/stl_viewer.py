import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.animation import FuncAnimation

"""import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import matplotlib.tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable"""
from scipy.interpolate import griddata
import imageio

plt.rcParams.update({'font.size': 12})
# plt.style.use('dark_background')

make_gif = False
save_frame = True

# Načtení STL souboru
stl_filename = "10-down.stl"
your_mesh = mesh.Mesh.from_file(stl_filename)

# Získání numpy pole vrcholů
vertices = your_mesh.vectors

x = vertices[:, :, 0].flatten() - np.min(vertices[:, :, 0].flatten())
y = vertices[:, :, 1].flatten() - np.min(vertices[:, :, 1].flatten())
z = vertices[:, :, 2].flatten() - np.min(vertices[:, :, 2].flatten())

# z = ((z - np.min(z)) / (np.max(z) - np.min(z)))*50

z_scale = 63.3176378351889 / 117.9609375

z = z * z_scale

maximal_z = 172.82 * z_scale  # np.max(z)
minimal_z = np.min(z)  # np.min(z)

# Rozložení trojúhelníků pomocí plot_trisurf
"""surf = ax.plot_trisurf(x, y, z, cmap='jet', alpha=1, edgecolor='k', linewidth=0.05, antialiased=False,
                       shade=False)
"""

# Interpolace dat pro získání 2D mřížky
x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 600), np.linspace(min(y), max(y), 600))
z_grid = griddata((x, y), z, (x_grid, y_grid), method='linear')

frames = {}
step = 5
for azim in range(0, 360 + step, step):
    # Vykreslení 3D grafu
    fig_3D = plt.figure()
    ax = fig_3D.add_subplot(111, projection='3d')

    ls = LightSource(azdeg=100, altdeg=-25)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(z_grid, cmap=plt.get_cmap('jet'), vert_exag=0.1, blend_mode='overlay', vmin=minimal_z,
                   vmax=maximal_z)
    surf = ax.plot_surface(x_grid, y_grid, z_grid, rstride=1, cstride=1, facecolors=rgb, cmap='jet', shade=True,
                           antialiased=False, edgecolor='none', linewidth=0, alpha=1, vmin=minimal_z, vmax=maximal_z)

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
    cbar = fig_3D.colorbar(surf, shrink=0.75, aspect=15, pad=0.075)
    t = cbar.get_ticks()
    cbar.set_ticks(t)
    t = np.linspace(minimal_z, maximal_z, len(t))
    cbar.set_ticklabels([f'{i:.2f}' for i in t])

    # Nastavení jednotek pro colorbar
    # cbar.set_label('[µm]')
    cbar.ax.text(1.5, 1.05, '[µm]', rotation=0, va='bottom', ha='left')

    # Nastavení os
    ax.set_xlabel(f'{np.max(x) + 5.023:.3f} µm', labelpad=-10)
    ax.set_ylabel(f'{np.max(y) + 5.023:.3f} µm', labelpad=-10)
    # ax.set_zlabel(f'{round(np.max(z))}')

    # Vypnutí hodnot na ose X a Y
    ax.set_xticks([])
    ax.set_yticks([])

    # ax.axis('off')
    z_max_rounded = np.ceil(maximal_z / 10) * 10
    ax.set_zticks(np.round(np.linspace(0, z_max_rounded + 1, 3) / 10) * 10)
    ax.set_zlim(0, z_max_rounded)
    # ax.axis('equal')

    ax.set_box_aspect([np.ptp(arr) for arr in [x, y, np.hstack((z, [maximal_z]))]])

    # ax.auto_scale_xyz(your_mesh.x.flatten(), your_mesh.y.flatten(), your_mesh.z.flatten())
    # Vykreslení 3D grafu
    ax.view_init(elev=30, azim=110 + azim, roll=0)
    ax.dist = 1  # Upravte vzdálenost osy podle potřeby
    # ax.invert_xaxis()

    ax.set_facecolor((0, 0, 0, 0))

    # Přidejte barevnou osu pro snadnou interpretaci hloubky
    ax.xaxis.set_pane_color((0.6, 0.6, 0.6, 0.5))
    ax.yaxis.set_pane_color((0.6, 0.6, 0.6, 0.5))
    ax.zaxis.set_pane_color((0.6, 0.6, 0.6, 0.5))

    plt.tight_layout()

    if save_frame:
        plt.savefig(f'./3D outputs/output_object_{stl_filename.split(".")[0]}_{azim}.jpg', dpi=500, bbox_inches='tight')

    if make_gif:
        frames[azim] = fig_3D
        plt.close(fig_3D)

        print(f"Vykreslení 3D grafu pro azimut {azim} dokončeno.")

    else:
        plt.show()
        break

if make_gif:
    print("Vykreslení 3D grafu dokončeno.")

    # Uložení snímků do GIFu
    gif_filename = "./3D output/azimuth_animation.gif"
    with imageio.get_writer(gif_filename, mode='I', duration=step / 10) as writer:
        for azim, fig in frames.items():
            # Uložení snímku do GIFu
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            writer.append_data(image)

    print(f"GIF byl vytvořen jako {gif_filename}.")

    plt.show()
