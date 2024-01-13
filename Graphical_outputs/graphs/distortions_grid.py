import numpy as np
import matplotlib.pyplot as plt


def pincushion_deformation(grid_size, radius_factor, deformation_factor):
    # Vytvoření pravidelného gridu
    x, y = np.meshgrid(np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size))

    # Převod kartézských souřadnic na polární souřadnice
    theta = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)

    # Radiální deformace s upravenou funkcí pro více deformace na kraji
    r_deformed = r + deformation_factor * (r * radius_factor) ** 2

    # Převod zpět na kartézské souřadnice
    x_deformed = r_deformed * np.cos(theta)
    y_deformed = r_deformed * np.sin(theta)

    return x, y, x_deformed, y_deformed


def barrel_deformation(grid_size, radius_factor, deformation_factor):
    # Vytvoření pravidelného gridu
    x, y = np.meshgrid(np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size))

    # Převod kartézských souřadnic na polární souřadnice
    theta = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)

    # Pincushion deformace s opačným efektem - body se od sebe vzdalují
    r_deformed = r - deformation_factor * (r * radius_factor) ** 2

    # Převod zpět na kartézské souřadnice
    x_deformed = r_deformed * np.cos(theta)
    y_deformed = r_deformed * np.sin(theta)

    return x, y, x_deformed, y_deformed


def plot_deformed_grid_lines(x, y, g_size, ax, color="black", title='Deformed grid'):
    # Vykreslení čar spojující sousední body
    for i in range(g_size):
        ax.plot(x[i, :], y[i, :], color=color)  # Spoje v řádku
        ax.plot(x[:, i], y[:, i], color=color)  # Spoje ve sloupci
    # plt.scatter(x, y, color='red')  # Vykreslení bodů pro lepší viditelnost
    ax.set_title(title, y=-0.1)


# Parametry deformace
grid_size = 21
radius_factor = 1.8
deformation_factor = 0.1

# Vytvoření radiálně deformovaného gridu
x, y, x_deformed, y_deformed = barrel_deformation(grid_size, radius_factor, deformation_factor)

# Vykreslení pravidelného a deformovaného gridu
s = 4
plt.figure(figsize=(3 * s, s))
plt.rcParams['font.family'] = 'Times New Roman'

ax1 = plt.subplot(1, 3, 1)
plot_deformed_grid_lines(x, y, grid_size, ax=ax1, title='Original grid', color='blue')
# plt.scatter(x, y, marker='.', color='blue')
ax1.set_aspect('equal')
ax1.axis('off')


ax2 = plt.subplot(1, 3, 2)
plot_deformed_grid_lines(x_deformed, y_deformed, grid_size, ax=ax2, title='Barrel distortion', color='red')
# plt.scatter(x_deformed, y_deformed, marker='.', color='red')
ax2.set_aspect('equal')
ax2.axis('off')

# Vytvoření radiálně deformovaného gridu
x, y, x_deformed, y_deformed = pincushion_deformation(grid_size, radius_factor, 3 * deformation_factor)

ax3 = plt.subplot(1, 3, 3)
plot_deformed_grid_lines(x_deformed, y_deformed, grid_size, ax=ax3, title='Pincushion distortion', color='green')
# plt.scatter(x_deformed, y_deformed, marker='.', color='green')
ax3.set_aspect('equal')
ax3.axis('off')


plt.tight_layout(pad=2)

plt.savefig("grid.pdf", format="pdf", bbox_inches='tight')
plt.show()
