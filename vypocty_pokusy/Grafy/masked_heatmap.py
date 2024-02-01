import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

size_x = 100
size_y = 100

x = np.random.rand(size_x)  # Příklad: 100 náhodných x souřadnic
y = np.random.rand(size_y)  # Příklad: 100 náhodných y souřadnic
z = np.sin(x * y)  # Příklad: nějaká funkce pro závislost z na x a y
xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), size_x), np.linspace(y.min(), y.max(), size_y))

# Lineární interpolace
zi = griddata((x, y), z, (xi, yi), method='linear')

hole_center_x = 0.5  # X souřadnice středu díry
hole_center_y = 0.5  # Y souřadnice středu díry
hole_radius = 0.1  # Poloměr díry
distance_to_center = np.sqrt((xi - hole_center_x) ** 2 + (yi - hole_center_y) ** 2)
mask = distance_to_center <= hole_radius
zi[mask] = np.nan  # Nastavte hodnoty uvnitř díry na NaN


# Maticová maska
mask = np.random.rand(len(x), len(y)) > 0.5
mask[:, 50] = True
zi_masked = np.ma.masked_array(zi, mask=~mask)

plt.figure()
plt.imshow(zi_masked, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis')
plt.colorbar()  # Přidá legendu s hodnotami
plt.show()
