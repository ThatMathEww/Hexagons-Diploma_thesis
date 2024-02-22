import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Seznam vrcholů trojúhelníků
triangle1 = np.array([[1, 1], [2, 1], [1.5, 2]])
triangle2 = np.array([[2, 1], [3, 1], [2.5, 2]])
triangle3 = np.array([[1.5, 2], [2, 2], [1.75, 2.5]])

# Vytvoření jediného seznamu vrcholů pro všechny trojúhelníky
vertices = np.concatenate([triangle1, triangle2, triangle3])

# Vytvoření polygonu
polygon = Polygon(vertices, closed=True, edgecolor='black', facecolor='gray')

# Vytvoření grafu
fig, ax = plt.subplots()
ax.add_patch(polygon)

# Nastavení limit pro osy
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)

# Zobrazení grafu
plt.show()
