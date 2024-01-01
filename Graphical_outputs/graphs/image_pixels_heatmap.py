import numpy as np
import matplotlib.pyplot as plt

matrix_size = (8, 10)

# Počet náhodných čísel
num_values = matrix_size[0] * matrix_size[1]

# Spojení dvou sad dohromady
rand_values = np.concatenate((np.random.normal(loc=0.01, scale=0.18, size=num_values*10),
                              np.random.normal(loc=0.99, scale=0.18, size=num_values*10)))

rand_values = rand_values[(rand_values >= 0) & (rand_values <= 1)]

# Vizualizace histogramu
plt.hist(rand_values, bins=100, density=True, alpha=0.6, color='g')

# Přidání vertikální linie na hodnotu 0.5 pro orientaci
plt.axvline(x=0.5, color='red', linestyle='dashed', linewidth=2)

# Nastavení popisků
plt.title('Spojená gaussovská rozdělení')
plt.xlabel('Hodnota')
plt.ylabel('Frekvence')

# Vytvoření náhodných dat
data = np.random.choice(rand_values, size=num_values).reshape(matrix_size)  # np.random.random((8, 10))

color_grid = "white"  # 'white' // 'none'
line_width_grid = 2
line_style = '-'

# Min-Max normalizace
min_val = np.min(data)
data = (data - min_val) / (np.max(data) - min_val)

fig, ax = plt.subplots()

# Vytvoření heatmapy
ax.imshow(data, cmap='gray', extent=[0, data.shape[1], 0, data.shape[0]], interpolation='none')

# Nastavení popisků os
ax.set_xlabel('x coordinates')
ax.set_ylabel('y coordinates')

# Zobrazení hodnot v každém čtverci s kontrastní barvou popisků
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        value = data[i, j]
        ax.text((j + 0.5), data.shape[0] - (i + 0.5), f'{value:.2f}', color='white' if 0 <= value < 0.35 else 'black',
                ha='center', va='center', fontsize=8)

ax.set_xticks(np.arange(data.shape[1]) + 0.5, labels=np.arange(data.shape[1]))
ax.set_yticks(np.arange(data.shape[0]) + 0.5, labels=np.arange(data.shape[0]))

# ax.spines[:].set_visible(False)
ax.spines[:].set_color(color_grid)
ax.spines[:].set_linewidth(line_width_grid)

ax.set_xticks(np.arange(data.shape[1] + 1), minor=True)
ax.set_yticks(np.arange(data.shape[0] + 1), minor=True)
ax.grid(which="minor", color=color_grid, linestyle=line_style, linewidth=line_width_grid)
ax.tick_params(which="minor", bottom=False, left=False)

ax.set_facecolor("none")
fig.set_facecolor("none")

# ax.set_yticklabels(ax.get_yticklabels()[::-1])
ax.invert_yaxis()

fig.tight_layout()
ax.set_aspect('equal', adjustable='box')

# Zobrazení grafu
plt.show()
