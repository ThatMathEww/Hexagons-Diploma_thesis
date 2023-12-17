import h5py
import matplotlib.pyplot as plt
import numpy as np

# Název souboru .mat
mat_file = r'C:\Programy\Ncorr\Ncorr_post_v2e\savedProjects\T01_example_long_17-I.mat'

# Index řádku i a název sloupce 'strain_xx'
i = 40  # můžete změnit podle vašich potřeb
column_name = 'strains_xx'

# Načtení .mat souboru
with h5py.File(mat_file, 'r') as file:
    keys = list(file.keys())
    # Výpis klíčů
    for key in keys:
        print(key)

    for key in file['plottingData'].keys():
        print("\t", key)

    # Přístup k datům ve struktuře
    i = min(i + 1, len(file['plottingData'][column_name]) - 1)

    roi_mask = np.array(file[file['plottingData']['roi'][0, 0]]).T

    subset_spacing = int(np.array(file[file['plottingData']['subsetSpacing'][0, 0]]).squeeze()) + 1

    data_matrix = np.array(file[file['plottingData'][column_name][i, 0]]).T
    mask = np.array(file[file['plottingData'][column_name][1, 0]]).T
    out_of_image = np.array(file[file['plottingData']['outOfRoiImage'][0, 0]]).T
    image = np.array(file[file['plottingData']['originalImage'][0, 0]]).T

    # offset = np.array(file[file['plottingData']['originOffset'][0, 0]]).squeeze()

# Vykreslení heatmapy
# resized_matrix = np.kron(data_matrix, np.ones((subset_spacing, subset_spacing)))
# Aplikace masky na matici
masked_data = np.where(mask != 0, data_matrix, np.nan)

plt.figure()
plt.imshow(roi_mask, cmap='gray', interpolation='none', aspect='equal')
plt.title("Roi")

plt.figure()
plt.imshow(image, cmap='gray', interpolation='none', aspect='equal',
           extent=[0, image.shape[1], 0, image.shape[0]])
plt.imshow(masked_data, cmap='jet', interpolation='nearest', aspect='equal',
           extent=[0, image.shape[1], 0, image.shape[0]], alpha=0.7)
plt.colorbar(label='Hodnota')
plt.title(f"{column_name}: {i} photo")



# Předpokládejme, že máte vektor x, vektor y a matici hodnot Z
x = np.linspace(0, 10, 100)
y = np.linspace(0, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)  # Předpokládaná matice hodnot

# Předpokládaná maska (můžete ji nahradit vaší vlastní maskou)
mask = (X + Y) < 5

# Oříznutí hodnot podle masky
Z_masked = np.ma.masked_where(mask, Z)

plt.figure()
# Vytvoření plošného grafu s maskou a horizontální mřížkou
plt.pcolormesh(X, Y, Z_masked, cmap='viridis')
plt.colorbar(label='Hodnota')

# Přidání horizontální mřížky (např. na hodnotě y=2)
plt.axhline(y=2, color='red', linestyle='--', linewidth=2, label='Horizontal Grid')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Pcolormesh graf s maskou a horizontální mřížkou')

# Přidání legendy
plt.legend()

plt.show()
