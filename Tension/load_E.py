import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

out_put_folder = ""

main_image_folder = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos'
folder_n_corr = r'C:\Programy\Ncorr\Ncorr_post_v2e\export'
folder_measurements = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\data'

########################################################################################################################

folders = [name for name in [os.path.splitext(file)[0] for file in os.listdir(main_image_folder)]
           if name.startswith("T01")]
folders = [folders[i] for i in (15, 16, 17, 21, 22, 23, 45, 46, 47, 48, 49, 50)]

m = 0  # I - 2 , II - 1    III - 0
n = [i + m for i in range(0, len(folders) - 1, 3)]
folders = [folders[i] for i in n]  # I - (2, 5, 8, 11) , II - (1, 4, 7, 10)    III - (0, 3, 6, 9)

found_strains, found_stresses = [], []
modules = []

for folder in folders:
    path_strain = os.path.join(folder_n_corr, folder, "virtualExtensometer_2", f"{folder}-virtExt_2_strain-y.txt")
    path_force = os.path.join(folder_measurements, "data_csv", f"{folder}.csv")

    data_strain = np.loadtxt(path_strain)[:-1]
    photo_times = np.arange(0, len(data_strain))
    data_tension = pd.read_csv(path_force)

    data_distances = data_tension.iloc[:, 0].values
    data_distances -= data_distances[0]
    data_force = data_tension.iloc[:, 1].values
    data_force -= data_force[0]
    data_time = data_tension.iloc[:, 2].values
    data_time -= data_time[0]

    # Index nejbližší nižší hodnoty
    index_max_strain = np.where(data_strain < 0.01, data_strain, np.inf).argmax()
    index_max_strain = len(data_time)

    # Získání indexů, které jsou nejblíže hodnotám v druhém vektoru
    index_at_photo = np.abs(data_time[:, np.newaxis] - photo_times).argmin(axis=0)
    found_times = data_time[index_at_photo][:index_max_strain + 1]
    found_force = data_force[index_at_photo][:index_max_strain + 1]
    found_strain = data_strain[:index_max_strain + 1]

    found_stress = found_force / (2.64 * 15.13)

    found_strains.append(found_strain)
    found_stresses.append(found_stress)

    index_max_strain = np.where(data_strain < 0.01, data_strain, np.inf).argmax()

    module = (found_stress[index_max_strain] - found_stress[1]) / (found_strain[index_max_strain] - found_strain[1])

    modules.append(module)

mean_module = np.mean(modules)
std_module = np.std(modules)

print(mean_module)
print(std_module)

plt.figure()
[plt.plot(s * 100, f, marker="o", zorder=3, label=l) for s, f, l in zip(found_strains, found_stresses, folders)]
"""strain = (np.mean([f[1] for f in found_strains]) * 100, np.mean([f[index_max_strain] for f in found_strains]) * 100)
stress = (np.mean([f[1] for f in found_stresses]), np.mean([f[index_max_strain] for f in found_stresses]))
plt.plot(strain, stress, zorder=4, color="black")"""
plt.plot((found_strain[1] * 100, found_strain[index_max_strain] * 100),
         (found_stress[1], found_stress[index_max_strain]), zorder=4, color="black")
plt.gca().set_xlabel("Strain [%]")
plt.gca().set_ylabel("Stress [MPa]")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
