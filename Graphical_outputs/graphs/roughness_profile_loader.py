import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.fft import fft, ifft
import pandas as pd
import os

folder = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\Roughness_data'

files = (r'M1\7-up-20 (type A)\7-up-20x-1.csv', r'M1\3-up-20 (type B)\3-up-20x-1.csv',
         r'M2\11-up-20 (type C)\11-up-20x-1.csv',
         r'M1\7-up-20 (type A)\7-up-20x-4.csv', r'M1\3-up-20 (type B)\3-up-20x-5.csv',
         r'M2\11-up-20 (type C)\11-up-20x-4.csv',)

plot_min_max = True

path = os.path.join(folder, files[0])

df = pd.read_csv(path, skiprows=16, header=None)

"""df['rolling_mean'] = df[1].rolling(window=10).mean()
z = df['rolling_mean'].values
z -= np.min(z)"""

# Příklad dat
x = df[0].values
y = df[1].values

y -= np.min(y)

plt.figure(num="Cross-section")

# plt.title("Sample cross-section")

# Vykreslení původní křivky
plt.plot(x, y, label='Sample profile', zorder=3)

if plot_min_max:
    # Definice průměrovacího jádra pro vyhlazení (např. jednoduchý klouzavý průměr)
    kernel_size = 15
    kernel = np.ones(kernel_size) / kernel_size

    # Použití konvoluce pro vyhlazení
    z = np.convolve(y, kernel, mode='valid')

    # plt.plot(x[:len(z)], z, color='lightgray', zorder=3)

    # Nalezení lokálních maxim
    lok_maxima, _ = find_peaks(z, width=100)

    # Nalezení lokálních minim s oknem širokým 200
    lok_minima, _ = find_peaks(-z, width=30)

    differences_max = [(x[lok_maxima][i + 1] - x[lok_maxima][i]) / 1000 for i in range(len(x[lok_maxima]) - 1)]
    differences_min = [(x[lok_minima][i + 1] - x[lok_minima][i]) / 1000 for i in range(len(x[lok_minima]) - 1)]
    differences_max_min = [(y[lok_maxima][i] - y[lok_minima][i]) / 1000 for i in
                           range(min(len(lok_minima), len(lok_maxima)))]

    print(differences_max, "=>", np.mean(differences_max))
    print(differences_min, "=>", np.mean(differences_min))
    print(differences_max_min, "=>", np.mean(differences_max_min))

    plt.plot(x[lok_maxima], y[lok_maxima], 'rx', label='Surface peak', zorder=4)
    plt.plot(x[lok_minima], y[lok_minima], 'gx', label='Surface valley', zorder=4)

plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())

plt.gca().set_xlim(-10, np.max(x) + 10)
plt.gca().set_ylim(-10, (np.ceil((np.max(y) + 20) / 20) * 20) + 10)

plt.xlabel('[μm]')
plt.ylabel('Height profile [μm]')

plt.grid(color="lightgray", linewidth=0.5, zorder=0)
for axis in ['top', 'right']:
    plt.gca().spines[axis].set_linewidth(0.5)
    plt.gca().spines[axis].set_color('lightgray')

if plt.gca().get_xlim()[1] % plt.gca().get_xticks()[-1] == 0:
    plt.gca().spines['right'].set_visible(False)
if plt.gca().get_ylim()[1] % plt.gca().get_yticks()[-1] == 0:
    plt.gca().spines['top'].set_visible(False)

# Nastavte značky čísel, aby šly dovnitř osy
plt.gca().tick_params(axis='both', which='minor', direction='in', width=0.5, length=2.5, zorder=5, color="black")
plt.gca().tick_params(axis='both', which='major', direction='in', width=0.8, length=5, zorder=5, color="black")

# Přesunutí legendy pod osu
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncols=4)

plt.gca().set_aspect('equal', adjustable='box')

plt.tight_layout()

"""
def approximate_waveform(x, y, num_components):
    # Proveďte Fourierovu transformaci
    fft_values = fft(y)

    # Nulování frekvencí nad num_components
    fft_values[num_components + 1:] = 0
    fft_values[-num_components:] = 0

    # Proveďte zpětnou Fourierovu transformaci
    y_approx = ifft(fft_values).real

    return y_approx


# Počet komponent pro aproximaci
num_components = 3  # Můžete změnit podle potřeby

# Aproximace vlnové křivky
y_waviness = approximate_waveform(x, y, num_components)

plt.plot(x, y_waviness, label=f"Aproximovaná vlnová křivka ({num_components} komponenty)", linestyle="--", color="red")
"""

# Zobrazení grafu
plt.show()
