import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd

data = pd.read_excel('COF_values.xlsx', sheet_name="COF", header=None, nrows=8).T
data.columns = data.iloc[0]
data = data.iloc[1:]


names = ['B', 'C1', 'C2', 'CP']

static_values = data.iloc[:, ::2]
kinetic_values = data.iloc[:, 1::2]

colors = ['tab:blue', 'tab:orange', 'limegreen', 'red', 'aqua', 'magenta']

# Hodnoty pro první sadu dat
static_bar_mean = static_values.mean(skipna=True)
static_std = static_values.std(skipna=True)

# Hodnoty pro druhou sadu dat
kinetic_bar_mean = kinetic_values.mean(skipna=True)
kinetic_std = kinetic_values.std(skipna=True)

# Šířka sloupců
bar_width = 0.3  # Upravte šířku sloupců podle potřeby
column_spacing = 1.5  # Upravte škálování sloupců podle potřeby
bar_spacing = 0.025  # Upravte mezeru mezi sloupci podle potřeby

# Vytvoření sloupcového grafu s mezerou
indexes = np.arange(0, len(names) * column_spacing, column_spacing)

plt.bar(indexes - (bar_width + bar_spacing) / 2, static_bar_mean, bar_width, label='Static', yerr=static_std,
        capsize=5, color=colors[0], zorder=4)
plt.bar(indexes + (bar_width + bar_spacing) / 2, kinetic_bar_mean, bar_width, label='Kinetic', yerr=kinetic_std,
        capsize=5, color=colors[1], zorder=4)

plt.gca().yaxis.set_minor_locator(AutoMinorLocator())

for axis in ['bottom', 'left', 'right']:
    plt.gca().spines[axis].set_linewidth(0.5)
    plt.gca().spines[axis].set_color('gray')

plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['top'].set_color('gray')
plt.gca().spines['top'].set_alpha(0.5)

plt.gca().tick_params(axis='x', which='major', direction='out', width=1, length=5, zorder=5, color="gray")
plt.gca().tick_params(axis='y', which='both', length=0)

for axis in ['left', 'right']:
    plt.gca().spines[axis].set_visible(False)

# Nastavení gridu pro major a minor ticks
plt.grid(axis='y', which='major', linestyle='-', linewidth='1.5', color='gray', alpha=0.5)
plt.grid(axis='y', which='minor', linestyle='-', linewidth='0.75', color='lightgray', alpha=0.8)

if plt.gca().get_ylim()[1] % plt.gca().get_yticks()[-1] != 0:
    step_ticks_y = np.mean(np.diff(plt.gca().get_yticks()))
    y_max = np.ceil(plt.gca().get_ylim()[1] / step_ticks_y) * step_ticks_y
    plt.gca().set_ylim(plt.gca().get_ylim()[0], y_max)

# Nastavení popisků os a titulku
plt.ylabel('Friction coefficient')

plt.xticks(indexes, names)
plt.legend()

# Zobrazení grafu
plt.show()
