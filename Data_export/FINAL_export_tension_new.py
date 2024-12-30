from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def swap_lists(list1, list2):
    return list2, list1


out_put_folder = ".outputs"

# Název Excel souboru
excel_file = f'Values_tension.xlsx'

do_tex = False

save_plot = True

file_type = "jpg"
out_dpi = 600

# Testy II a III musí být vůči testům hexagonů prohozeny
special_additional_information = {
    'T01_01-I_1s': 3}  # posunuté meření až u čtvté fotografie z důvodu opožděného zatěžování

main_image_folder = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos'
folder_n_corr = r'C:\Programy\Ncorr\Ncorr_post_v2e\export'
folder_measurements = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\data'

########################################################################################################################
if do_tex:
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{lmodern, amsmath, amsfonts, amssymb, amsthm, bm}')
    # plt.rcParams['font.size'] = 14

########################################################################################################################

folders_im = [name for name in [os.path.splitext(file)[0] for file in os.listdir(main_image_folder)]
              if name.startswith("T01")]
folders_nc = [name for name in [os.path.splitext(file)[0] for file in os.listdir(folder_n_corr)]
              if name.startswith("T01")]

folders = [name for name in folders_im if name in folders_nc]

# folders = [folders[i] for i in (15, 16, 17, 21, 22, 23, 45, 46, 47, 48, 49, 50)]

# m = 2  # I - 2 , II - 1    III - 0
# n = [i + m for i in range(0, len(folders) - 1, 3)]
# folders = [folders[i] for i in n]  # I - (2, 5, 8, 11) , II - (1, 4, 7, 10)    III - (0, 3, 6, 9)


type_1_sm = [name for name in range(len(folders)) if
             "-I_" in folders[name] and int(folders[name].split("-")[0].split("_")[1]) < 11]
type_2_sm = [name for name in range(len(folders)) if
             "-II_" in folders[name] and int(folders[name].split("-")[0].split("_")[1]) < 11]
type_3_sm = [name for name in range(len(folders)) if
             "-III_" in folders[name] and int(folders[name].split("-")[0].split("_")[1]) < 11]

type_1_la = [name for name in range(len(folders)) if
             "-I_" in folders[name] and int(folders[name].split("-")[0].split("_")[1]) > 10]
type_2_la = [name for name in range(len(folders)) if
             "-II_" in folders[name] and int(folders[name].split("-")[0].split("_")[1]) > 10]
type_3_la = [name for name in range(len(folders)) if
             "-III_" in folders[name] and int(folders[name].split("-")[0].split("_")[1]) > 10]

# Testy II a III musí být vůči testům hexagonů prohozeny
type_2_sm, type_3_sm = swap_lists(type_2_sm, type_3_sm)
type_2_la, type_3_la = swap_lists(type_2_la, type_3_la)

found_strains, found_stresses = [], []
modules = []

for folder in folders:
    path_strain = os.path.join(folder_n_corr, folder, "virtualExtensometer_2", f"{folder}-virtExt_2_strain-y.txt")
    path_force = os.path.join(folder_measurements, "data_csv", f"{folder}.csv")

    if not os.path.isfile(path_strain):
        print("Strain file not found:", folder)
        found_strains.append(None)
        found_stresses.append(None)
        continue
    if not os.path.isfile(path_force):
        print("Force file not found:", folder)
        found_strains.append(None)
        found_stresses.append(None)
        continue

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
    found_force -= found_force[0]

    found_strain = data_strain[:index_max_strain + 1]

    found_stress = found_force / (2.64 * 15.13)

    # if not found_strain[3] > 1.555600e-05:
    #     print(folder)
    #     found_strains.append(None)
    #     found_stresses.append(None)
    #     continue

    if folder in special_additional_information:
        moved_data = special_additional_information.get(folder, 0)

        found_strain = found_strain[moved_data:]

    m = np.where(found_strain < 0.03, found_strain, np.inf).argmax()

    found_strain = found_strain[:m]
    found_stress = found_stress[:m]

    found_strains.append(found_strain)
    found_stresses.append(found_stress)

    index_max_strain = np.where(data_strain < 0.01, data_strain, np.inf).argmax()

    module = (found_stress[index_max_strain] - found_stress[1]) / (found_strain[index_max_strain] - found_strain[1])

    modules.append(module)

mean_module = np.mean(modules)
std_module = np.std(modules)

print(mean_module)
print(std_module)

# indexes = [type_1_sm, type_2_sm, type_3_sm, type_1_la, type_2_la, type_3_la]
indexes = [type_1_sm, type_2_sm, type_3_sm, type_1_la, type_2_la, type_3_la]
# indexes = [type_1_sm, type_2_sm, type_3_sm]
# indexes = [type_1_la, type_2_la, type_3_la]

indexes_export = [[type_1_sm, type_2_sm, type_3_sm], [type_1_la, type_2_la, type_3_la]]

# Vytvoření subplots
colors = ("dodgerblue", "red", "limegreen", "orange", "purple", "cyan", "pink", "black", "yellow", "magenta")

if len(indexes) == 0:
    print("No data to plot.")
    exit(1)
elif len(indexes) == 1:
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
elif len(indexes) == 2:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
elif len(indexes) <= 4:
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
elif len(indexes) <= 6:
    fig, axs = plt.subplots(2, 3, figsize=(12, 5))
else:
    fig, axs = plt.subplots(3, np.ceil(len(indexes) / 3), figsize=(12, 8))
    i = int(len(indexes) - (3 * np.ceil(len(indexes) / 3)))
    if i < 0:
        axs[i:].remove()
    elif i > 0:
        print("Error in creating subplots.")
        exit(2)

try:
    axs = axs.flatten()
except AttributeError:
    axs = [axs]

if 7 > len(indexes) > 1 and len(indexes) % 2 == 1:  # pokud je lichý počet grafů a je jich od 2 do 6
    axs[-1].remove()

for i in range(len(indexes)):
    try:
        [axs[i].plot(found_strains[j] * 100 if "strain" in path_strain else 1, found_stresses[j], c='gray', lw=1,
                     alpha=0.5, zorder=5)
         for j in np.hstack(indexes[:i] + indexes[i + 1:]) if found_strains[j] is not None]
    except ValueError:
        pass

    [axs[i].plot(found_strains[j] * 100 if "strain" in path_strain else 1, found_stresses[j], color=colors[c], lw=1.5,
                 label=folders[j], zorder=40 - len(indexes[i]) - c) for c, j in enumerate(indexes[i]) if
     found_strains[j] is not None]

    axs[i].grid(color="lightgray", linewidth=0.5, zorder=0)
    for axis in ['top', 'right']:
        axs[i].spines[axis].set_linewidth(0.5)
        axs[i].spines[axis].set_color('lightgray')
        axs[i].spines[axis].set_zorder(0)

    if axs[i].get_xlim()[1] % axs[i].get_xticks()[-1] == 0:
        axs[i].spines['right'].set_visible(False)
    if axs[i].get_ylim()[1] % axs[i].get_yticks()[-1] == 0:
        axs[i].spines['top'].set_visible(False)

    axs[i].yaxis.set_minor_locator(AutoMinorLocator())
    axs[i].xaxis.set_minor_locator(AutoMinorLocator())

    axs[i].tick_params(axis='both', which='minor', direction='in', width=0.5, length=2.5, zorder=5, color="black")
    axs[i].tick_params(axis='both', which='major', direction='in', width=0.8, length=5, zorder=5, color="black")

    # axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    axs[i].set_xlabel(r"Strain [\%]" if do_tex else "Strain [%]")
    axs[i].set_ylabel(r"Stress [$MPa$]" if do_tex else "Stress [MPa]")

    axs[i].set_aspect('auto', adjustable='box')

handles, labels = axs[0].get_legend_handles_labels()
labels = [f"T-{l}" for l in range(len(labels))]
fig.legend(handles, labels, fontsize=8, borderaxespad=0, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=10)

fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.3)
if save_plot:
    plt.savefig(f"./{out_put_folder}/tension_all.{file_type}", format=file_type, dpi=out_dpi, bbox_inches='tight')
plt.figure()
[plt.plot(s * 100, f, zorder=3, label=l) for s, f, l in zip(found_strains, found_stresses, folders) if s is not None]
plt.gca().set_xlabel(r"Strain [\%]" if do_tex else "Strain [%]")
plt.gca().set_ylabel(r"Stress [$MPa$]" if do_tex else "Stress [MPa]")
plt.grid()
# plt.legend()
plt.tight_layout()

fig, ax1 = plt.subplots(figsize=(5.2, 3))
fig2, ax2 = plt.subplots(figsize=(5.2, 3))

indexes = [type_1_sm, type_3_sm, type_2_sm, type_1_la, type_3_la, type_2_la]
datas_pack = zip(("T01-I-S", "T01-II-S", "T01-III-S", "T01-I-L", "T01-II-L", "T01-III-L"),
                 [*indexes],
                 [*colors])

for n, (name, curve_index, color) in enumerate(datas_pack):
    min_len = np.min([len(found_strains[j]) for j in curve_index if found_strains[j] is not None])
    data_plot_x = [found_strains[j][:min_len] for j in curve_index if found_strains[j] is not None]
    data_plot_y = [found_stresses[j][:min_len] for j in curve_index if found_stresses[j] is not None]

    data_mean_x = np.mean(data_plot_x, axis=0)
    data_mean_y = np.mean(data_plot_y, axis=0)
    data_max = np.max(data_plot_y, axis=0)
    data_min = np.min(data_plot_y, axis=0)
    data_std = np.std(data_plot_y, axis=0)

    ax2.plot(data_mean_x * 100 if "strain" in path_strain else 1, data_mean_y, label=name, lw=2, c=color, zorder=20 + n)

    ax1.plot(data_mean_x * 100 if "strain" in path_strain else 1, data_mean_y, label=name, lw=2, c=color, zorder=20 + n)
    ax1.fill_between(data_mean_x * 100 if "strain" in path_strain else 1, data_mean_y + data_std,
                     data_mean_y - data_std, alpha=0.35, color=color, zorder=10 + n)
    ax1.plot(data_mean_x * 100 if "strain" in path_strain else 1, data_max, ls="--", lw=1, c=color, zorder=30 + n,
             alpha=0.7)
    ax1.plot(data_mean_x * 100 if "strain" in path_strain else 1, data_min, ls="--", lw=1, c=color, zorder=30 + n,
             alpha=0.7)

    modules = []

    for i in range(len(data_plot_x)):
        index_max_strain = np.where(data_plot_x[i] < 0.01, data_plot_x[i], np.inf).argmax()

        module = (data_plot_y[i][index_max_strain] - data_plot_y[i][1]) / (
                data_plot_x[i][index_max_strain] - data_plot_x[i][1])

        modules.append(module)

    mean_module = np.mean(modules)
    std_module = np.std(modules)

    print(mean_module)
    print(std_module)

for axes in [ax1, ax2]:
    axes.grid(color="lightgray", linewidth=0.5, zorder=0)
    for axis in ['top', 'right']:
        axes.spines[axis].set_linewidth(0.5)
        axes.spines[axis].set_color('lightgray')

    if axes.get_xlim()[1] % axes.get_xticks()[-1] == 0:
        axes.spines['right'].set_visible(False)
    if axes.get_ylim()[1] % axes.get_yticks()[-1] == 0:
        axes.spines['top'].set_visible(False)

    axes.yaxis.set_minor_locator(AutoMinorLocator())
    axes.xaxis.set_minor_locator(AutoMinorLocator())

    axes.tick_params(axis='both', which='minor', direction='in', width=0.5, length=2.5, zorder=5, color="black")
    axes.tick_params(axis='both', which='major', direction='in', width=0.8, length=5, zorder=5, color="black")

    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes.legend(fontsize=8, bbox_to_anchor=(0.5, -0.4), loc="center", borderaxespad=0, ncol=3)
    axes.set_xlabel(r"Strain [\%]" if do_tex else "Strain [%]")
    axes.set_ylabel(r"Stress [$MPa$]" if do_tex else "Stress [MPa]")

    axes.set_aspect('auto', adjustable='box')

ax2.set_ylim(ax1.get_ylim())
ax2.set_xlim(ax1.get_xlim())
fig.subplots_adjust(bottom=0.3, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.3)
fig2.subplots_adjust(bottom=0.3, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.3)
# fig.tight_layout()
# fig2.tight_layout()

if save_plot:
    fig.savefig(f"./{out_put_folder}/tension_finalplot_tot.{file_type}", format=file_type, dpi=out_dpi,
                bbox_inches='tight')
    fig2.savefig(f"./{out_put_folder}/tension_finalplot_single.{file_type}", format=file_type, dpi=out_dpi,
                 bbox_inches='tight')

# # ##############################################################################
# if not os.path.exists(out_put_folder):
#     os.makedirs(out_put_folder, exist_ok=True)
#
# try:
#     excel_writer = pd.ExcelWriter(os.path.join(out_put_folder, excel_file), engine='xlsxwriter')
# except PermissionError as e:
#     print(f'\033[31;1;21mERROR\033[0m\n\tSoubor [{excel_file}] nelze upravovat, pravděpodobně je otevřen.'
#           f'\n\tPOPIS: {e}')
#     exit(10)
# except (KeyError, Exception) as e:
#     print(f'\033[31;1;21mERROR\033[0m\n\tSoubor [{excel_file}] nelze uložit.\n\tPOPIS: {e}')
#     exit(11)
#
# # Zápis dat do listů
# for i, data in enumerate([all_datas[j] for j in
#                           np.hstack([data_indexes__I_O, data_indexes__II_O, data_indexes__III_O,
#                                      data_indexes__I_max_O, data_indexes__II_max_O, data_indexes__II_max_O])]):
#     sheet_name = data[0]
#     sheet_name = sheet_name.replace("T01_", "").replace("_1s", "")
#
#     # Přepsání názvů sloupců pro třetí DataFrame
#     # df3.columns = ['New_M', 'New_N']
#
#     # Ukládání jednotlivých DataFrame na různá místa
#     start_row = 0
#     col_start = 0
#     data[4].to_excel(excel_writer, sheet_name=sheet_name, startrow=start_row, startcol=col_start, index=False)
#     col_start += len(data[4].columns)
#     data[2].to_excel(excel_writer, sheet_name=sheet_name, startrow=start_row, startcol=col_start, index=False)
#     col_start += len(data[2].columns)
#     data[1].to_excel(excel_writer, sheet_name=sheet_name, startrow=start_row, startcol=col_start, index=False)
#     col_start += len(data[1].columns)
#     data[3].to_excel(excel_writer, sheet_name=sheet_name, startrow=start_row, startcol=col_start, index=False)
#
#
# # Zavření Excel souboru
# excel_writer.close()
# print(f"Soubor byl úspěšně uložen do: [ {os.path.join(out_put_folder, excel_file)} ]")
# # ##############################################################################

plt.show()
