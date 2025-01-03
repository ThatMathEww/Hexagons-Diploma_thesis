from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd
import numpy as np
import os


# Funkce pro hledání lineárních úseků
def find_linear_section(calculation_type, x, y, min_x, max_x, threshold=0.95, constant1=1,
                        constant2=0.999, min_length=4):
    i_1 = 0
    i_2 = 0
    r_value = 0

    if calculation_type == 1:
        n_x = len(x)
        best_score = -np.inf
        best_section = None
        best_r = 0

        sorted_ind = np.argsort(x)
        x = x[sorted_ind]
        y = y[sorted_ind]

        if min_x < np.min(x) or max_x > np.max(x):
            raise ValueError("min_x a max_x musí být v rozmezí x")

        max_x_i = np.argmin(np.abs(x - max_x))

        # Projdeme všechny možné úseky
        for i_1 in range(np.argmin(np.abs(x - min_x)), max_x_i):
            for i_2 in range(i_1 + 1, max_x_i):
                # Lineární regrese na úseku [i, j]
                _, _, r_value, _, _ = linregress(x[i_1:i_2 + 1], y[i_1:i_2 + 1])
                # slope, intercept, r_value, p_value, std_err

                # Hodnocení úseku: velikost okna a korelační koeficient
                window_size = i_2 - i_1 + 1
                if window_size >= min_length and abs(r_value) > threshold:
                    score = window_size / n_x * constant1 + abs(r_value) * constant2

                    # Uložení nejlepšího úseku
                    if score > best_score:
                        best_score = score
                        best_section = (i_1, i_2)
                        best_r = r_value

        return best_section, [best_score, abs(best_r)]

    elif calculation_type == 2:
        ind_min = np.where(x >= min_x)[0][0]
        ind_max = np.where(x <= max_x)[0][-1] + 1
        x, y = x[ind_min:ind_max], y[ind_min:ind_max]
        n_x = len(x)

        linear_sections = []
        r_coef = []

        while i_1 < n_x - 1:
            for i_2 in range(i_1 + 1, n_x):
                # Lineární regrese na úseku [i, j]
                _, _, r_value, _, _ = linregress(x[i_1:i_2 + 1], y[i_1:i_2 + 1])
                # slope, intercept, r_value, p_value, std_err

                # Pokud není lineární, ukonči smyčku
                if abs(r_value) < threshold:
                    if i_2 - i_1 > 1 and (i_2 - i_1 - 1) >= min_length:  # Přidej pouze úseky delší než 1 bod
                        linear_sections.append((i_1, i_2 - 1))
                        r_coef.append(abs(r_value))
                    i_1 = i_2  # Posuň začátek na nový úsek
                    break
            else:
                # Pokud jsme prošli až na konec
                if i_2 - i_1 > 1 and (n_x - i_1 - 1) >= min_length:
                    linear_sections.append((i_1, n_x - 1))
                    r_coef.append(abs(r_value))
                break

        if not linear_sections:
            return [], []
        else:
            best_index = np.argmax(r_coef)
            return linear_sections[best_index], r_coef[best_index]
    else:
        raise ValueError("Neplatný výpočetní typ")


def swap_lists(list1, list2):
    return list2, list1


out_put_folder = ".outputs"

# Název Excel souboru
excel_file = f'Values_tension.xlsx'

data_type = "T01"
sample_cross_section_area = (2.64 * 15.13)  # [mm2]

pair_by_displacement = False  # False - podle času // True - podle posunů
data_by_index = False  # False - interpolace podle času nebo posunů // True - podle indexu času nebo posunů
swap_II_and_III = True

do_tex = False

save_plot = True

file_type = "jpg"
out_dpi = 600

# Najít nejlepší lineární úsek
linear_method = 1  # 1 - klouzavé okno, 2 - multicriteriální okno
quality_threshold = 0.99999
con1 = 1
con2 = 0.98
step = 0.9999999

min_x_strain = 0
max_x_strain = 0.012

max_data_stain_limiter = 3  # [%] np.inf
max_linear_strain = 0.01

# Testy II a III musí být vůči testům hexagonů prohozeny
special_additional_information = {'T01_01-I_1s': 3}
# posunuté meření až u čtvté fotografie z důvodu opožděného zatěžování

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
if swap_II_and_III:
    type_2_sm, type_3_sm = swap_lists(type_2_sm, type_3_sm)
    type_2_la, type_3_la = swap_lists(type_2_la, type_3_la)

found_strains, found_stresses = [None] * len(folders), [None] * len(folders)
modules = []
all_datas = [None] * len(folders)

indexes = [type_1_sm, type_2_sm, type_3_sm, type_1_la, type_2_la, type_3_la]

for inds in indexes:

    for ind in inds:

        folder = folders[ind]

        path_displacement = os.path.join(folder_n_corr, folder, "virtualExtensometer_1",
                                         f"{folder}-virtExt_1_extension-tot (meters).txt")
        path_strain = os.path.join(folder_n_corr, folder, "virtualExtensometer_2",
                                   f"{folder}-virtExt_2_strain-total.txt")
        path_force = os.path.join(folder_measurements, "data_csv", f"{folder}.csv")

        experiment_name = folder

        # TODO
        # Změna názvů typu infillu dle stran hexagonů
        if swap_II_and_III:
            if data_type == "S01" or data_type == "T01":
                if "-II_" in experiment_name:
                    experiment_name = experiment_name.replace("-II_", "-III_")
                elif "-III_" in experiment_name:
                    experiment_name = experiment_name.replace("-III_", "-II_")

        data_frames = [experiment_name]

        if not os.path.isfile(path_strain):
            print("Strain file not found:", folder)
            found_strains[ind] = None
            found_stresses[ind] = None
            continue
        if not os.path.isfile(path_displacement):
            print("Displacement file not found:", folder)
            found_strains[ind] = None
            found_stresses[ind] = None
            continue
        if not os.path.isfile(path_force):
            print("Force file not found:", folder)
            found_strains[ind] = None
            found_stresses[ind] = None
            continue

        # NCORR data - strain, displacement
        data_strain = np.loadtxt(path_strain)
        data_displacement = np.loadtxt(path_displacement) * 1000  # m na mm

        if folder in special_additional_information:
            data_strain = data_strain[special_additional_information.get(folder, 0):]
            data_displacement = data_displacement[special_additional_information.get(folder, 0):]

        # CSV data - force, distance, time
        data_tension = pd.read_csv(path_force)

        # TIME
        photo_times = np.arange(0, len(data_strain))
        photos = np.arange(1, len(data_strain) + 1)

        data_distances = data_tension.iloc[:, 0].values
        # data_distances -= data_distances[0]
        data_force = data_tension.iloc[:, 1].values
        data_force -= data_force[0]
        data_time = data_tension.iloc[:, 2].values
        # data_time -= data_time[0]

        data_distances = np.hstack((0, data_distances))
        data_force = np.hstack((0, data_force))
        data_time = np.hstack((0, data_time))

        data_photos = np.full(len(data_time), np.nan)
        data_strain_long = np.full(len(data_time), np.nan)

        # Získání indexů, které jsou nejblíže hodnotám v druhém vektoru
        if pair_by_displacement:
            index_at_photo = np.abs(data_distances[:, np.newaxis] - data_displacement).argmin(axis=0)
        else:
            index_at_photo = np.abs(data_time[:, np.newaxis] - photo_times).argmin(axis=0)
        # v ideálním případě by měly být indexy 'index_at_photo' a 'index_at_displacement' stejné

        differences = np.diff(index_at_photo)

        # Tolerance
        tolerance = round(np.mean(differences[:len(differences) // 4]) * 0.2)

        # Zjistit index, do kterého jsou mezery stejné v rámci tolerance
        same_until = np.where(~np.isclose(differences, differences[0], atol=tolerance))[0]
        if same_until.size > 0:
            breakpoint_index = same_until[0]  # První index, kde se mezery liší
        else:
            breakpoint_index = len(differences)  # Všechny mezery jsou stejné

        same_index_at_photo = index_at_photo[:breakpoint_index + 1]  # Indexy fotek od bodu zlomu

        data_photos[same_index_at_photo] = photos[:breakpoint_index + 1]
        data_strain_long[same_index_at_photo] = data_strain[:breakpoint_index + 1]

        if data_by_index:
            found_times = data_time[same_index_at_photo]
            found_force = data_force[same_index_at_photo]
            found_distances = data_distances[same_index_at_photo]
            # found_force -= found_force[0]
        else:
            found_times = photo_times

            # Získání hodnot, které jsou interpolovány z druhého vektoru na základě prvního
            if pair_by_displacement:
                found_force = np.interp(data_displacement, data_distances, data_force)
                found_distances = np.interp(data_displacement, data_distances, data_distances)

            else:
                found_force = np.interp(photo_times, data_time, data_force)
                found_distances = np.interp(photo_times, data_time, data_distances)

        data_stress = data_force / sample_cross_section_area

        found_stress = found_force / sample_cross_section_area

        end_index = np.where(np.abs(np.diff(found_stress)) > np.max(found_stress) * 0.5)[0]
        if len(end_index) == 0:
            end_index = len(found_stress)
        else:
            end_index = end_index[0]

        index_max_strain = min(np.where(data_strain <= max_data_stain_limiter / 100, data_strain, np.inf).argmax(),
                               np.where(np.abs(np.diff(data_strain)) <= 0.02)[0].argmax(), end_index) + 1

        # Index nejbližší nižší hodnoty

        found_photos = photos[:index_max_strain]
        found_times = found_times[:index_max_strain]
        found_distances = found_distances[:index_max_strain]
        found_force = found_force[:index_max_strain]
        found_strain = data_strain[:index_max_strain]
        found_stress = found_stress[:index_max_strain]

        found_strains[ind] = found_strain
        found_stresses[ind] = found_stress

        index_max_strain = np.where(found_strain <= max_linear_strain, found_strain, np.inf).argmax()

        module = (found_stress[index_max_strain] - found_stress[1]) / (found_strain[index_max_strain] - found_strain[1])

        modules.append(module)

        # Celá data - DLOUHÁ
        data_frames.append(pd.DataFrame({'Indicative Photo': data_photos,
                                         'Time [s]': data_time,
                                         'Distance [mm]': data_distances,
                                         'Force [N]': data_force,
                                         'Indicative Strain [-]': data_strain_long,
                                         'Stress [MPa]': data_stress}))

        # Hodnoty ve chvíli fotek - KRÁTKÁ
        data_frames.append(pd.DataFrame({'Photo': found_photos,
                                         'Time [s]': found_times,
                                         'Distance [mm]': found_distances,
                                         'Force [N]': found_force,
                                         'Strain [-]': found_strain,
                                         'Stress [MPa]': found_stress}))

        all_datas[ind] = data_frames

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
    fig_m, axs_m = plt.subplots(1, 1, figsize=(6, 4))
elif len(indexes) == 2:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig_m, axs_m = plt.subplots(1, 2, figsize=(12, 4))
elif len(indexes) <= 4:
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    fig_m, axs_m = plt.subplots(2, 2, figsize=(12, 6))
elif len(indexes) <= 6:
    fig, axs = plt.subplots(2, 3, figsize=(12, 5))
    fig_m, axs_m = plt.subplots(2, 3, figsize=(12, 5))
else:
    fig, axs = plt.subplots(3, np.ceil(len(indexes) / 3), figsize=(12, 8))
    fig_m, axs_m = plt.subplots(3, np.ceil(len(indexes) / 3), figsize=(12, 8))
    i = int(len(indexes) - (3 * np.ceil(len(indexes) / 3)))
    if i < 0:
        axs[i:].remove()
        axs_m[i:].remove()
    elif i > 0:
        print("Error in creating subplots.")
        exit(2)

try:
    axs = axs.flatten()
    axs_m = axs_m.flatten()
except AttributeError:
    axs = [axs]
    axs_m = [axs_m]

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

    threshold = quality_threshold
    linear_sections = ()

    # Vizualizace

    for j in range(len(data_plot_x)):
        min_x_val = max(min_x_strain, np.min(data_plot_x[j]))
        max_x_val = max(min(max_x_strain, np.max(data_plot_x[j])), min_x_val)

        while not linear_sections:
            linear_sections, coefs = find_linear_section(linear_method, data_plot_x[j], data_plot_y[j],
                                                      min_x_val, max_x_val, threshold, con1, con2)
            threshold *= step
        start, end = linear_sections

        # index_max_strain = np.where(data_plot_x[i] < max_linear_strain, data_plot_x[i], np.inf).argmax()

        # module = (data_plot_x[j][index_max_strain] - data_plot_y[j][1]) / (
        #         data_plot_x[j][index_max_strain] - data_plot_x[j][1])
        module = (data_plot_y[j][end] - data_plot_y[j][start]) / (
                data_plot_x[j][end] - data_plot_x[j][start])

        modules.append(module)

        axs_m[n].plot(data_plot_x[j], data_plot_y[j], label="Data", alpha=0.45)
        axs_m[n].plot(data_plot_x[j][start:end + 1], data_plot_y[j][start:end + 1], linewidth=3, color="red")

    axs_m[n].set_title(f"Optimalizovaná detekce lineární části: {coefs[0] if isinstance(coefs, list) else coefs:.6f}",
                       fontsize=9)
    fig_m.tight_layout()

    mean_module = np.mean(modules)
    std_module = np.std(modules)

    print("\t", mean_module)
    print("\t", std_module)

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

# ##############################################################################
if not os.path.exists(out_put_folder):
    os.makedirs(out_put_folder, exist_ok=True)

try:
    excel_writer = pd.ExcelWriter(os.path.join(out_put_folder, excel_file), engine='xlsxwriter')
except PermissionError as e:
    print(f'\033[31;1;21mERROR\033[0m\n\tSoubor [{excel_file}] nelze upravovat, pravděpodobně je otevřen.'
          f'\n\tPOPIS: {e}')
    exit(10)
except (KeyError, Exception) as e:
    print(f'\033[31;1;21mERROR\033[0m\n\tSoubor [{excel_file}] nelze uložit.\n\tPOPIS: {e}')
    exit(11)

# Zápis dat do listů
for i, data in enumerate([all_datas[j] for j in
                          np.hstack([type_1_sm, type_2_sm, type_3_sm,
                                     type_1_la, type_2_la, type_3_la])]):

    if data is None:
        continue
    else:

        sheet_name = data[0]
        sheet_name = sheet_name.replace(f"{data_type}_", "").replace("_1s", "")

        # TODO #########################################################################################################
        # ##############################################################################################################
        # TODO #########################################################################################################
        # Uložení textu v listu
        text1 = f"Typ měření: {str(excel_file).replace('.xlsx', '')}"
        text2 = f"(Měření: {sheet_name})"
        description = "Obsah jednotlivých listů:"

        # Vytvoření listu pro popis
        df_description = pd.DataFrame({'Popis': [description]})

        # Zápis popisu na zvláštní list
        df_description.to_excel(excel_writer, sheet_name='Popis', index=False, startrow=5)

        worksheet = excel_writer.sheets['Popis']
        worksheet.write(0, 0, text1)
        worksheet.write(1, 0, text2)
        # TODO #########################################################################################################
        # ##############################################################################################################
        # TODO #########################################################################################################

        # Ukládání jednotlivých DataFrame na různá místa
        start_row = 0
        col_start = 0
        data[1].to_excel(excel_writer, sheet_name=sheet_name, startrow=start_row, startcol=col_start, index=False)
        col_start += len(data[1].columns)
        data[2].to_excel(excel_writer, sheet_name=sheet_name, startrow=start_row, startcol=col_start, index=False)

# Zavření Excel souboru
excel_writer.close()
print(f"Soubor byl úspěšně uložen do: [ {os.path.join(out_put_folder, excel_file)} ]")
# ##############################################################################

plt.show()
