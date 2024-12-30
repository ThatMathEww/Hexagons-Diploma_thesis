from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import zipfile
import h5py
import time
import cv2
import os


def swap_lists(list1, list2):
    return list2, list1

"""
ratio:
mean = 24.392700425124065
std = 0.24670669587238472
median = 24.292378586161387
"""

# Nastavení globální palety barev
custom_colors1 = ("dodgerblue", "red", "limegreen", "orange", "purple", "cyan", "pink", "black", "yellow", "magenta")
custom_colors2 = ['#78DCE8', '#FF6188', '#A9DC76', '#AB9DF2', '#FC9867', '#FFD866']  # Monokai Pro
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors1)

save_figures = False

do_tex = False

load_keypoints = False

file_type = "jpg"
out_dpi = 600

cut_spikes = True
data_type = "S01"  # "H01" # "H02"  # "S01"  # "M01"

# Název Excel souboru
excel_file = f'Values_bending.xlsx'

scale_m01 = True

mark_linear_part = True

# Definice velikosti okna pro klouzavý průměr
average_window_size_data = True
window_size_data = 5
average_window_size_plot = True
window_size_plot = 5

saved_data_name = "data_export_new.zip"
out_put_folder = ".outputs"

main_image_folder = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos'
folder_measurements = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\data'
folder_n_corr = r'C:\Programy\Ncorr\Ncorr_post_v2e\export'

########################################################################################################################
if do_tex:
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{lmodern, amsmath, amsfonts, amssymb, amsthm, bm}')
    # plt.rcParams['font.size'] = 14

images_folders = [name for name in [os.path.splitext(file)[0] for file in os.listdir(main_image_folder)]
                  if name.startswith(data_type)]

if data_type == "H01":
    """data_indexes_I = np.arange(0, 4 * 7, 5) + 3
    data_indexes_II = np.arange(0, 4 * 7, 5) + 2
    data_indexes_III = np.arange(0, 4 * 7, 5) + 1
    data_indexes_max = np.arange(0, 4 * 7, 5)
    data_indexes_can_norm = np.arange(0, 4 * 7, 5) + 4
    data_indexes_can_snapped = np.arange(30, 36)"""
    data_indexes_I = np.array([i for i in range(len(images_folders)) if "-I_" in images_folders[i]])
    data_indexes_II = np.array([i for i in range(len(images_folders)) if "-II_" in images_folders[i]])
    data_indexes_III = np.array([i for i in range(len(images_folders)) if "-III_" in images_folders[i]])
    data_indexes_max = np.array([i for i in range(len(images_folders)) if "-max_" in images_folders[i]])
    data_indexes_can_norm = np.array([i for i in range(len(images_folders)) if "_p" not in images_folders[i]
                                      and "max" not in images_folders[i] and "I" not in images_folders[i]])
    data_indexes_can_snapped = np.array([i for i in range(len(images_folders)) if "_p" in images_folders[i]
                                         and "max" not in images_folders[i] and "I" not in images_folders[i]])
    linear_part = [2, 3]

elif data_type == "H02":
    """data_indexes_I_K = np.arange(0, 8 * 6, 8) + 6
    data_indexes_II_K = np.arange(0, 8 * 6, 8) + 4
    data_indexes_III_K = np.arange(0, 8 * 6, 8) + 2
    data_indexes_I_N = np.arange(0, 8 * 6, 8) + 7
    data_indexes_II_N = np.arange(0, 8 * 6, 8) + 5
    data_indexes_III_N = np.arange(0, 8 * 6, 8) + 3
    data_indexes_max_N = np.arange(0, 8 * 6, 8) + 1
    data_indexes_max_K = np.arange(0, 8 * 6, 8) + 0"""
    data_indexes_I_K = np.array(
        [i for i in range(len(images_folders)) if "-I_" in images_folders[i] and "k" in images_folders[i]])
    data_indexes_II_K = np.array(
        [i for i in range(len(images_folders)) if "-II_" in images_folders[i] and "k" in images_folders[i]])
    data_indexes_III_K = np.array(
        [i for i in range(len(images_folders)) if "-III_" in images_folders[i] and "k" in images_folders[i]])
    data_indexes_I_N = np.array(
        [i for i in range(len(images_folders)) if "-I_" in images_folders[i] and "n" in images_folders[i]])
    data_indexes_II_N = np.array(
        [i for i in range(len(images_folders)) if "-II_" in images_folders[i] and "n" in images_folders[i]])
    data_indexes_III_N = np.array(
        [i for i in range(len(images_folders)) if "-III_" in images_folders[i] and "n" in images_folders[i]])
    data_indexes_max_N = np.array(
        [i for i in range(len(images_folders)) if "-max_" in images_folders[i] and "n" in images_folders[i]])
    data_indexes_max_K = np.array(
        [i for i in range(len(images_folders)) if "-max_" in images_folders[i] and "k" in images_folders[i]])
    linear_part = [3, 6]

    folders_nc = [name for name in [os.path.splitext(file)[0] for file in os.listdir(folder_n_corr)]
                  if name.startswith(data_type)]

    corner_1 = np.array([i for i in range(len(images_folders)) if
                         "12s" in images_folders[i] and "max" in images_folders[i] and images_folders[i] in folders_nc])
    corner_2 = np.array([i for i in range(len(images_folders)) if
                         "10s" in images_folders[i] and "max" not in images_folders[i] and images_folders[
                             i] in folders_nc])

    datas_dic = []

elif data_type == "S01":
    names = []

    data_indexes__I = np.array(
        [i for i in range(len(images_folders)) if "-I-" in images_folders[i] and "MAX" not in images_folders[i]])
    data_indexes__II = np.array(
        [i for i in range(len(images_folders)) if "-II-" in images_folders[i] and "MAX" not in images_folders[i]])
    data_indexes__III = np.array(
        [i for i in range(len(images_folders)) if "-III-" in images_folders[i] and "MAX" not in images_folders[i]])
    data_indexes__I_max = np.array(
        [i for i in range(len(images_folders)) if "-I-" in images_folders[i] and "MAX" in images_folders[i]])
    data_indexes__II_max = np.array(
        [i for i in range(len(images_folders)) if "-II-" in images_folders[i] and "MAX" in images_folders[i]])
    data_indexes__III_max = np.array(
        [i for i in range(len(images_folders)) if "-III-" in images_folders[i] and "MAX" in images_folders[i]])

    # Testy II a III musí být vůči testům hexagonů prohozeny
    data_indexes__II, data_indexes__III = swap_lists(data_indexes__II, data_indexes__III)
    data_indexes__II_max, data_indexes__III_max = swap_lists(data_indexes__II_max, data_indexes__III_max)

    data_indexes__I_O = np.array([i for i in range(len(images_folders)) if "-I-" in images_folders[i] and
                                  "MAX" not in images_folders[i] and "_O" in images_folders[i]])
    data_indexes__II_O = np.array([i for i in range(len(images_folders)) if "-II-" in images_folders[i] and
                                   "MAX" not in images_folders[i] and "_O" in images_folders[i]])
    data_indexes__III_O = np.array([i for i in range(len(images_folders)) if "-III-" in images_folders[i] and
                                    "MAX" not in images_folders[i] and "_O" in images_folders[i]])
    data_indexes__I_max_O = np.array([i for i in range(len(images_folders)) if "-I-" in images_folders[i] and
                                      "MAX" in images_folders[i] and "_O" in images_folders[i]])
    data_indexes__II_max_O = np.array([i for i in range(len(images_folders)) if "-II-" in images_folders[i] and
                                       "MAX" in images_folders[i] and "_O" in images_folders[i]])
    data_indexes__III_max_O = np.array([i for i in range(len(images_folders)) if "-III-" in images_folders[i] and
                                        "MAX" in images_folders[i] and "_O" in images_folders[i]])

    # Testy II a III musí být vůči testům hexagonů prohozeny
    data_indexes__II_O, data_indexes__III_O = swap_lists(data_indexes__II_O, data_indexes__III_O)
    data_indexes__II_max_O, data_indexes__III_max_O = swap_lists(data_indexes__II_max_O, data_indexes__III_max_O)

    data_indexes__I_G = np.array([i for i in range(len(images_folders)) if "-I-" in images_folders[i] and
                                  "MAX" not in images_folders[i] and "_S" in images_folders[i]])
    data_indexes__II_G = np.array([i for i in range(len(images_folders)) if "-II-" in images_folders[i] and
                                   "MAX" not in images_folders[i] and "_S" in images_folders[i]])
    data_indexes__III_G = np.array([i for i in range(len(images_folders)) if "-III-" in images_folders[i] and
                                    "MAX" not in images_folders[i] and "_S" in images_folders[i]])
    data_indexes__I_max_G = np.array([i for i in range(len(images_folders)) if "-I-" in images_folders[i] and
                                      "MAX" in images_folders[i] and "_S" in images_folders[i]])
    data_indexes__II_max_G = np.array([i for i in range(len(images_folders)) if "-II-" in images_folders[i] and
                                       "MAX" in images_folders[i] and "_S" in images_folders[i]])
    data_indexes__III_max_G = np.array([i for i in range(len(images_folders)) if "-III-" in images_folders[i] and
                                        "MAX" in images_folders[i] and "_S" in images_folders[i]])

    # Testy II a III musí být vůči testům hexagonů prohozeny
    data_indexes__II_G, data_indexes__III_G = swap_lists(data_indexes__II_G, data_indexes__III_G)
    data_indexes__II_max_G, data_indexes__III_max_G = swap_lists(data_indexes__II_max_G, data_indexes__III_max_G)

    data_indexes__I_W = np.array([i for i in range(len(images_folders)) if "-I-" in images_folders[i] and
                                  "MAX" not in images_folders[i] and "_B" in images_folders[i]])
    data_indexes__II_W = np.array([i for i in range(len(images_folders)) if "-II-" in images_folders[i] and
                                   "MAX" not in images_folders[i] and "_B" in images_folders[i]])
    data_indexes__III_W = np.array([i for i in range(len(images_folders)) if "-III-" in images_folders[i] and
                                    "MAX" not in images_folders[i] and "_B" in images_folders[i]])
    data_indexes__I_max_W = np.array([i for i in range(len(images_folders)) if "-I-" in images_folders[i] and
                                      "MAX" in images_folders[i] and "_B" in images_folders[i]])
    data_indexes__II_max_W = np.array([i for i in range(len(images_folders)) if "-II-" in images_folders[i] and
                                       "MAX" in images_folders[i] and "_B" in images_folders[i]])
    data_indexes__III_max_W = np.array([i for i in range(len(images_folders)) if "-III-" in images_folders[i] and
                                        "MAX" in images_folders[i] and "_B" in images_folders[i]])

    # Testy II a III musí být vůči testům hexagonů prohozeny
    data_indexes__II_W, data_indexes__III_W = swap_lists(data_indexes__II_W, data_indexes__III_W)
    data_indexes__II_max_W, data_indexes__III_max_W = swap_lists(data_indexes__II_max_W, data_indexes__III_max_W)

    linear_part = [1.2, 2.2]

elif data_type == "M01":
    data_indexes_glued = [0]
    data_indexes_whole = [2]
    linear_part = [1, 2]
elif data_type == "T01":
    raise ValueError(f"Tah je v jiném souboru: {data_type}")


else:
    raise ValueError(f"Neznámý typ dat: {data_type}")

all_datas = []
########################################################################################################################
########################################################################################################################

tot_folders = len(images_folders)
for exp, current_image_folder in enumerate(images_folders):

    print(f"\nNačítání uložených dat: ' \033[94;1m{current_image_folder}\033[0m ' -  [ {exp + 1} / {tot_folders}]")

    correlation_points, tracked_points, tracked_rotations, distances, forces, photo_indexes = [None] * 6
    dataset_1, dataset_2, dataset_3, time_stamps, start_value = [None] * 5
    scale: float = 1

    current_folder_path = os.path.join(main_image_folder, current_image_folder)
    zip_files = [f for f in os.listdir(current_folder_path)
                 if os.path.isfile(os.path.join(current_folder_path, f)) and f.lower().endswith(".zip")]

    zip_file_name = os.path.join(current_folder_path, saved_data_name)
    if saved_data_name not in zip_files:
        all_datas.append(None)
        if data_type == "H02":
            datas_dic.append(None)
        print(f'\033[31;1;21mERROR\033[0m\n\tVe složce [{current_image_folder}] se nenachází daný soubor ZIP')
        continue

    try:
        # Načtení dat z zip archivu
        with zipfile.ZipFile(zip_file_name, 'r') as zipf:

            # Zjištění, zda je zip soubor prázdný
            zip_list = zipf.namelist()

            if not zip_list:
                raise Exception(f"\033[31;1;21mERROR:\033[0m Zip file [{zip_file_name}] is empty.")

            #  ######################################################################################################  #
            #  ############################################     CSV      ############################################  #
            #  ######################################################################################################  #
            # Načítání souboru CSV
            csv_file_name = f"{current_image_folder}.csv"
            if csv_file_name in zip_list:
                path_to_csv = zipf.open(csv_file_name)
            else:
                path_to_csv = os.path.join(folder_measurements, "data_csv", csv_file_name)
                print(f'\033[33;1;21mWARRNING\033[0m\n\tV uložených datech se nenachází soubor: "{csv_file_name}"'
                      f'\n\t➤ Pokus o načtení souboru ze složky')

            df = pd.read_csv(path_to_csv)  # DATAFRAME

            zr = 5
            d_len = df.shape[0]
            zr = max(min(zr, d_len // 3), 3)
            z2 = max(zr // 2, 1)

            distances = df.iloc[:, 0].values  # - posun
            forces = -(df.iloc[:, 1].values + df.iloc[:, 2].values)  # - celková síla
            forces -= forces[:zr].mean()
            photo_indexes = df[df['Photos'].notna()].index

            # photo_indexes = photo_indexes[np.where(photo_indexes <= len(distances))[0]]

            # Najdi indexy, kde je okno rovno `pocet_podminka`
            start_index = np.max(np.where(forces[:max(min(int(np.where(np.convolve(np.where(np.abs(np.array(
                [np.mean(forces[s - z2:s + z2]) for s in range(z2, d_len // 3)])[zr:] - np.median(
                forces[1:zr * 2 + 1])) >= np.mean(
                np.array([np.std(forces[s - z2:s + z2]) for s in range(z2, d_len // 300)])[zr:]
                * 1.5), 1, 0), np.ones(int(np.ceil(d_len * 0.1))), mode='valid') == np.ceil(d_len * 0.1))[0][0]) - zr,
                                                          d_len), 0) + 1] <= 0)[0])

            # Najdi indexy, kde je rozdíl menší než -5
            snap_index = np.where(np.diff(forces) <= -5)[0]

            if data_type == "H01" and cut_spikes and len(snap_index) > 0:
                if forces[snap_index[0]] > forces[snap_index[0] - 1] + 5:
                    forces[snap_index[0]] = forces[snap_index[0] - 1]

                if np.mean(forces[snap_index[0] + 2:snap_index[0] + 7]) - 10 > forces[snap_index[0] + 1] < np.mean(
                        forces[snap_index[0] + 2:snap_index[0] + 7]) + 10:
                    forces[snap_index[0] + 1] = np.mean(forces[snap_index[0] + 2:snap_index[0] + 7])

            # Průměrování dat
            if isinstance(window_size_data, int) and window_size_data >= 1 and average_window_size_data:
                # Vytvoření průměrového filtru
                window = np.ones(window_size_data) / window_size_data

                # Aplikace klouzavého průměru
                distances = distances[:-window_size_data // 2]
                forces = np.convolve(forces, window, mode='same')[:-window_size_data // 2]
                p = df['Photos'].notna()[:-window_size_data // 2]
                photo_indexes = df['Photos'][:-window_size_data // 2]
                photo_indexes = photo_indexes[p].index

            # Nalezení nejbližší vyšší hodnoty
            """beginning = np.where(photo_indexes >= start_index)[0][
                np.argmin(photo_indexes[photo_indexes >= start_index])]"""
            # Nalezení indexu nejbližší vyšší hodnoty
            beginning = np.argmax(photo_indexes >= start_index)

            #  ######################################################################################################  #
            #  ###########################################     NCORR      ###########################################  #
            #  ######################################################################################################  #
            if data_type == "H02":
                path_strain = os.path.join(folder_n_corr, current_image_folder, "virtualExtensometer_1",
                                           f"{current_image_folder}-virtExt_1_strain-total.txt")

                if os.path.isfile(path_strain):
                    try:
                        datas_dic.append(np.loadtxt(path_strain)[beginning:] *
                                         (100 if "strain" in path_strain else 1000))
                    except (ValueError, Exception):
                        datas_dic.append(None)
                else:
                    datas_dic.append(None)

            #  ######################################################################################################  #
            #  ############################################      H5      ############################################  #
            #  ######################################################################################################  #
            # Načtení .h5 souboru
            with zipf.open('data.h5') as h5_file:
                with h5py.File(h5_file, 'r') as file:
                    # Seznam skupin v souboru
                    group_names = list(file.keys())

                    if 'variables' in group_names:
                        # Načtení skupiny, ve které jsou uložené proměnné
                        data_group = file['variables']

                        # Načtení jednotlivých proměnných z datasetů a uložení do seznamu
                        dataset_1 = [data_group[f'var{i:05d}'][()] for i in range(len(data_group))]

                        # Slovník statusů (atributů)
                        for i in [key for key in file.keys() if key.startswith('dictionary_')]:
                            dataset_1 += [{key: value for key, value in file[i].attrs.items()}]

                    if 'additional_variables' in group_names:
                        dataset_3 = {key: value[:] for key, value in file['additional_variables'].items()}

                    dataset_2 = dict(data_correlation=None, data_point_detect=None)
                    for group_name in dataset_2.keys():
                        if group_name in group_names:
                            data_group = file[group_name]
                            dataset_2[group_name] = [[dataset[:] for dataset in subgroup.values()]
                                                     for subgroup in data_group.values()]
                file.close()
            h5_file.close()

            angle_correction_matrix, photos_times = None, None
            datasets = dict(Correlation=False, Tracked_points=False, Forces=False, Others=False)

            try:
                if isinstance(dataset_1, list) and len(dataset_1) >= 3:
                    scale = dataset_1[2]

                if dataset_2['data_correlation'] is not None:
                    correlation_points = dataset_2['data_correlation']
                    datasets['Correlation'] = True

                if dataset_2['data_point_detect'] is not None:
                    [tracked_points, tracked_rotations] = dataset_2['data_point_detect']
                    datasets['Tracked_points'] = True

                if isinstance(dataset_3, dict) and len(dataset_3) > 0:
                    for name in dataset_3.keys():
                        globals()[name] = dataset_3.get(name, None)
                    datasets['Others'] = True

                if datasets['Correlation']:
                    correlation_points = [
                        cv2.transform(np.float64(point[0]).reshape(1, 2, 2), angle_correction_matrix).reshape(2, 2) for
                        point in correlation_points]

                if datasets['Tracked_points']:
                    tracked_points = [
                        cv2.transform(np.float64(point).reshape(1, -1, 2), angle_correction_matrix).reshape(-1, 2)
                        for point in tracked_points]

                """scaled_vector2 = np.interp(
                    np.linspace(0, 1, len(distances)),
                    np.linspace(0, 1, len(correlation_points)),
                    np.float64([c[0][0, 1] for c in correlation_points])
                )
                dd_ = np.array([distances[i + 1] - distances[i] for i in range(len(distances) - 1)])
                sd_ = np.array([scaled_vector2[i + 1] - scaled_vector2[i] for i in range(len(scaled_vector2) - 1)])"""

                if all(v is not None for v in (distances, forces, photo_indexes)):
                    ratio = (np.linalg.norm(correlation_points[0][0, 1] - correlation_points[-1][0, 1])
                             / np.linalg.norm(distances[0] - distances[-1]))
                    distances *= ratio
                    distances = (distances - distances[0]) * scale
                    start_value = distances[start_index]
                    distances = distances - start_value  # Stanovení 0 pozice zatěžovnání
                    datasets['Forces'] = True

            except (ValueError, Exception) as e:
                print(f'\033[31;1;21mERROR\033[0m\n\tSelhalo přiřazení hodnot uložených dat\n\tPOPIS: {e}')
                continue

            #  ######################################################################################################  #
            #  ############################################     TIME     ############################################  #
            #  ######################################################################################################  #

            load_photos_time = False

            try:
                time_values = np.float64([0.0 if not np.isnan(t) else np.nan for t in df['Photos'].values])

                if load_photos_time:
                    if "image_folder/" in zip_list:
                        # Převod času z GMT na lokální čas změny fotek v ZIPu a uložení do seznamu
                        time_stamps = [int(time.mktime(zipf.getinfo(file).date_time + (0, 0, 0))) for file
                                       in [name for name in zip_list if name.startswith("image_folder/")][1:]]

                        """t1 = [os.path.getmtime(os.path.join(current_folder_path, "modified", i)) for i in
                             os.listdir(os.path.join(current_folder_path, "modified"))][1:]
                        t2 = [0] + [t1[i + 1] - t1[i] for i in range(len(t1) - 1)]"""
                        if len(time_stamps) < 2:
                            raise Exception("Minimální počet fotek pro tvorbu časových razítek je 2.")
                        if len(time_stamps) - 1 != int(np.nanmax(df['Photos'].values)):
                            raise Exception("Neshodujhe se počet fotek v zipu s data z csv.")

                        # time_stamps = time_stamps[beginning:]
                        time_stamps = np.float64(
                            [0] + [time_stamps[i + 1] - time_stamps[i] for i in range(len(time_stamps) - 1)])
                        time_stamps[1:-1] = np.median(time_stamps[1:-1])
                    else:
                        print("\033[33;1;21mWARRNING\033[0m\n\t - V souboru ZIP se nenachází fotografie")
                else:
                    time_stamps = np.array(photos_times)
                    time_stamps[1:-1] = np.median(time_stamps[1:-1])

                t = np.int64(
                    [0] + [photo_indexes[i + 1] - photo_indexes[i] for i in range(len(photo_indexes) - 1)])
                t[1:-1] = np.median(t[1:-1])
                time_stamps[-1] = time_stamps[1] * (t[-1] / t[1])
                time_stamps = [np.sum(time_stamps[:i + 1]) for i in range(len(time_stamps))]

                nan_indices = np.isnan(time_values)  # Najděte indexy NaN hodnot
                time_values[~nan_indices] = time_stamps
                time_stamps = time_values.copy()  # Vytvořte kopii vektoru pro interpolaci
                # Nahraďte NaN hodnoty interpolovanými hodnotami
                time_stamps[nan_indices] = np.interp(np.flatnonzero(nan_indices),
                                                     np.flatnonzero(~nan_indices), time_values[~nan_indices])
                time_stamps = time_stamps[start_index:] - time_stamps[start_index]
                # time_stamps = [t - time_stamps[0] for t in time_stamps]

                if isinstance(window_size_data, int) and window_size_data >= 1 and average_window_size_data:
                    time_stamps = time_stamps[:-window_size_data // 2]


            except Exception as e:
                print("\033[33;1;21mWARRNING\033[0m\n\t - "
                      f"Chyba načtení časového nastavení měření ze složky: [{current_image_folder}]\n\tPOPIS: {e}")
                continue

        zipf.close()
    except (KeyError, Exception) as e:
        print(f'\033[31;1;21mERROR\033[0m\n\tSelhalo načtení uložených dat\n\tPOPIS: {e}')
        continue

    """try:
        def poly_function(poly_x, coefficients, start_ind, end_ind=""):
            function = "y = "
            # Získání počtu koeficientů
            len_c = len(coefficients)
            poly_y = np.zeros_like(poly_x)
            # Výpočet hodnoty y_fit
            for i in range(len_c):
                poly_y += coefficients[i] * poly_x ** (len_c - i - 1)
                if (len_c - i - 1) == 0:
                    function += f"{coefficients[i]:.5f}" if -0.00001 < coefficients[
                        i] < 0.00001 else f"{coefficients[i]:.5f}"
                elif (len_c - i - 1) == 1:
                    function += f"{coefficients[i]:.5e}*x + " if -0.00001 < coefficients[i] < 0.00001 \
                        else f"{coefficients[i]:.5f}*x + "
                else:
                    function += f"{coefficients[i]:.5e}*x^{len_c - i - 1} + " if -0.00001 < coefficients[i] < 0.00001 \
                        else f"{coefficients[i]:.5f}*x^{len_c - i - 1} + "
            if function.endswith("+ "):
                function = function[:-2]
            function += f";[ {poly_x[0]} : {poly_x[-1]} ];[ {start_ind} : {end_ind} ]"
            return poly_y, function, coefficients


        x_fit = []
        y_fit = []
        funcs = []
        coefs = []
        mses = []
        if snap_index.size > 0:
            snap_index = snap_index[0]
            mean_values2 = np.array([np.mean(forces[s:s + 2]) for s in range(snap_index)])
            mean_values2[0] = forces[0]
            mean_values2[-1] = forces[snap_index - 1]
            mean_values2[-2] = (mean_values2[-1] + mean_values2[-3]) / 2

            differ_index = np.where(np.diff(forces[snap_index:]) <= -0.3)[0][[0, -1]] + snap_index

            d = [np.mean(forces[s:s + 2]) for s in range(differ_index[0], differ_index[1] + 1)]
            d = np.array(
                [np.mean(d[s:s + 2]) for s in range(len(d))])
            mean_values2 = np.append(mean_values2, d)

            d = [np.mean(forces[s:s + 2]) for s in range(differ_index[1] + 1, d_len)]
            mean_values2 = np.append(mean_values2, d)

            x_ = distances[start_index:snap_index - 1]
            y = mean_values2[start_index:snap_index - 1]

            y_, func, c_ = poly_function(x_, np.polyfit(x_, y, 5), start_index - start_index,
                                         snap_index - 1 - start_index)
            coefs.append(c_)
            funcs.append(func)
            x_fit.append(x_)
            y_fit.append(y_)

            # Výpočet kvadratické chyby
            mse = np.mean((y - y_) ** 2)
            mses.append(mse)
            # print("\nMean Squared Error (MSE):", mse)

            x_ = distances[snap_index + 1:]
            y = mean_values2[snap_index + 1:]
            y_, func, c_ = poly_function(x_, np.polyfit(x_, y, 5), snap_index - 1 - start_index,
                                         len(distances) - start_index)
            coefs.append(c_)
            funcs.append(func)
            x_fit.append(x_)
            y_fit.append(y_)

            # Výpočet kvadratické chyby
            mse = np.mean((y - y_) ** 2)
            mses.append(mse)
            # print("Mean Squared Error (MSE):", mse)

        else:
            mean_values2 = np.array([np.mean(forces[s:s + 2]) for s in range(d_len)])
            mean_values2[0] = forces[0]
            x_fit.append(distances[start_index:])
            y_, func, c_ = poly_function(distances[start_index:],
                                         np.polyfit(distances[start_index:], mean_values2[start_index:], 5),
                                         start_index - start_index, len(distances) - start_index)
            coefs.append(c_)
            funcs.append(func)
            y_fit.append(y_)

            # Výpočet kvadratické chyby
            mse = np.mean((mean_values2[start_index:] - y_fit[0]) ** 2)
            mses.append(mse)
            # print("\nMean Squared Error (MSE):", mse)

    except (ValueError, Exception) as e:
        print(f'\033[31;1;21mERROR\033[0m\n\tSelhalo vytvoření křivek\n\tPOPIS: {e}')
        continue"""

    try:
        data_frames = [current_image_folder]

        photos = np.arange(beginning, len(photo_indexes), 1)  # int(np.nanmax(df['Photos'].values)) + 1
        time_values = time_stamps[photo_indexes - start_index][beginning:]

        data_frames.append(pd.DataFrame({'Photo': photos,
                                         'Time [s]': time_values}))

        if datasets['Correlation']:
            data = np.float64([np.mean(c, axis=0) for c in correlation_points])[:len(photo_indexes)]
            data_x = (data[beginning:, 0] - data[0, 0]) * scale
            data_y = (data[beginning:, 1] - start_value - data[0, 1]) * scale
            """dat = distances[photo_indexes][beginning:]
            d1 = np.median([data_y[i+1] - data_y[i] for i in range(len(data_y)-1)])
            d2 = np.median([dat[i+1] - dat[i] for i in range(len(dat)-1)])
            d3 = np.mean([dat[i] - data_y[i] for i in range(len(dat))])"""
            # Vytvoření datových rámce pro listy
            data_frames.append(pd.DataFrame({'X [mm]': data_x,
                                             'Y [mm]': distances[photo_indexes][beginning:]}))

        if datasets['Forces']:
            # Vytvoření datových rámce pro listy
            data_frames.append(pd.DataFrame({'Distance [mm]': distances[photo_indexes[beginning:]],
                                             'Force [N]': forces[photo_indexes[beginning:]]}))

        if datasets['Forces']:
            # Vytvoření datových rámce pro listy
            data_frames.append(pd.DataFrame({'Photo': df['Photos'].values[start_index:len(distances)],
                                             'Time [s]': time_stamps,
                                             'Distance [mm]': distances[start_index:],
                                             'Force [N]': forces[start_index:]}))

        if datasets['Others']:
            pass

        if datasets['Tracked_points'] and load_keypoints:

            len_points = len(tracked_points[0])
            len_photos = len(tracked_points)

            data = [(np.float64([tracked_points[i][j] for i in range(len_photos)]),
                     np.float64([tracked_rotations[i][j] for i in range(len_photos)])) for j in range(len_points)]

            data = [(np.float64([d[0][i] - d[0][0] for i in range(len_photos)]) * scale, d[1]) for d in data]

            # Vytvoření datových rámce pro listy
            df_tr = pd.DataFrame()

            # Přidání tří sloupců ve smyčce
            for i in range(len_points):  # Přidáme tři skupiny sloupců
                df_temp = pd.DataFrame(np.vstack((data[i][0][:, 0], data[i][0][:, 1], data[i][1])).T[beginning:],
                                       columns=[f'Point_{i + 1} - {v}' for v in ('X [mm]', 'Y [mm]', 'Rotation [rad]')])
                df_tr = pd.concat([df_tr, df_temp], axis=1)
            data_frames.append(df_tr)
        else:
            data_frames.append([])

        all_datas.append(data_frames)

    except (ValueError, Exception) as e:
        print(f'\033[31;1;21mERROR\033[0m\n\tSoubor [{current_image_folder}] se nepovedlo načíst.\n\tPOPIS: {e}')
        continue

print("\n\033[33;1mNačítání hotovo.\033[0m")

"""plt.figure()
data_plot = all_datas[0]
d = np.array(data_plot[-1])
plt.plot(d[:, 0], d[:, 1], label=data_plot[0])
# [plt.plot(d[:, j], d[:, j + 1], label=data_plot[0]) for j in range(0, d.shape[1], 2)]
plt.legend()
plt.show()"""

"""for j, dat in enumerate(all_datas):
    plt.figure()
    plt.title(images_folders[j])
    d = np.array(dat[-1])
    [plt.plot(d[:, j], d[:, j + 1], ) for j in range(0, d.shape[1], 2)]
    plt.show()"""

# TODO INDEXES
if data_type == "H01":
    indexes = [data_indexes_I, data_indexes_II, data_indexes_III, data_indexes_max]
    indexes = [data_indexes_can_norm]
    # indexes = [data_indexes_can_snapped]
elif data_type == "H02":
    # indexes = []
    indexes = [data_indexes_I_K, data_indexes_II_K, data_indexes_III_K, data_indexes_max_K]
elif data_type == "S01":
    """indexes = [np.hstack((data_indexes__I_O, data_indexes__II_O, data_indexes__III_O)),
               np.hstack((data_indexes__I_W, data_indexes__II_W, data_indexes__III_W)),
               np.hstack((data_indexes__I_G, data_indexes__II_G, data_indexes__III_G))
               ]"""
    indexes = [data_indexes__I_max, data_indexes__II_max, data_indexes__III_max]
    # indexes = [data_indexes__I, data_indexes__II, data_indexes__III]

elif data_type == "M01":
    indexes = [data_indexes_glued, data_indexes_whole]

# Vytvoření subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 6)) if 5 > len(indexes) >= 3 else plt.subplots(1, 2, figsize=(12, 4)) \
    if len(indexes) == 2 else plt.subplots(1, 1, figsize=(6, 4))
try:
    axs = axs.flatten()
except AttributeError:
    axs = [axs]

if len(indexes) == 3:
    axs[-1].remove()

for i in range(len(indexes)):
    try:
        [axs[i].plot(all_datas[j][-2].iloc[:, 2].values, all_datas[j][-2].iloc[:, 3].values,
                     c='gray', lw=1, alpha=0.5, zorder=30) for j in np.hstack(indexes[:i] + indexes[i + 1:]) if
         all_datas[j] is not None]
    except ValueError:
        pass

    [axs[i].plot(all_datas[j][-2].iloc[:, 2].values, all_datas[j][-2].iloc[:, 3].values, lw=1.5,
                 label=all_datas[j][0], zorder=40 - len(indexes[i]) - c) for c, j in enumerate(indexes[i]) if
     all_datas[j] is not None]

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

    axs[i].set_xlabel('Displacement [mm]')
    axs[i].set_ylabel('Force [N]')

    axs[i].set_aspect('auto', adjustable='box')

handles, labels = axs[0].get_legend_handles_labels()
labels = [f"H1_{l + 1:02d}_B2" for l in range(len(labels))]  # TODO LABELS
fig.legend(handles, labels, fontsize=8, borderaxespad=0, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=10)

fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.3)
if save_figures:
    plt.savefig(f"{out_put_folder}/{data_type}_multipleplot.{file_type}", format=file_type, dpi=out_dpi, bbox_inches='tight')

# plt.tight_layout()

########################################################################################################################

# Vytvoření subplots
if mark_linear_part:
    fig, axs = plt.subplots(2, 2, figsize=(12, 6)) if 5 > len(indexes) >= 3 else plt.subplots(1, 2, figsize=(12, 4)) \
        if len(indexes) == 2 else plt.subplots(1, 1, figsize=(6, 4))
    try:
        axs = axs.flatten()
    except AttributeError:
        axs = [axs]

    if len(indexes) == 3:
        axs[-1].remove()

    for i in range(len(indexes)):
        try:
            [axs[i].plot(all_datas[j][-2].iloc[:, 2].values, all_datas[j][-2].iloc[:, 3].values, c='dodgerblue',
                         lw=1,
                         alpha=0.5, zorder=4) for j in indexes[i] if all_datas[j] is not None]
        except ValueError:
            pass

        [axs[i].plot(
            all_datas[j][-2].iloc[:, 2].values[(linear_part[0] <= all_datas[j][-2].iloc[:, 2].values) & (
                    all_datas[j][-2].iloc[:, 2].values <= linear_part[1])],
            all_datas[j][-2].iloc[:, 3].values[(linear_part[0] <= all_datas[j][-2].iloc[:, 2].values) & (
                    all_datas[j][-2].iloc[:, 2].values <= linear_part[1])],
            c='red', lw=1.2, alpha=1, zorder=5) for j in indexes[i] if all_datas[j] is not None]

        axs[i].grid(color="lightgray", linewidth=0.5, zorder=0)
        for axis in ['top', 'right']:
            axs[i].spines[axis].set_linewidth(0.5)
            axs[i].spines[axis].set_color('lightgray')

        if axs[i].get_xlim()[1] % axs[i].get_xticks()[-1] == 0:
            axs[i].spines['right'].set_visible(False)
        if axs[i].get_ylim()[1] % axs[i].get_yticks()[-1] == 0:
            axs[i].spines['top'].set_visible(False)

        axs[i].yaxis.set_minor_locator(AutoMinorLocator())
        axs[i].xaxis.set_minor_locator(AutoMinorLocator())

        axs[i].tick_params(axis='both', which='minor', direction='in', width=0.5, length=2.5, zorder=5,
                           color="black")
        axs[i].tick_params(axis='both', which='major', direction='in', width=0.8, length=5, zorder=5, color="black")

        # axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # axs[i].legend(fontsize=8, bbox_to_anchor=(0.5, -0.25), loc="center", borderaxespad=0, ncol=4)

        axs[i].set_xlabel('Displacement [mm]')
        axs[i].set_ylabel('Force [N]')

        axs[i].set_aspect('auto', adjustable='box')

    # plt.tight_layout()
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.3)

    fig, ax = plt.subplots(figsize=(5.2, 3))

    linear_lines = []

    for i in range(len(indexes)):
        [ax.plot(all_datas[j][-2].iloc[:, 2].values, all_datas[j][-2].iloc[:, 3].values, c='dodgerblue', lw=1,
                 alpha=0.5, zorder=4) for j in indexes[i] if all_datas[j] is not None]

        """l1 = np.array([all_datas[j][-2].iloc[:, 2].values[(linear_part[0] <= all_datas[j][-2].iloc[:, 2].values) & (
                all_datas[j][-2].iloc[:, 2].values <= linear_part[1])] for j in indexes[i] if all_datas[j] is not None])
        l2 = np.array([all_datas[j][-2].iloc[:, 3].values[(linear_part[0] <= all_datas[j][-2].iloc[:, 2].values) & (
                all_datas[j][-2].iloc[:, 2].values <= linear_part[1])] for j in indexes[i] if all_datas[j] is not None])

        [ax.plot(a, b, c='red', lw=1.2, alpha=1, zorder=5) for a, b in zip(l1, l2)]"""

        [ax.plot(all_datas[j][-2].iloc[:, 2].values[(linear_part[0] <= all_datas[j][-2].iloc[:, 2].values) & (
                all_datas[j][-2].iloc[:, 2].values <= linear_part[1])],
                 all_datas[j][-2].iloc[:, 3].values[(linear_part[0] <= all_datas[j][-2].iloc[:, 2].values) & (
                         all_datas[j][-2].iloc[:, 2].values <= linear_part[1])],
                 c='red', lw=1.2, alpha=1, zorder=5) for j in indexes[i] if all_datas[j] is not None]

        """linear_lines.append(np.array(
            [all_datas[j][-2].iloc[:, 3].values[all_datas[j][-2].iloc[:, 2].values <= linear_part[1]][-1] for j in
             indexes[i] if all_datas[j] is not None]) * (50 ** 3) / np.array(
            [all_datas[j][-2].iloc[:, 2].values[all_datas[j][-2].iloc[:, 2].values <= linear_part[1]][-1] for j in
             indexes[i] if all_datas[j] is not None]) * 48 * (1 / 12 * 15.13 * 2.64 ** 3))"""

        if data_type == 'S01':
            f = np.array([all_datas[j][-2].iloc[:, 3].values[all_datas[j][-2].iloc[:, 2].values <= linear_part[1]][
                              -1] for j in
                          indexes[i] if all_datas[j] is not None]) * (50 ** 3)
            w = np.array([all_datas[j][-2].iloc[:, 2].values[
                              all_datas[j][-2].iloc[:, 2].values <= linear_part[1]][-1] for j in
                          indexes[i] if all_datas[j] is not None]) * 48 * (1 / 12 * 15.13 * 2.64 ** 3)


        else:
            """f = np.mean([l[-1] - l[0] for l in l1])
            w = np.mean([l[-1] - l[0] for l in l2])"""
            pass

        try:
            e = f / w

            linear_lines.append(e)

            print(f"\n\t {np.mean(e):.3f}")
            print(f"\t {np.std(e):.3f}")
        except Exception as e:
            pass

    ax.grid(color="lightgray", linewidth=0.5, zorder=0)
    for axis in ['top', 'right']:
        ax.spines[axis].set_linewidth(0.5)
        ax.spines[axis].set_color('lightgray')

    if ax.get_xlim()[1] % ax.get_xticks()[-1] == 0:
        ax.spines['right'].set_visible(False)
    if ax.get_ylim()[1] % ax.get_yticks()[-1] == 0:
        ax.spines['top'].set_visible(False)

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(axis='both', which='minor', direction='in', width=0.5, length=2.5, zorder=5, color="black")
    ax.tick_params(axis='both', which='major', direction='in', width=0.8, length=5, zorder=5, color="black")

    # ax.legend(fontsize=8, bbox_to_anchor=(0.5, -0.25), loc="center", borderaxespad=0, ncol=4)

    ax.set_xlabel('Displacement [mm]')
    ax.set_ylabel('Force [N]')

    ax.set_aspect('auto', adjustable='box')

    # plt.tight_layout()
    fig.subplots_adjust(bottom=0.3, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.3)
if save_figures:
    plt.savefig(f"{out_put_folder}/{data_type}_singleplot_linearpart.{file_type}", format=file_type, dpi=out_dpi,
                bbox_inches='tight')

########################################################################################################################

if data_type == "H01":
    datas_pack = zip(("A", "B1", "B2"),
                     # ("I", "II", "III") // ("MAX", "NORM", "SNAPPED")
                     # (data_indexes_I,data_indexes_II,data_indexes_III),
                     (data_indexes_max, data_indexes_can_norm, data_indexes_can_snapped),
                     # (data_indexes_I, data_indexes_II, data_indexes_III)
                     ("dodgerblue", "red", "limegreen"))
elif data_type == "H02":
    datas_pack = zip(("Infill type 1", "Infill type 2"),
                     # ("I-K", "II-K", "III-K", "I-N", "II-N", "III-N") // ("MAX-K", "MAX-N")
                     # (data_indexes_I_N, data_indexes_I_K),
                     # (data_indexes_I_K, data_indexes_II_K, data_indexes_III_K) //
                     (data_indexes_max_K, data_indexes_max_N),
                     ("dodgerblue", "red", "limegreen"))
elif data_type == "S01":
    datas_pack = zip(("Orange", "White", "Gray"),
                     #                           ("Infill I", "Infill II", "Infill III") // ("Orange", "White", "Gray")
                     #                           ("I-K", "II-K", "III-K", "I-N", "II-N", "III-N") // ("MAX-K", "MAX-N")

                     # (np.hstack((data_indexes__I_O, data_indexes__I_G)),
                     # np.hstack((data_indexes__II_O, data_indexes__II_G)),
                     # np.hstack((data_indexes__III_O, data_indexes__III_G))),

                     (np.hstack((data_indexes__I_O, data_indexes__II_O, data_indexes__III_O)),
                      np.hstack((data_indexes__I_W, data_indexes__II_W, data_indexes__III_W)),
                      np.hstack((data_indexes__I_G, data_indexes__II_G, data_indexes__III_G))),

                     # (data_indexes__I, data_indexes__II, data_indexes__III) //
                     # (data_indexes__I_max, data_indexes__II_max, data_indexes__III_max),

                     ("Orange", "red", "dodgerblue"))
elif data_type == "M01":
    datas_pack = zip(("M-01", "M-02"),
                     (data_indexes_glued, data_indexes_whole),
                     ("dodgerblue", "red"))

    datas_y = []

fig, ax = plt.subplots(figsize=(5.2, 3))
fig2, ax2 = plt.subplots(figsize=(5.2, 3))

mean_std = []

for n, (name, curve_index, color) in enumerate(datas_pack):
    datas = [all_datas[j] for j in curve_index if all_datas[j] is not None]
    data_plot_x = np.array(
        [[x[-2].iloc[i, 2] for x in datas] for i in range(np.min([x[-2].shape[0] for x in datas]))])
    data_plot_y = np.array(
        [[y[-2].iloc[i, 3] for y in datas] for i in range(np.min([y[-2].shape[0] for y in datas]))])

    if data_type == "M01":
        datas_y.append(data_plot_y)

    data_mean_x = np.mean(data_plot_x, axis=1)
    data_mean_y = np.mean(data_plot_y, axis=1)
    data_max = np.max(data_plot_y, axis=1)
    data_min = np.min(data_plot_y, axis=1)
    data_std = np.std(data_plot_y, axis=1)
    mean_std.extend(data_std)

    if average_window_size_plot and isinstance(window_size_plot, int) and window_size_plot >= 1:
        # Vytvoření průměrového filtru
        window = np.ones(window_size_plot) / window_size_plot

        """
        # Převod dat na pandas DataFrame
        df = pd.DataFrame({'y': y})

        # Vytvoření klouzavého průměru
        window_size = 10
        y_smooth = df['y'].rolling(window=window_size).mean()
        """

        # Aplikace klouzavého průměru
        data_mean_x = data_mean_x[:-window_size_plot // 2]
        data_mean_y = np.convolve(data_mean_y, window, mode='same')[:-window_size_plot // 2]
        data_max = np.convolve(data_max, window, mode='same')[:-window_size_plot // 2]
        data_min = np.convolve(data_min, window, mode='same')[:-window_size_plot // 2]
        data_std = np.convolve(data_std, window, mode='same')[:-window_size_plot // 2]

    ax2.plot(data_mean_x, data_mean_y, label=name, lw=2, c=color, zorder=20 + n)

    ax.plot(data_mean_x, data_mean_y, label=name, lw=2, c=color, zorder=20 + n)
    ax.fill_between(data_mean_x, data_mean_y + data_std, data_mean_y - data_std, alpha=0.35, color=color,
                    zorder=10 + n)
    ax.plot(data_mean_x, data_max, ls="--", lw=1, c=color, zorder=30 + n, alpha=0.7)
    ax.plot(data_mean_x, data_min, ls="--", lw=1, c=color, zorder=30 + n, alpha=0.7)

print(f"\nMean STD: {np.mean(mean_std):.3f}")

if data_type == "M01" and scale_m01:
    datas = [all_datas[j] for j in data_indexes_glued if all_datas[j] is not None]
    data_plot_x = np.array(
        [[x[-2].iloc[i, 2] for x in datas] for i in range(np.min([x[-2].shape[0] for x in datas]))])

    ratio = np.mean(datas_y[1] / datas_y[0])
    print(f"\nRatio: {ratio: .5f}")
    plt.plot(data_plot_x, datas_y[0] * ratio, ls="-", lw=1.5, c="tab:green", label=f'Scaled M-01 ({ratio:.2f})',
             zorder=40, alpha=0.7)

for axes in [ax, ax2]:
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
    axes.legend(fontsize=8, bbox_to_anchor=(0.5, -0.3), loc="center", borderaxespad=0, ncol=4)
    axes.set_xlabel('Displacement [mm]')
    axes.set_ylabel('Force [N]')

    axes.set_aspect('auto', adjustable='box')

ax2.set_ylim(ax.get_ylim())
ax2.set_xlim(ax.get_xlim())
fig.subplots_adjust(bottom=0.3, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.3)
fig2.subplots_adjust(bottom=0.3, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.3)
# fig.tight_layout()
# fig2.tight_layout()

if save_figures:
    fig.savefig(f"{out_put_folder}/{data_type}_finalplot_tot.{file_type}", format=file_type, dpi=out_dpi, bbox_inches='tight')
    fig2.savefig(f"{out_put_folder}/{data_type}_finalplot_singleline.{file_type}", format=file_type, dpi=out_dpi,
                 bbox_inches='tight')

if data_type == "S01":
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
                              np.hstack([data_indexes__I_O, data_indexes__III_O, data_indexes__II_O,
                                         data_indexes__I_max_O, data_indexes__III_max_O, data_indexes__II_max_O])]):
        sheet_name = data[0]

        # TODO
        # Změna názvů typu infillu dle stran hexagonů
        if "-II-" in sheet_name:
            sheet_name = sheet_name.replace("-II-", "-III-")
        elif "-III-" in sheet_name:
            sheet_name = sheet_name.replace("-III-", "-II-")

        sheet_name = sheet_name.replace("S01_", "").replace("-10S", "").replace("_O", "")

        # Přepsání názvů sloupců pro třetí DataFrame
        # df3.columns = ['New_M', 'New_N']

        # Ukládání jednotlivých DataFrame na různá místa
        start_row = 0
        col_start = 0
        data[4].to_excel(excel_writer, sheet_name=sheet_name, startrow=start_row, startcol=col_start, index=False)
        col_start += len(data[4].columns)
        data[2].to_excel(excel_writer, sheet_name=sheet_name, startrow=start_row, startcol=col_start, index=False)
        col_start += len(data[2].columns)
        data[1].to_excel(excel_writer, sheet_name=sheet_name, startrow=start_row, startcol=col_start, index=False)
        col_start += len(data[1].columns)
        data[3].to_excel(excel_writer, sheet_name=sheet_name, startrow=start_row, startcol=col_start, index=False)


    # Zavření Excel souboru
    excel_writer.close()
    print(f"Soubor byl úspěšně uložen do: [ {os.path.join(out_put_folder, excel_file)} ]")


########################################################################################################################
if data_type == "H02":
    corner = corner_1

    type_A = []
    type_B = []

    datas_pack = (("Infill type 1", "Infill type 2"),
                  (np.array([i for i in corner if "n" in images_folders[i].lower()]),
                   np.array([i for i in corner if "k" in images_folders[i].lower()])),
                  ("dodgerblue", "red"), (type_A, type_B))

    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    for n, (name, curve_indexes, color, t) in enumerate(zip(*datas_pack)):
        data_plot_x = [all_datas[j][-3].iloc[:, 0].values for j in curve_indexes if all_datas[j] is not None]
        data_plot_y = [all_datas[j][-3].iloc[:, 1].values for j in curve_indexes if all_datas[j] is not None]
        data_plot_dic = [datas_dic[j] for j in curve_indexes if datas_dic[j] is not None]
        data_plot_dic = [data_plot_dic[j][abs(len(data_plot_dic[j]) - len(data_plot_y[j])):] for j in
                         range(len(data_plot_dic))]

        [ax1.plot(np.hstack((0, data_plot_y[i])), np.hstack((0, data_plot_dic[i])), lw=2, c=color, zorder=10 + n,
                  label=name) for i in range(len(data_plot_dic))]

        t.append(np.hstack((0, np.mean(data_plot_x, axis=0))))
        t.append(np.hstack((0, np.mean(data_plot_y, axis=0))))
        t.append(np.hstack((0, np.mean(data_plot_dic, axis=0))))
        t.append(np.hstack((0, np.std(data_plot_dic, axis=0))))
        t.append(np.hstack((0, np.max(data_plot_dic, axis=0))))
        t.append(np.hstack((0, np.min(data_plot_dic, axis=0))))

    for i, (n, _, c, t) in enumerate(zip(*datas_pack)):
        ax2.plot(t[0], t[2], lw=2, c=c, zorder=20, label=n)
        ax2.fill_between(t[0], t[2] + t[3], t[2] - t[3], alpha=0.35, color=c, zorder=20 - i)
        ax2.plot(t[0], t[4], ls="--", lw=1, c=c, zorder=20 - i, alpha=0.7)
        ax2.plot(t[0], t[5], ls="--", lw=1, c=c, zorder=20 - i, alpha=0.7)

    for ax in (ax1, ax2):
        ax.grid(color="lightgray", linewidth=0.5, zorder=0)
        for axis in ['top', 'right']:
            ax.spines[axis].set_linewidth(0.5)
            ax.spines[axis].set_color('lightgray')

        if ax.get_xlim()[1] % ax.get_xticks()[-1] == 0:
            ax.spines['right'].set_visible(False)
        if ax.get_ylim()[1] % plt.gca().get_yticks()[-1] == 0:
            ax.spines['top'].set_visible(False)

        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())

        ax.tick_params(axis='both', which='minor', direction='in', width=0.5, length=2.5, zorder=5, color="black")
        ax.tick_params(axis='both', which='major', direction='in', width=0.8, length=5, zorder=5, color="black")

        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[0], handles[-1]], [labels[0], labels[-1]],
                  fontsize=8, bbox_to_anchor=(0.5, -0.3), loc="center", borderaxespad=0, ncol=4)

        ax.set_ylabel(r'Total relative strain [\%]' if do_tex else 'Total relative strain [%]')
        ax.set_xlabel("Displacement [mm]")
        # ax.set_ylabel("Force [$N$]")

        ax.set_aspect('auto', adjustable='box')

    fig1.subplots_adjust(bottom=0.3, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.3)
    fig2.subplots_adjust(bottom=0.3, top=0.9, left=0.1, right=0.9, wspace=0.3, hspace=0.3)

    if save_figures:
        fig1.savefig(f"{out_put_folder}/hex_corner1.{file_type}", format=file_type, dpi=out_dpi, bbox_inches='tight')
        fig2.savefig(f"{out_put_folder}/hex_corner2.{file_type}", format=file_type, dpi=out_dpi, bbox_inches='tight')

plt.show()

