import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import zipfile
import h5py
import time
import cv2
import os

saved_data_name = "data_export_new.zip"
out_put_folder = ""

main_image_folder = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos'
folder_measurements = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\data'

data_indexes_I = np.arange(0, 4 * 7, 5) + 3
data_indexes_II = np.arange(0, 4 * 7, 5) + 2
data_indexes_III = np.arange(0, 4 * 7, 5) + 1
data_indexes_max = np.arange(0, 4 * 7, 5)
data_indexes_can_norm = np.arange(0, 4 * 7, 5) + 4
data_indexes_can_snapped = np.arange(30, 36)

########################################################################################################################

images_folders = [name for name in [os.path.splitext(file)[0] for file in os.listdir(main_image_folder)]
                  if name.startswith("H01")]

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

            # Načtení dat
            distances = df.iloc[:, 0].values  # - posun
            forces = -((df.iloc[:, 1].values - df.iloc[:zr, 1].mean()) +
                       (df.iloc[:, 2].values - df.iloc[:zr, 2].mean()))  # - celková síla
            photo_indexes = df[df['Photos'].notna()].index

            # Najdi indexy, kde je okno rovno `pocet_podminka`
            start_index = np.max(np.where(forces[:max(min(int(np.where(np.convolve(np.where(np.abs(np.array(
                [np.mean(forces[s - z2:s + z2]) for s in range(z2, d_len // 3)])[zr:] - np.median(
                forces[1:zr * 2 + 1])) >= np.mean(
                np.array([np.std(forces[s - z2:s + z2]) for s in range(z2, d_len // 300)])[zr:]
                * 1.5), 1, 0), np.ones(int(np.ceil(d_len * 0.1))), mode='valid') == np.ceil(d_len * 0.1))[0][0]) - zr,
                                                          d_len), 0) + 1] <= 0)[0])

            # Najdi indexy, kde je rozdíl menší než -5
            snap_index = np.where(np.diff(forces) <= -5)[0]

            # Nalezení nejbližší vyšší hodnoty
            """beginning = np.where(photo_indexes >= start_index)[0][
                np.argmin(photo_indexes[photo_indexes >= start_index])]"""
            # Nalezení indexu nejbližší vyšší hodnoty
            beginning = np.argmax(photo_indexes >= start_index)

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

                correlation_points = [
                    cv2.transform(np.float64(point[0]).reshape(1, 2, 2), angle_correction_matrix).reshape(2, 2) for
                    point in correlation_points]

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
                    distances = (distances * (np.linalg.norm(correlation_points[0][0, 1] - correlation_points[-1][0, 1])
                                              / np.linalg.norm(distances[0] - distances[-1])))
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
            data = np.float64([np.mean(c, axis=0) for c in correlation_points])
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
            data_frames.append(pd.DataFrame({'Photo': df['Photos'].values[start_index:],
                                             'Time [s]': time_stamps,
                                             'Distance [mm]': distances[start_index:],
                                             'Force [N]': forces[start_index:]}))

        if datasets['Others']:
            pass

        if datasets['Tracked_points'] and False:

            len_points = len(tracked_points[0])
            len_photos = len(tracked_points)

            data = [(np.float64([tracked_points[i][j] for i in range(len_photos)]),
                     np.float64([tracked_rotations[i][j] for i in range(len_photos)])) for j in range(len_points)]

            data = [(np.float64([d[0][i] - d[0][0] for i in range(len_photos)]) * scale, d[1]) for d in data]

            # Vytvoření datových rámce pro listy
            df_tr = pd.DataFrame()

            # Přidání tří sloupců ve smyčce
            for i in range(len_points):  # Přidáme tři skupiny sloupců
                df_tr[[f'Point_{i + 1} - {v}' for v in ('X [mm]', 'Y [mm]')]] = np.vstack(
                    (data[i][0][:, 0], data[i][0][:, 1])).T[beginning:]
            data_frames.append(df_tr)
        else:
            data_frames.append([])

        all_datas.append(data_frames)

    except (ValueError, Exception) as e:
        print(f'\033[31;1;21mERROR\033[0m\n\tSoubor [{current_image_folder}] se nepovedlo uložit.\n\tPOPIS: {e}')
        continue

print("\n\033[33;1mHotovo.\033[0m")

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

# indexes = [data_indexes_I, data_indexes_II, data_indexes_III, data_indexes_max]
indexes = [data_indexes_can_snapped]

# Vytvoření subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8)) if len(indexes) == 4 else plt.subplots(2, 1, figsize=(12, 4)) \
    if len(indexes) == 2 else plt.subplots(1, 1, figsize=(6, 4))
try:
    axs = axs.flatten()
except AttributeError:
    axs = [axs]

for i in range(len(indexes)):
    try:
        [axs[i].plot(all_datas[j][-2].iloc[:, 2].values, all_datas[j][-2].iloc[:, 3].values,
                     c='gray', lw=1, alpha=0.5, zorder=4) for j in np.hstack(indexes[:i] + indexes[i + 1:])]
    except ValueError:
        pass

    [axs[i].plot(all_datas[j][-2].iloc[:, 2].values, all_datas[j][-2].iloc[:, 3].values,
                 lw=2, label=all_datas[j][0], zorder=5) for j in indexes[i]]

    axs[i].grid()
    axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    axs[i].set_xlabel('Distance [mm]')
    axs[i].set_ylabel('Force [N]')

    axs[i].set_aspect('auto', adjustable='box')

plt.tight_layout()

fig, ax = plt.subplots(figsize=(6, 6))

for name, curve_index, color in zip(("I", "II", "III"),
                                    # ("I", "II", "III") // ("MAX", "NORM", "SNAPPED")
                                    (data_indexes_I, data_indexes_II, data_indexes_III),
                                    # (data_indexes_max, data_indexes_can_norm, data_indexes_can_snapped) //
                                    # (data_indexes_I, data_indexes_II, data_indexes_III)
                                    ("dodgerblue", "red", "limegreen")):
    datas = [all_datas[j] for j in curve_index]
    data_plot_x = np.array([[x[-2].iloc[i, 2] for x in datas] for i in range(np.min([x[-2].shape[0] for x in datas]))])
    data_plot_y = np.array([[y[-2].iloc[i, 3] for y in datas] for i in range(np.min([y[-2].shape[0] for y in datas]))])

    # Definice velikosti okna pro klouzavý průměr
    window_size = 5
    # Vytvoření průměrového filtru
    window = np.ones(window_size) / window_size

    """
    # Převod dat na pandas DataFrame
    df = pd.DataFrame({'y': y})
    
    # Vytvoření klouzavého průměru
    window_size = 10
    y_smooth = df['y'].rolling(window=window_size).mean()
    """

    data_mean_x = np.mean(data_plot_x, axis=1)
    data_mean_y = np.mean(data_plot_y, axis=1)
    data_max = np.max(data_plot_y, axis=1)
    data_min = np.min(data_plot_y, axis=1)
    data_std = np.std(data_plot_y, axis=1)

    # Aplikace klouzavého průměru
    data_mean_x = data_mean_x[:-2]
    data_mean_y = np.convolve(data_mean_y, window, mode='same')[:-2]
    data_max = np.convolve(data_max, window, mode='same')[:-2]
    data_min = np.convolve(data_min, window, mode='same')[:-2]
    data_std = np.convolve(data_std, window, mode='same')[:-2]

    ax.plot(data_mean_x, data_mean_y, label=name, lw=2, c=color, zorder=6)
    ax.fill_between(data_mean_x, data_mean_y + data_std, data_mean_y - data_std, alpha=0.35, color=color, zorder=4)
    ax.plot(data_mean_x, data_max, ls="--", lw=1, c=color, zorder=5)
    ax.plot(data_mean_x, data_min, ls="--", lw=1, c=color, zorder=5)

ax.grid()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel('Distance [mm]')
ax.set_ylabel('Force [N]')

ax.set_aspect('auto', adjustable='box')
plt.tight_layout()

plt.show()
