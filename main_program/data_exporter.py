import pandas as pd
import numpy as np
import zipfile
import h5py
import time
import os

saved_data_name = "data_pokus.zip"

main_image_folder = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos'
folder_measurements = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\data'

"""folder = "H01_03_12s"
t1 = [os.path.getmtime(os.path.join(main_image_folder, folder, "original", i)) for i in
      os.listdir(os.path.join(main_image_folder, folder, "original"))][1:]
t2 = [0] + [t1[i + 1] - t1[i] for i in range(len(t1) - 1)]"""

########################################################################################################################

images_folders = [name for name in [os.path.splitext(file)[0] for file in os.listdir(main_image_folder)]
                  if name.startswith("H01") or name.startswith("_")]
images_folders = [images_folders[i] for i in (37, 38)]  # (10, 11, 12, 13, 19, 33, 37, 38)

########################################################################################################################
########################################################################################################################

tot_folders = len(images_folders)
for exp, current_image_folder in enumerate(images_folders):
    print(f"\nNačítání uložených dat: ' \033[94;1m{current_image_folder}\033[0m ' -  [ {exp + 1} / {tot_folders}]")

    # Název Excel souboru
    excel_file = f'hexagon_values_{exp + 1}.xlsx'

    # correlation_points, tracked_points, tracked_rotations, distances, forces, photo_indexes = [None] * 6
    dataset_1, dataset_2, dataset_3, distances, forces, photo_indexes, time_stamps = [None] * 7

    current_folder_path = os.path.join(main_image_folder, current_image_folder)
    zip_files = [f for f in os.listdir(current_folder_path)
                 if os.path.isfile(os.path.join(current_folder_path, f)) and f.lower().endswith(".zip")]

    zip_file_name = os.path.join(current_folder_path, saved_data_name)
    if saved_data_name not in zip_files:
        print(f'\n\033[31;1;21mERROR\033[0m\n\tVe složce [{current_image_folder}] se nenachází daný soubor ZIP')
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
                print(f'\n\033[33;1;21mWARRNING\033[0m\n\tV uložených datech se nenachází soubor: "{csv_file_name}"'
                      f'\n\t➤ Pokus o načtení souboru ze složky')

            zero_stage = 10  # TODO &&&&&&&&&&&
            window_size_start = 5  # TODO &&&&&&&&&&&

            df = pd.read_csv(path_to_csv)  # DATAFRAME

            # Odečtení průměru od všech následujících hodnot v 2. a 3. sloupci
            df.iloc[zero_stage:, 1] -= df.iloc[:zero_stage, 1].mean()
            df.iloc[zero_stage:, 2] -= df.iloc[:zero_stage, 2].mean()

            # Načtení dat
            distances = df.iloc[:, 0].values  # První sloupec jako osa x - posun
            forces = - (df.iloc[:, 1].values + df.iloc[:, 2].values)  # - celková síla
            photo_indexes = df[df['Photos'].notna()].index

            # Klouzavý průměr s oknem šířky 5
            window_start = max(3, window_size_start)
            while True:
                try:
                    cumulative_sum = np.cumsum(forces)
                    cumulative_sum[window_start:] = cumulative_sum[window_start:] - cumulative_sum[:-window_start]
                    # Najděte kladná čísla
                    positive_numbers = forces[max(window_start - 2, 0):window_start + 1][
                        forces[max(window_start - 2, 0):window_start + 1] > 0]
                    min_positive = np.min(positive_numbers)
                    # Porovnáme průměry 5 po sobě jdoucích čísel s hodnotami na daných pozicích
                    condition = (cumulative_sum / window_start) < min_positive
                    break
                except ValueError:
                    if window_start > window_size_start + 50:
                        condition = [True]
                        print("\nDosažení limitu pro hledání počátku měření")
                        break
                    window_start += 1

            # Najdeme pozice, kde podmínka platí
            start_position = np.where(condition)[0][-1]
            distances = distances - distances[start_position]  # Stanovení 0 pozice zatěžovnání

            # Nalezení nejbližší vyšší hodnoty
            """beginning = np.where(photo_indexes >= start_position)[0][
                np.argmin(photo_indexes[photo_indexes >= start_position])]"""
            # Nalezení indexu nejbližší vyšší hodnoty
            beginning = np.argmax(photo_indexes >= start_position)

            #  ######################################################################################################  #
            #  ############################################     TIME     ############################################  #
            #  ######################################################################################################  #

            try:
                if "image_folder/" in zip_list:
                    # Převod času z GMT na lokální čas změny fotek v ZIPu a uložení do seznamu
                    time_stamps = [int(time.mktime(zipf.getinfo(file).date_time + (0, 0, 0))) for file
                                   in [name for name in zip_list if name.startswith("image_folder/")][1:]]

                    time_values = np.int64([0.0 if not np.isnan(t) else np.nan for t in df['Photos'].values])

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

                    t = np.int64([0] + [photo_indexes[i + 1] - photo_indexes[i] for i in range(len(photo_indexes) - 1)])
                    t[1:-1] = np.median(t[1:-1])
                    time_stamps[-1] = time_stamps[1] * (t[-1] / t[1])
                    time_stamps = [np.sum(time_stamps[:i + 1]) for i in range(len(time_stamps))]

                    nan_indices = np.isnan(time_values)  # Najděte indexy NaN hodnot
                    time_values[~nan_indices] = time_stamps
                    time_stamps = time_values.copy()  # Vytvořte kopii vektoru pro interpolaci
                    # Nahraďte NaN hodnoty interpolovanými hodnotami
                    time_stamps[nan_indices] = np.interp(np.flatnonzero(nan_indices),
                                                         np.flatnonzero(~nan_indices), time_values[~nan_indices])
                    time_stamps = time_stamps[start_position:] - time_stamps[start_position]
                    # time_stamps = [t - time_stamps[0] for t in time_stamps]
                else:
                    print("\n\033[33;1;21mWARRNING\033[0m\n\t - V souboru ZIP se nenachází fotografie")
            except Exception as e:
                print("\n\033[33;1;21mWARRNING\033[0m\n\t - "
                      f"Chyba načtení časového nastavení měření ze složky: [{current_image_folder}]\n\tPOPIS: {e}")

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
                        for d in [key for key in file.keys() if key.startswith('dictionary_')]:
                            dataset_1 += [{key: value for key, value in file[d].attrs.items()}]

                    if 'additional_variables' in group_names:
                        dataset_3 = {key: value for key, value in file['additional_variables'].attrs.items()}

                    dataset_2 = dict(data_correlation=None, data_point_detect=None)
                    for group_name in dataset_2.keys():
                        if group_name in group_names:
                            data_group = file[group_name]
                            dataset_2[group_name] = [[dataset[:] for dataset in subgroup.values()]
                                                     for subgroup in data_group.values()]
                file.close()
            h5_file.close()
        zipf.close()
    except (KeyError, Exception) as e:
        print(f'\n\033[31;1;21mERROR\033[0m\n\tSelhalo načtení uložených dat\n\tPOPIS: {e}')
        continue

    datasets = dict(Correlation=None, Tracked_points=None, Forces=None, Others=None)

    try:
        scale = dataset_1[-1]

        if dataset_2['data_correlation'] is not None:
            datasets['Correlation'] = dataset_2['data_correlation']  # correlation_points

        if dataset_2['data_point_detect'] is not None:
            datasets['Tracked_points'] = dataset_2['data_point_detect']  # [tracked_points, tracked_rotations]

        if isinstance(dataset_3, dict) and len(dataset_3) > 0:
            datasets['Others'] = dataset_3

        datasets['Forces'] = (distances, forces, photo_indexes)

    except (ValueError, Exception) as e:
        print(f'\n\033[31;1;21mERROR\033[0m\n\tSelhalo přiřazení hodnot uložených dat\n\tPOPIS: {e}')
        continue

    data_frames = []
    data_frames_names = []

    photos = np.arange(beginning, len(photo_indexes), 1)  # int(np.nanmax(df['Photos'].values)) + 1
    time_values = time_stamps[photo_indexes - start_position][beginning:]

    if datasets['Correlation'] is not None:
        data = np.float64([np.mean(c[0], axis=0) for c in datasets['Correlation'][beginning:]])
        data = (data - data[0]) * scale
        # Vytvoření datových rámce pro listy
        data_frames.append(pd.DataFrame({'Photo': photos,
                                         'Time [s]': time_values,
                                         'X [mm]': data[:, 0],
                                         'Y [mm]': data[:, 1]}))
        data_frames_names.append('Movement of loading bar')

    if datasets['Tracked_points'] is not None:
        [tracked_points, tracked_rotations] = datasets['Tracked_points']
        len_points = len(tracked_points[0])
        len_photos = len(tracked_points)

        data = [(np.float64([tracked_points[i][j] for i in range(len_photos)]),
                 np.float64([tracked_rotations[i][j] for i in range(len_photos)]))
                for j in range(len_points)][beginning:]

        # Vytvoření datových rámce pro listy
        df_tr = pd.DataFrame({'Photo': photos,
                              'Time [s]': time_values})

        # Přidání tří sloupců ve smyčce
        for i in range(len_points):  # Přidáme tři skupiny sloupců
            df_tr[[f'Point_{i + 1} - {v}' for j, v in zip(range(3), ('X [mm]', 'Y [mm]', 'Rotation [rad]'))]] = [
                data[i][0][:, 0], data[i][0][:, 1], data[i][1]]
        data_frames.append(df_tr)
        data_frames_names.append(f'Tracked points - {len_points}. points')

    if datasets['Forces'] is not None:
        # Vytvoření datových rámce pro listy
        data_frames.append(pd.DataFrame({'Photo': photos,
                                         'Time [s]': time_values,
                                         'Distance [mm]': distances[photo_indexes[beginning:]],
                                         'Force [N]': forces[photo_indexes[beginning:]]}))
        data_frames_names.append('Forces on photo')

    if datasets['Forces'] is not None:
        # Vytvoření datových rámce pro listy
        data_frames.append(pd.DataFrame({'Photo': df['Photos'].values[start_position:],
                                         'Time [s]': time_stamps,
                                         'Distance [mm]': distances[start_position:],
                                         'Force [N]': forces[start_position:]}))
        data_frames_names.append('All forces')

    if datasets['Others'] is not None:
        pass

    # Vytvoření ExcelWriter
    try:
        excel_writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
    except PermissionError:
        print(f'\n\033[31;1;21mERROR\033[0m\n\tSoubor [{excel_file}] nelze upravovat, pravděpodobně je otevřen.')
        continue

    # Uložení textu v listu
    text1 = "some text here"
    text2 = "other text here"
    description = "Toto je popis souboru CSV s více listy."

    # Vytvoření listu pro popis
    df_description = pd.DataFrame({'Popis': [description]})

    # Zápis popisu na zvláštní list
    df_description.to_excel(excel_writer, sheet_name='Popis', index=False, startrow=5)

    worksheet = excel_writer.sheets['Popis']
    worksheet.write(0, 0, text1)
    worksheet.write(1, 0, text2)

    # Zápis dat do listů
    for i, data_frame in enumerate(data_frames):
        data_frame.to_excel(excel_writer, sheet_name=f'List_{i}', index=False)
        worksheet.write(i + 7, 0, f'List_{i}: {data_frames_names[i]}')

    # Zavření Excel souboru
    excel_writer.close()
    print(f"\tData úspěšně uložena.")

print("\nHotovo.")
