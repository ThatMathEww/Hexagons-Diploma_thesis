import pandas as pd
import numpy as np
import h5py
import zipfile
import os

main_image_folder = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos'
folder_measurements = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\data'
saved_data_name = "data_pokus.zip"
images_folders = [name for name in [os.path.splitext(file)[0] for file in os.listdir(main_image_folder)]
                  if name.startswith("H01") or name.startswith("_")]
images_folders = [images_folders[i] for i in (37, 38)]  # (10, 11, 12, 13, 19, 33, 37, 38)

for current_image_folder in images_folders:
    # correlation_points, tracked_points, tracked_rotations, distances, forces, photo_indexes = [None] * 6
    dataset_1, dataset_2, dataset_3, distances, forces, photo_indexes = [None] * 6

    print("\nNačítání uložených dat.")

    current_folder_path = os.path.join(main_image_folder, current_image_folder)
    zip_files = [f for f in os.listdir(current_folder_path)
                 if os.path.isfile(os.path.join(current_folder_path, f)) and f.lower().endswith(".zip")]

    zip_file_name = os.path.join(current_folder_path, saved_data_name)
    if zip_file_name not in zip_files:
        pass

    try:
        # Načtení dat z zip archivu
        with zipfile.ZipFile(zip_file_name, 'r') as zipf:

            # Zjištění, zda je zip soubor prázdný
            if not zipf.namelist():
                raise Exception(f"\033[31;1;21mError:\033[0m Zip file [{zip_file_name}] is empty.")

            #  ######################################################################################################  #
            #  ############################################     CSV      ############################################  #
            #  ######################################################################################################  #
            # Načítání souboru CSV
            csv_file_name = f"{current_image_folder}.csv"
            if csv_file_name in zipf.namelist():
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
        pass

    datasets = dict(Correlation=None, Tracked_points=None, Forces=None, Others=None)

    try:
        scale = dataset_1[-1]

        if dataset_2['data_correlation'] is not None:
            datasets['Correlation'] = dataset_2['data_correlation'] # correlation_points

        if dataset_2['data_point_detect'] is not None:
            datasets['Tracked_points'] = dataset_2['data_point_detect']  # [tracked_points, tracked_rotations]

        if isinstance(dataset_3, dict) and len(dataset_3) > 0:
            datasets['Others'] = dataset_3

        datasets['Forces'] = (distances, forces, photo_indexes)

    except (ValueError, Exception) as e:
        print(f'\n\033[31;1;21mERROR\033[0m\n\tSelhalo přiřazení hodnot uložených dat\n\tPOPIS: {e}')
        pass

    # Název Excel souboru
    excel_file = 'vase_multi_listy.xlsx'

    data_frames = []

    # Vytvoření dat pro listy
    data_list1 = {'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]}
    data_list2 = {'Y': [10, 20, 30, 40], 'X': ['apple', 'banana', 'cherry', 'date']}

    if datasets['Correlation'] is not None:
        data_1 = np.array([np.mean(c[0], axis=0) for c in datasets['Correlation'][beginning:]])
        # Vytvoření datových rámce pro listy
        data_frames.append(pd.DataFrame({'Photo': None,
                                         'Time': None,
                                         'X': data_1[:, 0],
                                         'Y': data_1[:, 1]}))

    if datasets['Tracked_points'] is not None:
        data_1 = np.array(datasets['Tracked_points'][0][beginning:])
        data_2 = np.array(datasets['Tracked_points'][1][beginning:])
        # Vytvoření datových rámce pro listy
        data_frames.append(pd.DataFrame({'Photo': None,
                                         'Time': None,
                                         'X': data_1[:, 0],
                                         'Y': data_1[:, 1],
                                         'Rotation': data_2}))

    if datasets['Forces'] is not None:
        # Vytvoření datových rámce pro listy
        data_frames.append(pd.DataFrame({'Photo': None,
                                         'Time': None,
                                         'Distance': distances[photo_indexes[beginning:]],
                                         'Force': forces[photo_indexes[beginning:]]}))

    if datasets['Forces'] is not None:
        # Vytvoření datových rámce pro listy
        data_frames.append(pd.DataFrame({'Photo': df['Photos'].values,
                                         'Time': None,
                                         'Distance': distances[start_position:],
                                         'Force': forces[start_position:]}))

    if datasets['Others'] is not None:
        pass

    # Vytvoření ExcelWriter
    excel_writer = pd.ExcelWriter('vase_multi_listy.xlsx', engine='xlsxwriter')

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

    # Zavření Excel souboru
    excel_writer.close()
