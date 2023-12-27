import os
# import numpy as np


def rename_folders():
    # Získání seznamu souborů ve složce "slozka1"
    files = [os.path.splitext(file)[0] for file in os.listdir(folder_measurement) if
             os.path.isfile(os.path.join(folder_measurement, file)) and file.lower().endswith(".txt")]

    files = files[:]

    n = 1
    print("\n====================================================\n\nZahájení programu.")
    # cyklus na ukládání souborů mezi složkami
    for f in files:
        path = os.path.join(folder_measurement, f + ".txt")
        if os.path.exists(path):
            # Přejmenujeme složku
            os.rename(path, os.path.join(os.path.dirname(path), f'S01_{os.path.basename(path)}'))

        print(n, "/", len(files), ":", "přejmenováno", "\t[", f, "]")
        n += 1
    print("\n====================================================\n\nVeškeré složky vytvořeny.")


def create_folders():
    # Získání seznamu souborů ve složce "slozka1"
    files = [os.path.splitext(file)[0] for file in os.listdir(folder_measurement) if
             os.path.isfile(os.path.join(folder_measurement, file)) and file.lower().endswith(".csv") and
             file.startswith("T01")]

    files = files[:]

    print("\n====================================================\n\nZahájení programu.")
    # cyklus na ukládání souborů mezi složkami
    for n, f in enumerate(files):
        f = f'{f}'
        # Vytvoření cesty k cílovému souboru
        new_path_1 = os.path.join(image_folder, f, "original")
        new_path_2 = os.path.join(image_folder, f, "modified")

        # Zkontrolování, zda cílová složka již existuje
        if not os.path.exists(new_path_1):
            os.makedirs(new_path_1)
        if not os.path.exists(new_path_2):
            os.makedirs(new_path_2)

        print(n + 1, "/", len(files), ":", "vytvořeno", "\t[", f, "]")
    print("\n====================================================\n\nVeškeré složky vytvořeny.")


if __name__ == '__main__':
    # Nastavení cesty k složce s obrázky
    image_folder = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos'

    # Nastavení cest ke složkám
    folder_measurement = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\data\data_csv'

    # Spuštění programu
    create_folders()
    # rename_folders()
