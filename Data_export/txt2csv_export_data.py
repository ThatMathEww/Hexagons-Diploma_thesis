import os
import re
import pandas as pd

# import numpy as np

# Zadání cesty ke složce
folder_path_load = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\data\data_txt'
folder_path_save = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\data\data_csv'

# Získání seznamu souborů v dané složce
files = [file for file in os.listdir(folder_path_load) if os.path.isfile(os.path.join(folder_path_load, file)) and
         file.lower().endswith(".txt") and file.startswith("B01_")][-1:]

operation = 0

# Procházení souborů a provádění operace pro soubory typu TXT
for file in files:
    if file.endswith('.txt'):
        # Získání úplné cesty k souboru
        file_name = os.path.join(folder_path_load, file)

        # jméno souboru
        base_name = os.path.splitext(file)[0]

        # Otevření souboru
        with open(file_name, 'r') as file:
            # Přečtení obsahu souboru
            content = file.readlines()

        # Vytvoření prázdných seznamů pro jednotlivé sloupce dat
        distances = []
        f0_values = []
        f1_values = []
        f2_values = []
        photo_numbers = []

        # Zpracování jednotlivých řádků od druhého do předposledního
        lines = content[:]

        # Hledání shodných vzorů ve formátu pomocí regulárních výrazů a zpracování jednotlivých řádků
        pattern_1 = r"Distance: ([+-]?\d+(?:\.\d+)?) mm, F0 = ([+-]?\d+(?:\.\d+)?) N, F1 = ([+-]?\d+(?:\.\d+)?) N, " \
                    r"F2 = ([+-]?\d+(?:\.\d+)?)N"

        pattern_2 = r"Distance: ([+-]?\d+(?:\.\d+)?) mm, F0 = ([+-]?\d+(?:\.\d+)?) N, F1 = ([+-]?\d+(?:\.\d+)?) N, " \
                    r"F2 = ([+-]?\d+(?:\.\d+)?) N"

        last_photo = 0
        match = None

        for line in lines:
            match_1 = re.search(pattern_1, line)
            match_2 = re.search(pattern_2, line)
            if match_1:
                match = match_1
            elif match_2:
                match = match_2
            else:
                continue

            distance = float(match.group(1))
            f0 = float(match.group(2))
            f1 = float(match.group(3))
            # f2 = float(match.group(4))

            distances.append(distance)
            f0_values.append(f0)
            f1_values.append(f1)
            # f2_values.append(f2)

            photo_number_match = re.search(r" N -> photo (\d+) taken at this point", line)
            if photo_number_match:
                photo_number = int(photo_number_match.group(1))
                last_photo = photo_number
            else:
                photo_number = None
            photo_numbers.append(photo_number)

        photo_numbers[0], photo_numbers[-1] = 0, last_photo + 1

        # print(len(distances))
        # print(last_photo)

        """# Převedení seznamů na numpy pole (vektor)
        distances = np.array(distances)
        f0_values = np.array(f0_values)
        f1_values = np.array(f1_values)
        # f2_values = np.array(f2_values)
        photo_numbers = np.array(photo_numbers)"""

        """# Výpis načtených dat
        print("Distances:", distances)
        print("F0 values:", f0_values)
        print("F1 values:", f1_values)
        # print("F2 values:", f2_values)
        print("Photo numbers:", photo_numbers)"""

        # Vytvoření DataFrame z listů
        data = {'Distance': distances, 'F0': f0_values, 'F1': f1_values, 'Photos': photo_numbers}

        df = pd.DataFrame(data)

        """if base_name == 'B01_06':
            import numpy as np

            mask = (df['Distance'] <= df['Distance'][0] - 8) & (df['Distance'] >= df['Distance'][0] - 16.5)
            for f in ('F0', 'F1'):
                start = df.loc[mask, f].iloc[0]
                end = df[f][-20:-1].mean()

                new_val = np.linspace(start, end, sum(mask)) * -((np.linspace(1, 0, sum(mask)) ** 3))
                new_val = new_val * np.cos(0.5 * np.pi * np.linspace(0, 1, sum(mask)) * 0.5)

                new_val = (new_val - min(new_val)) / (max(new_val) - min(new_val)) * (start - end) + end

                # Aktualizujeme hodnoty v druhém sloupci na místě masky
                df.loc[mask, f] = new_val  + np.random.uniform(-0.015, 0.015, sum(mask))"""

        # Uložení do CSV souboru s odpovídajícím jménem
        csv_file = base_name + '.csv'
        cvs_path = os.path.join(folder_path_save, csv_file)
        df.to_csv(cvs_path, index=False)

        operation += 1

        print(f"Operace číslo:\t{operation}\t\tSoubor:\t{base_name}")
    else:
        print("Neplatný formát souboru.")
