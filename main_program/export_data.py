def create_csv(os, pd, file_name, folder_path_load='data/data_txt', folder_path_save='data/dada_cvs'):
    import re

    # Získání seznamu souborů v dané složce
    file_name_txt = file_name + ".txt"

    if file_name_txt in os.listdir(folder_path_load):
        # Získání úplné cesty k souboru
        file_name_txt = os.path.join(folder_path_load, file_name_txt)

        # Otevření souboru
        with open(file_name_txt, 'r') as file:
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

        # Vytvoření DataFrame z listů
        data = {'Distance': distances, 'F0': f0_values, 'F1': f1_values, 'Photos': photo_numbers}
        df = pd.DataFrame(data)

        # Uložení do CSV souboru s odpovídajícím jménem
        file_name_csv = file_name + '.csv'
        cvs_path = os.path.join(folder_path_save, file_name_csv)
        df.to_csv(cvs_path, index=False)

        print(f"\n\tSoubor uložen:\t{file_name_csv}\n\t\t Cesta: ' {cvs_path} '")

    else:
        raise Exception(f"\nSoubor nebyl ve složce nelezen \n\tSoubor: ' {file_name_txt} ',  "
                        f"Složka: ' {folder_path_load} '")
