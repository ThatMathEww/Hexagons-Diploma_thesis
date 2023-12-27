import os

# Adresář s obrázky
photos_main_folder = r"C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos"

images_folders = [name for name in os.listdir(photos_main_folder) if name.startswith("H02")]

for folder in images_folders:
    input_path = os.path.join(photos_main_folder, folder, 'detail_original')

    if not os.path.isdir(input_path):
        print(f"Folder {folder} does not exist.")
        continue

    # Seznam souborů ve formátu 'Frame_XXXX'
    images = [name for name in os.listdir(input_path) if name.startswith('Frame_') and name.endswith(".jpg")]

    for old_name in images:
        file_number = int(old_name.split('_')[1].replace('.jpg', ''))  # celé číslo ze starého názvu

        old_path = os.path.join(input_path, old_name)  # cesta k souboru

        # Získáme datum poslední úpravy
        mtime = os.path.getmtime(old_path)
        ctime = os.path.getctime(old_path)

        if file_number == 1:
            new_number = -1
            print(f"Soubor neodpovídá ostatním: {old_name}")
        else:
            new_number = max(file_number - 1, 0)  # snížit číslo o 1

        new_path = os.path.join(input_path, f'Frame_{new_number:04d}.jpg')  # nová cesta k souboru

        os.rename(old_path, new_path)  # Přejmenování soubor

        os.utime(new_path, times=(ctime, mtime))  # původní časy souboru

        # print(f"Přejmenován soubor: {old_name} -> {new_name}")
    print(f"Folder {folder} was renamed.")
