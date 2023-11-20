import os


def main():
    # Získání seznamu jmen názvů složek dle jmen měření
    folder_names = [name for name in [f for f in os.listdir(image_folder)] if
                    os.path.isdir(os.path.join(image_folder, name))]

    for folder in folder_names:
        folder_path = os.path.join(image_folder, folder, "modified")
        files_names = [name for name in [f for f in os.listdir(folder_path)] if name.endswith(".JPG")]
        for file in files_names:
            file_path = os.path.join(folder_path, file)

            # Získání původních dat a času
            original_creation_time = os.path.getctime(file_path)
            original_modification_time = os.path.getmtime(file_path)

            # Přejmenování souboru
            new_file_path = os.path.join(os.path.dirname(file_path), "mod-" + file.replace("-mod", ""))
            os.rename(file_path, new_file_path)

            # Nastavení nových dat a času
            os.utime(new_file_path, (original_creation_time, original_modification_time))


if __name__ == '__main__':
    # Nastavení cesty k složce s obrázky
    image_folder = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos'

    main()
