"""import cv2
import numpy as np
import matplotlib.pyplot as plt"""
import muDIC as dic


def convert_to_tiff(input_folder, output_folder):
    from PIL import Image
    import os
    # Vytvoření výstupní složky, pokud ještě neexistuje
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Seznam vstupních souborů ve složce
    file_list = os.listdir(input_folder)

    for filename in file_list:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.tif")

        # Načtení obrázku
        img = Image.open(input_path)

        # Uložení jako TIFF soubor
        img.save(output_path, format='TIFF')

        # Zavření obrázku
        img.close()


# Uveďte cestu ke složce s vstupními fotografiemi a cestu k výstupní složce pro TIFF soubory
input_folder = "photos/1/"
output_folder = "photos/1/"

# convert_to_tiff(input_folder, output_folder)

path = r"photos/1/"
image_stack = dic.image_stack_from_folder(path, file_type=".tif")
mesher = dic.Mesher()
mesh = mesher.mesh(image_stack)

inputs = dic.DICInput(mesh, image_stack)
dic_job = dic.DICAnalysis(inputs)

results = dic_job.run()
fields = dic.Fields(results)
true_strain = fields.true_strain()

viz = dic.Visualizer(fields, images=image_stack)
viz.show(field="True strain", component=(1, 1), frame=2)
