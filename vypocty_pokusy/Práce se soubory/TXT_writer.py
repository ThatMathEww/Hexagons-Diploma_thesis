import io
import numpy as np

# Cesta k souboru
file_path = 'output.txt'

# Otevření souboru v režimu zápisu ('w' znamená zápis)
with io.open(file_path, 'w', encoding='utf-8') as file:
    # Cyklus pro generování a zápis textu
    for i in range(10):
        # Příklad: Generování nějakého textu v každé iteraci
        text = f'This is line {i} of text.'

        # Zápis textu do souboru
        file.write(text + '\n')
