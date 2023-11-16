import pandas as pd
import numpy as np
import os

# Zadání cesty ke složce
folder_path_load = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\tension\Tension_measurements'
folder_path_save = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\data\data_csv'

files = [f for f in os.listdir(folder_path_load) if
         os.path.isdir(os.path.join(folder_path_load, f)) and not f.lower().endswith('.test')]

i = 2
m = 1  # 1 // 11
a = 0  # 0 // 30
b = 10  # 10 // 7

files = sorted(files, key=lambda x: int(x.split(" ")[2]))[a + (i * b):(a + b) + (i * b)]
# [0+(i * 10):10+(i * 10)] // [30+(i * 7):37+(i * 7)]

# n = 0
# c = 0

for file in files:
    files_txt = [f for f in os.listdir(os.path.join(folder_path_load, file)) if
                 os.path.isfile(os.path.join(folder_path_load, file, f)) and f.lower().endswith('.txt')]

    for file_txt in files_txt:
        df = pd.read_csv(os.path.join(folder_path_load, file, file_txt), skiprows=8, sep="\t",
                         names=['Distance', 'Force', 'Time'])
        # df = df.reindex(columns=['Distance', 'Time', 'Force'])

        """if c % 10:
            n += 1"""

        df = df.replace(',', '.', regex=True)
        names = df.axes[1]
        values = np.array(df.values, dtype=float)
        values[:, 1] *= 1000
        df = pd.DataFrame(dtype=float)
        df[names] = values

        csv_name = f'T01_{m:02d}-{"I" * (i + 1)}_1s.csv'
        cvs_path = os.path.join(folder_path_save, csv_name)
        df.to_csv(cvs_path, index=False)

        # c += 1
        m += 1

        if m == 11:
            m = 1

    print(file, csv_name)

print('\nHotovo')
