import pandas as pd
import numpy as np
import os

# Zadání cesty ke složce
folder_path_load = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\tension\Tension_measurements'
folder_path_save = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\data\data_csv'

files = [f for f in os.listdir(folder_path_load) if
         os.path.isdir(os.path.join(folder_path_load, f)) and not f.lower().endswith('.test')]
files = sorted(files, key=lambda x: int(x.split(" ")[2]))

i = 2  # 0 // 1 // 2
typ = 2  # 1 // 2

s = [j + i for j in list(range(30 if typ == 2 else 0, 51 if typ == 2 else 28, 3))]
# files = sorted(files, key=lambda x: int(x.split(" ")[2]))[a + (i * b):(a + b) + (i * b)]
d = [files[j] for j in s]
for n, file in enumerate([files[j] for j in s]):
    files_txt = [f for f in os.listdir(os.path.join(folder_path_load, file)) if
                 os.path.isfile(os.path.join(folder_path_load, file, f)) and f.lower().endswith('.txt')]

    n = n + 10 if typ == 2 else n

    for file_txt in files_txt:
        df = pd.read_csv(os.path.join(folder_path_load, file, file_txt), skiprows=8, sep="\t",
                         names=['Distance', 'Force', 'Time'])
        # df = df.reindex(columns=['Distance', 'Time', 'Force'])

        df = df.replace(',', '.', regex=True)
        names = df.axes[1]
        values = np.array(df.values, dtype=float)
        values[:, 1] *= 1000
        df = pd.DataFrame(dtype=float)
        df[names] = values

        csv_name = f'T01_{n + 1:02d}-{"I" * (i + 1)}_1s.csv'
        cvs_path = os.path.join(folder_path_save, csv_name)
        df.to_csv(cvs_path, index=False)

        print(file, csv_name)

print('\nHotovo')
