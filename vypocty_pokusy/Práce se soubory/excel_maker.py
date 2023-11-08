import pandas as pd

# Název Excel souboru
excel_file = 'vase_multi_listy.xlsx'
list_name = 'List'

"""try:
    # Načtení Excel souboru do datového rámce Pandas
    df = pd.read_excel(excel_file, sheet_name=list_name)

    # Vypsání obsahu datového rámce
    print(df)
except ValueError as e:
    print(e)


# Načtení všech listů Excel souboru
dfs = pd.read_excel(excel_file, sheet_name=None)

# Vypsání obsahu všech datových rámce
for sheet_name, df in dfs.items():
    print(f"\nList: {sheet_name}")
    print(df)

exit()"""

# Vytvoření dat pro listy
data_list1 = {'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]}
data_list2 = {'Y': [10, 20, 30, 40], 'X': ['apple', 'banana', 'cherry', 'date']}

# Create multiple lists
technologies = ['Spark', 'Pandas', 'Java', 'Python', 'PHP']
fee = [25000, 20000, 15000, 15000, 18000]
duration = ['5o Days', '35 Days', None, '30 Days', '30 Days']
discount = [2000, 1000, 800, 500, 800]
columns = ['Courses', 'Fee', 'Duration', 'Discount']

# Create DataFrame from multiple lists
df_list0 = pd.DataFrame(list(zip(technologies, fee, duration, discount)), columns=columns)

# Vytvoření datových rámce pro listy
df_list1 = pd.DataFrame(data_list1)
df_list2 = pd.DataFrame(data_list2)

# Vytvoření ExcelWriter
excel_writer = pd.ExcelWriter('vase_multi_listy.xlsx', engine='xlsxwriter')

# Uložení popisu nahoře nad listy
description = "Toto je popis souboru CSV s více listy."

# Vytvoření listu pro popis
df_description = pd.DataFrame({'Popis': [description]})

# Zápis dat do listů
df_list1.to_excel(excel_writer, sheet_name='List1', index=False)
df_list2.to_excel(excel_writer, sheet_name='List1', startcol=4, startrow=2, index=False)
df_list2.to_excel(excel_writer, sheet_name='List2', index=False, header=False)
df_list0.to_excel(excel_writer, sheet_name='List3', index=False, columns=['Fee','Duration'])

# Zápis popisu na zvláštní list
df_description.to_excel(excel_writer, sheet_name='Popis', index=False, startrow=5)

text1 = "some text here"
text2 = "other text here"

worksheet = excel_writer.sheets['Popis']
worksheet.write(0, 0, text1)
worksheet.write(1, 0, text2)

# Zavření Excel souboru
excel_writer.close()

# Převod Excel souboru na CSV
xlsx = pd.ExcelFile('vase_multi_listy.xlsx')
csv_file = 'vase_multi_listy.csv'

# Načtení všech listů z Excelu a uložení jako CSV
for sheet_name in xlsx.sheet_names:
    df = pd.read_excel(xlsx, sheet_name)
    df.to_csv(f'{csv_file}.{sheet_name}.csv', index=False)
