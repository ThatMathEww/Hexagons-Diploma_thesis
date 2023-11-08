import pandas as pd

# Načtení dat ze souboru CSV
folder_path = 'VZ4_2.00_4.csv'  # cesta k souboru
# data = pd.read_csv(cesta_k_souboru, skiprows=15, header=None,  # skiprows=15 nebo header=16
# names=['Sloupec1', 'Sloupec2', 'Sloupec3', 'Sloupec4'], sep="\t")  # sep: "," ";"

"""
sep nebo delimiter:     Určuje oddělovač (znak nebo řetězec), který je použit mezi hodnotami v jednom řádku. 
                        Defaultně je to čárka, ale můžete použít i jiný znak, například středník, tabulátor atd.

header:                 Určuje, který řádek souboru má být použit jako hlavička sloupců. 
                        Možnosti zahrnují None (žádný řádek), 0 (první řádek), n (řádek číslo n).

names:                  Pokud nechcete použít první řádek jako názvy sloupců, 
                        můžete specifikovat vlastní názvy sloupců pomocí seznamu názvů.

skiprows:               Určuje, kolik prvních řádků chcete přeskočit. 
                        To může být užitečné, pokud jsou v souboru záhlaví nebo komentáře.

encoding:               Určuje kódování souboru, např. 'utf-8', 'latin1', 'cp1252' atd.

dtype:                  Určuje datové typy sloupců.

na_values:              Specifikuje seznam hodnot, které mají být považovány za chybějící hodnoty (NaN).

index_col:              Určuje sloupec, který má být použit jako index.

parse_dates:            Určuje sloupce, které mají být analyzovány jako datumové hodnoty.

usecols:                Určuje, které sloupce chcete načíst z datového souboru.

comment:                Určuje znak nebo řetězec, který označuje komentáře a má být ignorován.

skip_blank_lines:       Určuje, zda mají být přeskočeny prázdné řádky.

thousands:              Určuje znak pro tisícovou oddělení čísel.
"""

data = pd.read_csv(folder_path, skiprows=15, sep="\t")

# Vytvoření grafu
col1 = data.iloc[:, 0].values  # načtení prvního sloupce
col2 = data.iloc[:, 1].values  # načtení druhého sloupce
col3 = data.iloc[:, 2].values
col4 = data.iloc[:, 3].values
