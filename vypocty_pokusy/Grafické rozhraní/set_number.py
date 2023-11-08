import tkinter as tk
from tkinter import ttk
from tkinter.simpledialog import askfloat, askinteger
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

st = 0
######################################################################
if st == 0:
    # Vytvoření funkce pro vykreslení grafu
    def vykresli_graf():
        # Získání čísla z vstupního okna
        cislo = askinteger("Vstup", "Zadejte číslo:")

        # Pokud uživatel zadá číslo, vykreslíme ho v grafu
        if cislo is not None:
            plt.figure()
            plt.plot(cislo, 0, 'ro')  # Zobrazíme číslo v grafu
            plt.xlabel('Osa x')
            plt.ylabel('Osa y')
            plt.title(f'Zadané číslo: {cislo}')
            plt.grid(True)
            plt.show()
            exit(0)


    # Vytvoření grafického okna pro vstup čísla
    root = tk.Tk()
    root.withdraw()  # Skryje hlavní okno Tkinter
    cislo = askinteger("Vstup", "Zadejte číslo:")

    # Pokud uživatel zadá číslo, vykreslíme ho v grafu
    if cislo is not None:
        vykresli_graf()

    root.mainloop()

######################################################################

elif st == 1:
    # Funkce pro vykreslení grafu
    def vykresli_graf():
        plt.draw()
        # Získání čísel z vstupních polí
        cislo1 = int(vstup1.get())
        cislo2 = int(vstup2.get())

        # Vytvoření grafu
        fig, ax = plt.subplots()
        ax.plot([cislo1, cislo2], [0, 0], 'ro-')
        ax.set_xlabel('Osa x')
        ax.set_ylabel('Osa y')
        ax.set_title(f'Zadaná čísla: {cislo1}, {cislo2}')
        ax.grid(True)

        # Zobrazení grafu v okně
        canvas = FigureCanvasTkAgg(fig, master=okno)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)


    # Vytvoření hlavního okna
    okno = tk.Tk()
    okno.title("Graf s vstupními čísly")

    # Vytvoření vstupního pole pro první číslo
    frame1 = ttk.Frame(okno)
    frame1.pack(side=tk.LEFT, padx=10, pady=10)
    label1 = ttk.Label(frame1, text="Číslo 1:")
    label1.pack()
    vstup1 = ttk.Entry(frame1)
    vstup1.pack()

    # Vytvoření vstupního pole pro druhé číslo
    frame2 = ttk.Frame(okno)
    frame2.pack(side=tk.LEFT, padx=10, pady=10)
    label2 = ttk.Label(frame2, text="Číslo 2:")
    label2.pack()
    vstup2 = ttk.Entry(frame2)
    vstup2.pack()

    # Tlačítko pro vykreslení grafu
    tlacitko = ttk.Button(okno, text="Vykreslit graf", command=vykresli_graf)
    tlacitko.pack()

    okno.mainloop()
    exit(0)
