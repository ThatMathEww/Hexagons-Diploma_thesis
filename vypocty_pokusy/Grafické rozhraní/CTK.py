import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class InteractiveWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Interaktivní okno")

        # Vytvoření hlavního rámce pro rozložení na dva sloupce
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill=ctk.BOTH, expand=True)

        # Levý sloupec
        left_frame = ctk.CTkFrame(main_frame)
        left_frame.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True)

        # Vytvoření grafu
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)

        # Vytvoření tk.Canvas pro zobrazení grafu v tkinter
        canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=True)

        # Vytvoření ovládacího panelu Matplotlibu
        toolbar = NavigationToolbar2Tk(canvas, left_frame)
        toolbar.update()
        toolbar.pack(side=ctk.TOP, fill=ctk.BOTH, expand=False)

        # Pravý sloupec
        right_frame = ctk.CTkFrame(main_frame)
        right_frame.pack(side=ctk.RIGHT, padx=10)

        self.buttons = []
        for i in range(7):
            button = ctk.CTkButton(right_frame, text=f"Tlačítko {i+1}", command=lambda i=i: self.button_clicked(i))
            button.pack(pady=5)
            self.buttons.append(button)

        # Vytvoření sedmého tlačítka pod sloupci
        button7 = ctk.CTkButton(right_frame, text="Tlačítko 7", command=lambda i=6: self.button_clicked(i))
        button7.pack(pady=5)
        self.buttons.append(button7)

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.points = [[] for _ in range(7)]
        self.last_button = None

    def button_clicked(self, button_index):
        # Nastavení posledního tlačítka
        self.last_button = button_index
        print(f"Kliknuto na tlačítko {button_index + 1}")
        print(f"Souřadnice: {self.points[button_index]}")

    def on_click(self, event):
        # Registrování kliknutí v grafu na základě posledního tlačítka
        if self.last_button is not None and event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            self.ax.plt(x, y, 'ro')
            self.fig.canvas.draw()

            # Uložení souřadnic do příslušné paralelní proměnné na základě posledního tlačítka
            self.points[self.last_button].append((x, y))


if __name__ == '__main__':
    root = ctk.CTk()
    app = InteractiveWindow(root)
    root.mainloop()
