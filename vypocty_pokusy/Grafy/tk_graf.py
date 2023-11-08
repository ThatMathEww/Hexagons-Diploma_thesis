import tkinter as tk
from tkinter import ttk
from tkinter import Entry
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import style as mtp_style

import ttkbootstrap as tb
from ttkbootstrap.toast import ToastNotification

# Funkce pro aktualizaci grafu
def update_graph(event=None):
    if event == None:
        # root = tb.Window(themename="superhero")

        """root.title("TTK toast")
        root.iconbitmap('')
        root.iconbitmap(default='')"""
        # root.geometry('0x0')

        def clicker():
            toast.show_toast()

        def close_window():
            root.destroy()

        toast = ToastNotification(
            title='WOHOOO',
            message='Tohle je zprava',
            # duration=3000,
            alert=True,
            # position=(1600, -120, 'sw'),
            bootstyle='DARK',
            # icon='֍'
        )

        """btn = tb.Button(root, text='Click', command=clicker)
        btn.pack(pady=40)"""
        clicker()
        #root.after(5000, close_window)
        #root.mainloop()
        return
    # Získání odkazu na textové pole, které aktivovalo funkci
    entry = event.widget
    widget_name = event.widget.winfo_name()
    print(f"Jméno widgetu: {widget_name}")


    try:
        w = int(float(widget_name.split("!entry")[1]))
    except ValueError:
        w = 0

    if "!entry3" == widget_name:
        print(5444444444444)

    # Zde můžete provádět další operace s tímto textovým polem
    # Například, získání hodnoty z textového pole: value = entry.get()

    # Výpis pro kontrolu
    print(f"Bylo aktivováno textové pole {entry}: {entry.get()}")
    submit_x(0)
    submit_y(0)
    submit_r(0)
    pass


def submit_r(_):
    global polygon_cor, center_cor, polygon
    try:
        rot = round(float(eval(entries[2].get().replace(",", "."), {'np': np})), 5)
        rot = int(rot) if rot.is_integer() else rot
        entries[2].delete(0, "end")  # Vymaže stávající text z textového pole
        if not 0 <= rot < 359:
            entries[2].insert(0, f"{rot % 360}")
        else:
            entries[2].insert(0, f"{rot}")
        rot = np.deg2rad(rot)
        scale = round(max(float(entries[3].get().replace(",", ".")), 0), 5)
        scale = int(scale) if scale.is_integer() else scale
        entries[3].delete(0, "end")
        entries[3].insert(0, f"{scale}")
        # Vytvoření transformační matice pro rotaci
        rotation_matrix = np.array([[np.cos(rot), -np.sin(rot)],
                                    [np.sin(rot), np.cos(rot)]]) * scale
        polygon.set_xy(np.dot(polygon_cor - center_cor, rotation_matrix.T) + center_cor)
        s.set_offsets(center_cor)
        fig.canvas.draw()
    except (ValueError, SyntaxError):
        pass


def submit_x(_):
    global polygon_cor, center_cor
    try:
        points = polygon.get_path().vertices
        c = float(entries[0].get().replace(",", ".")) - center_cor[0]
        polygon_cor[:, 0] = polygon_cor[:, 0] + c
        points[:, 0] = points[:, 0] + c
        center_cor[0] = center_cor[0] + c
        polygon.set_xy(points)
        s.set_offsets(center_cor)
        """ax.relim()
        ax.autoscale_view()"""
        fig.canvas.draw()
    except (ValueError, SyntaxError):
        pass


def submit_y(_):
    global polygon_cor, center_cor
    try:
        points = polygon.get_path().vertices
        c = float(entries[1].get().replace(",", ".")) - center_cor[1]
        points[:, 1] = points[:, 1] + c
        polygon_cor[:, 1] = polygon_cor[:, 1] + c
        center_cor[1] = center_cor[1] + c
        polygon.set_xy(points)
        s.set_offsets(center_cor)
        """ax.relim()
        ax.autoscale_view()"""
        fig.canvas.draw()
    except (ValueError, SyntaxError):
        pass


# Vytvoření hlavního okna Tkinter
root = tb.Window()
root.title("Tkinter Okno s Grafem")

input_frame = ttk.Frame(root, style='My.TFrame')
input_frame.grid(row=0, column=0, padx=10, pady=10)

root.configure(bg="#F5F5F5")

entries = []
num_entries = 4
for i, f in zip(range(num_entries), (submit_x, submit_y, submit_r, submit_r)):
    label = ttk.Label(input_frame, text=f"Hodnota {i + 1}:")
    label.grid(row=i, column=0, padx=5, pady=5)
    entry = Entry(input_frame)
    entry.grid(row=i, column=1, padx=5, pady=5)
    entry.bind("<Return>", f)
    entry.bind("<FocusOut>", f)
    entries.append(entry)

# Vytvoření tlačítka pro aktualizaci grafu
update_button = ttk.Button(input_frame, text="Aktualizovat Graf", command=update_graph)
update_button.grid(row=4, column=0, columnspan=2, padx=5, pady=10)

# Vytvoření grafu vpravo
graph_frame = ttk.Frame(root)
graph_frame.grid(row=0, column=1, padx=10, pady=10)

# Vytvoření figury Matplotlib pro graf
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)

polygon_cor = np.array([[-2, -2], [2, -2], [0, 4]], dtype=np.float64)
polygon = plt.Polygon(polygon_cor, closed=True, fill=False, ec="green", lw=2.0)
center_cor = np.mean(polygon_cor, axis=0, dtype=np.float64)
s = ax.scatter(*center_cor)
ax.add_patch(polygon)
ax.imshow(plt.imread(r"C:\Users\matej\Downloads\IMG_0487.JPG"))

ax.set_aspect('equal', adjustable='box')
fig.tight_layout(pad=0)
# Vytvoření navigačního panelu Matplotlib
canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

# Vytvoření navigačního panelu
toolbar = NavigationToolbar2Tk(canvas, graph_frame)
toolbar.update()
canvas_widget.pack()
toolbar.configure(padx=5, pady=5)
# Předefinování třídy pro tlačítko Save
mtp_style.use('default')  # Použijeme defaultní styl Matplotlib
save_button = toolbar.children
for child in toolbar.winfo_children():
    if isinstance(child, tk.Button) or isinstance(child, tk.Checkbutton):
        child.configure(bg='lightgray', width=25, height=25, activebackground='gray',
                        relief="sunken")
        # overrelief: Literal["raised", "sunken", "flat", "ridge", "solid", "groove"]
        if isinstance(child, tk.Button):
            child.configure(width=29, height=29)

entries[0].insert(0, f"{int(center_cor[0])}" if center_cor[0].is_integer() else f"{center_cor[0]:.2f}")
entries[1].insert(0, f"{int(center_cor[1])}" if center_cor[1].is_integer() else f"{center_cor[1]:.2f}")
entries[2].insert(0, "0")
entries[3].insert(0, "1")


# Spuštění hlavní smyčky Tkinter
root.mainloop()
