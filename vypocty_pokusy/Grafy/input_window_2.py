import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import numpy as np

fig, ax = plt.subplots()
fig.subplots_adjust(right=0.7)
# Vytvoření polygonu
polygon = plt.Polygon(np.array(((1, 1), (2, 1), (6, 5), (5, 6))), closed=True, fill=False, edgecolor='red')

# Přidání polygonu do grafu
ax.add_patch(polygon)

# Získání aktuálních bodů polygonu
current_paths = polygon.get_path()
current_verts = current_paths.vertices[:-1]

# Vytvoření pole pro TextBoxy
text_boxes = []
texts = []

top = fig.subplotpars.top
# Vytvoření TextBoxu pro každý vrchol polygonu
for i, (x, y) in enumerate(current_verts):
    ax_box = fig.add_axes([0.82, top - (i * 0.06) - 0.05, 0.15, 0.05])
    text_boxes.append(TextBox(ax_box, f'Vrchol {i + 1}: ', textalignment="center",
                              initial=f'{int(x) if x.is_integer() else x} ; {int(y) if y.is_integer() else y}'))
    texts.append(ax.text(x + 0.1, y - 0.1, f"{i + 1}.", fontweight='bold', color='black', ha='left', va='top',
                         bbox=dict(facecolor='white', edgecolor='none', boxstyle='round', pad=0.3, alpha=0.7)))


# Funkce pro aktualizaci souřadnic po upravě TextBoxu
def update_coords(text):
    index = [i for i, t in enumerate(text_boxes) if text == t.text]
    if len(index) == 1:
        if index[0] == 0:
            index = [0, -1]
    else:
        print("\n\033[33mWARRNING:\tBody mají stejné souřadnice.\033[0m")
        return
    try:
        new_coords = text.replace(",", ".").strip('()').strip('[]').split(';')
        if len(new_coords) == 2:
            new_coords = (float(new_coords[0]), float(new_coords[1]))
            points = polygon.get_path().vertices
            points[index] = [new_coords[0], new_coords[1]]
            polygon.set_xy(points)
            texts[index[0]].set_position((new_coords[0] + 0.1, new_coords[1] - 0.1))
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
    except (ValueError, NameError):
        print("\n\033[33mWARRNING:\tChyba v získání hodnot vrcholů.\033[0m")
        return


# Přiřazení funkce pro aktualizaci TextBoxů
for text_box in text_boxes:
    text_box.on_submit(update_coords)

# Zde můžete provést další operace s grafem

# Zobrazte legendu (jen pro vizualizaci)

ax.relim()
ax.autoscale_view()
ax.margins(x=0.1, y=0.1)
plt.show()

c = '#335DA6', '#9E242C', '#32A57E', '#AF7A38'
