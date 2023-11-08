import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch

# Data pro graf
x = [1, 2, 3, 4, 5]
y1 = [10, 20, 15, 25, 30]
y2 = [5, 15, 10, 20, 25]

# Vytvoření grafu
fig, ax = plt.subplots()
plt.title('Points paths - Image {}: {}', wrap=True, pad=12)
line1, = ax.plot(x, y1, zorder=3)
line2, = ax.plot(x, y2, zorder=3)
text = ax.text(2, 15, "Text", fontweight='bold', zorder=3)
area1 = Rectangle((3, 10), 3, 4, facecolor='firebrick', alpha=0.5, zorder=2)
ax.add_patch(area1)

show_menu = True
show1 = False
show2 = False

width = 0.2 if show_menu else 0

fig.set_facecolor('none')
ax.set_facecolor('none')

fig.subplots_adjust(right=1 - width - 0.02, left=0.1, top=0.9, bottom=0.1, wspace=0, hspace=0)

# print(fig.subplotpars.left, fig.subplotpars.bottom, fig.subplotpars.right, fig.subplotpars.top)

a = {'Oblast 1': area1, 'Text': text}

# Vytvoření checkboxu
rax = plt.axes([fig.subplotpars.right + 0.01, fig.subplotpars.top - 0.15, width, 0.15])  # left, bottom, width, height
check = CheckButtons(ax=rax, labels=a.keys(), actives=(show1, show2),
                     frame_props={'linewidth': [2, ], 'sizes': [100, ]},
                     label_props={'fontsize': [20, ], 'fontweight': ['bold', ]})

rax.set_visible(show_menu)


# Funkce pro aktualizaci viditelnosti prvků
def update(name):
    ln = a[name]
    ln.set_visible(not ln.get_visible())
    ln.figure.canvas.draw_idle()


# Nastavení velikosti a barvy textu v tlačítcích
text_objs = check.labels
for text_obj in text_objs:
    text_obj.set_fontsize(8)  # Nastavení velikosti textu

rax.add_patch(FancyBboxPatch((0, 0), 1, 1, edgecolor='gray', facecolor='#CCCCCC',
                             boxstyle="round,pad=0.1",
                             alpha=0.8, zorder=0))

check.on_clicked(update)
[update(key) for key in a.keys()]
rax.axis('off')
# ax.axis('off')
ax.axis('equal')
rax.autoscale(True)
ax.autoscale(True)

ax.spines[["bottom", "left", "right", "top"]].set_visible(False)
ax.add_patch(FancyBboxPatch((0, 0), 1, 1, edgecolor='darkgray', facecolor='none',
                            boxstyle="round,pad=-0.0015,rounding_size=0.03", transform=ax.transAxes,
                            alpha=1, zorder=-1))

ax.grid()
plt.show()
