import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# Načtení původní fotografie
image1 = cv2.imread(r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\Hexagons-Diploma_thesis'
                    r'\vypocty_pokusy\photos\IMG_0385.JPG')
image2 = cv2.imread(r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\Hexagons-Diploma_thesis'
                    r'\vypocty_pokusy\photos\IMG_0417.JPG')

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Načtení hledané oblasti (například šablony)
"""x1, y1 = 2600, 400
x2, y2 = 3450, 650"""
x1, y1 = 2750, 450
x2, y2 = x1 + 100, y1 + 100

"""x1, y1 = 2720, 1100
x2, y2 = 2890, 1170"""

template = gray1[y1:y2, x1:x2]

# Použití pixel Matching pro nalezení podobných míst
result = cv2.matchTemplate(gray2, template, cv2.TM_CCOEFF_NORMED)
# ### TM_CCOEFF_NORMED / TM_CCORR_NORMED /  TM_SQDIFF_NORMED -> min_loc

# Získání souřadnic nejlepšího výskytu hledané oblasti
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc
bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

# Vytvoření kopie původní fotografie s označenou podobnou oblastí
marked_image = image2.copy()
cv2.rectangle(marked_image, top_left, bottom_right, (0, 0, 255), 10)
cv2.rectangle(image1, (x1, y1), (x2, y2), (0, 0, 255), 10)

# Zobrazení grafu
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.title('Původní fotografie')
plt.tight_layout()
plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
plt.title('Označená podobná oblast')
plt.tight_layout()

# Normalizace výsledků na rozsah 0-1
normalized_result = cv2.normalize(result, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

"""
import numpy as np

# Vytvoření grafu tepelné mapy
heatmap = cv2.applyColorMap(np.uint8(255 * normalized_result), cv2.COLORMAP_JET)

# Zobrazení grafu tepelné mapy
fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
ax.set_title('Tepelná mapa podobnosti')
ax.axis('off')
plt.tight_layout()"""

# Vytvoření grafu tepelné mapy
plt.figure(figsize=(8, 6))
plt.imshow(normalized_result[top_left[1] - 100:top_left[1] + 100, top_left[0] - 100:top_left[0] + 100],
           cmap='jet', vmin=0, vmax=1)
plt.colorbar()
plt.title('Tepelná mapa podobnosti')
plt.axis('off')
plt.tight_layout()

# Vytvoření grafu tepelné mapy
fig, ax = plt.subplots(figsize=(8, 6))

plt.imshow(normalized_result, cmap='jet', vmin=0, vmax=1)

# inset axes
x1, x2, y1, y2 = top_left[0] - 100, top_left[0] + 100, top_left[1] - 100, top_left[1] + 100  # subregion of the image
ax_ins = ax.inset_axes([0.5, 0.5, 0.6, 0.47], xlim=(x1, x2), ylim=(y1, y2),
                       xticks=[], xticklabels=[], yticks=[], yticklabels=[])
ax_ins.imshow(normalized_result, cmap='jet', vmin=0, vmax=1, alpha=1, origin="upper")
# change all spines
[ax_ins.spines[axis].set_linewidth(1.05) for axis in ['top', 'bottom', 'left', 'right']]

ax.indicate_inset_zoom(ax_ins, edgecolor="black", linewidth=1.05, alpha=0.8)

"""plt.gca().add_patch(plt.Rectangle((top_left[0] - 100, top_left[1] - 100), 200, 200, edgecolor='blacck', 
facecolor='none', linewidth=2, zorder=5))"""
plt.colorbar(shrink=0.675, aspect=20)
plt.title('Tepelná mapa podobnosti')
# plt.axis('off')
plt.tight_layout()

# Vytvoření grafu tepelné mapy
fig = plt.figure(figsize=(8, 6))

fig_ax = plt.gca().axis('off')
plt.suptitle('Tepelná mapa podobnosti')

ax = fig.add_subplot(111, projection='3d')
# Nastavení polohy osy
ax.set_position([0.05, 0.05, 0.85, 0.85])  # [left, bottom, width, height]

z = normalized_result[top_left[1] - 100:top_left[1] + 100, top_left[0] - 100:top_left[0] + 100]
x = np.arange(0, z.shape[1])
y = np.arange(0, z.shape[0])
x, y = np.meshgrid(x, y)

surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='jet', vmin=0, vmax=1)  # cmap určuje barevnou mapu
ax.set_xticks(ax.get_xticks())
ax.set_yticks(ax.get_yticks())

# fig.autofmt_xdate()
# ax.set_xticklabels(np.int32(ax.get_xticks() + (top_left[1] - 100)), rotation=-30)
# ax.set_yticklabels(np.int32(ax.get_yticks() + (top_left[0] - 100)))

# Vytvoření colorbaru
cax = fig.add_axes((0.8, 0.05, 0.03, 0.8))  # [left, bottom, width, height]
cbar = plt.colorbar(surf, cax=cax, aspect=10)  # Přidání colorbaru

# Nastavení os a zobrazení grafu
ax.set_xlabel('x position [pixels]', labelpad=10)
ax.set_ylabel('y position [pixels]', labelpad=10)
ax.set_zlabel('Correlation coefficient', labelpad=10)

ax.invert_yaxis()

ax.auto_scale_xyz(x.flatten(), y.flatten(), z.flatten())
ax.dist = 1  # Upravte vzdálenost osy podle potřeby
# Vykreslení 3D grafu
ax.view_init(elev=30, azim=-120, roll=0)
ax.set_zlim(0, 1)
# plt.tight_layout()


# Vytvoření grafu tepelné mapy
fig, ax = plt.subplots(figsize=(8, 6))

fig.set_facecolor('none')
ax.set_facecolor('none')

plt.imshow(normalized_result, cmap='jet', vmin=0, vmax=1)

# inset axes
x1, x2, y1, y2 = top_left[0] - 100, top_left[0] + 100, top_left[1] - 100, top_left[1] + 100  # subregion of the image
ax_ins = ax.inset_axes([0.5, 0.5, 0.6, 0.47], xlim=(x1, x2), ylim=(y1, y2),
                       xticks=[], xticklabels=[], yticks=[], yticklabels=[])
# change all spines
[ax_ins.spines[axis].set_linewidth(1.05) for axis in ['top', 'bottom', 'left', 'right']]

ax_ins_3d = ax_ins.inset_axes([0, 0, 1, 1], projection='3d', xticklabels=[], yticklabels=[],
                              zticks=np.arange(0, 1.2, 0.2), zticklabels=[], zlim=(0, 1))
ax_ins_3d.plot_surface(x, y, z, rstride=1, cstride=1, cmap='jet', vmin=0, vmax=1)
ax_ins_3d.invert_yaxis()
ax_ins_3d.view_init(elev=30, azim=-120, roll=0)

ax_ins_3d.xaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))  # Barva pozadí osy x
ax_ins_3d.yaxis.set_pane_color((0.95, 0.95, 0.95, 0.8))  # Barva pozadí osy y
ax_ins_3d.zaxis.set_pane_color((0.85, 0.85, 0.85, 0.8))  # Barva pozadí osy z

ax_ins.set_facecolor((0.5, 0.5, 0.5, 0.8))
ax_ins_3d.set_facecolor('none')
ax_ins.set_aspect('equal', adjustable='box')
ax.set_aspect('equal', adjustable='box')

ax.indicate_inset_zoom(ax_ins, edgecolor="black", linewidth=1.05, alpha=0.8)

"""plt.imshow(normalized_result, cmap='jet', vmin=0, vmax=1)
plt.gca().add_patch(plt.Rectangle((top_left[0] - 100, top_left[1] - 100), 200, 200, edgecolor='red', facecolor='none',
                                  linewidth=2, zorder=5))"""
plt.colorbar(shrink=0.675, aspect=20)
plt.title('Tepelná mapa podobnosti')
# plt.axis('off')
# plt.tight_layout()

plt.show()
