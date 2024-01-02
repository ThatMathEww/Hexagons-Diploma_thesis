import cv2
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import seaborn as sns
import numpy as np

"""
sns.set_context("notebook")
sns.heatmap(data, cmap='jet', vmin=0, vmax=1)
"""

save_fig = False

# Pixel map:
color_grid = "white"  # 'white' // 'none'
line_width_grid = 2
line_style = '-'

# Načtení hledané oblasti (například šablony)
x1, y1 = 1330, 100
x2, y2 = x1 + 100, y1 + 100

# Načtení původní fotografie
image1 = cv2.imread(r'first_img.JPG')
image2 = cv2.imread(r'last_img.JPG')

"""fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.tight_layout()
plt.show()"""

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

template = gray1[y1:y2, x1:x2]

height, width = template.shape[:2]

# Použití pixel Matching pro nalezení podobných míst
result = cv2.matchTemplate(gray2, template, cv2.TM_CCOEFF_NORMED)
# ### TM_CCOEFF_NORMED / TM_CCORR_NORMED /  TM_SQDIFF_NORMED -> min_loc

# Získání souřadnic nejlepšího výskytu hledané oblasti
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc
bottom_right = (top_left[0] + height, top_left[1] + width)

# Vytvoření kopie původní fotografie s označenou podobnou oblastí
marked_image = image2.copy()
# cv2.rectangle(marked_image, top_left, bottom_right, (0, 0, 255), 10)
# cv2.rectangle(image1, (x1, y1), (x1, y1), (0, 0, 255), 10)

# Zobrazení grafu
########################################################################################################################
########################################################################################################################
plt.figure(figsize=(10, 6), num="First and last image with marked tracking area")
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.gca().add_patch(plt.Rectangle((x1, y1), height, width, linewidth=2, edgecolor='r', facecolor='none'))
ax_ins = plt.gca().inset_axes([0.65, 0.65, 0.33, 0.33], xticks=[], xticklabels=[], yticks=[], yticklabels=[])
ax_ins.imshow(cv2.cvtColor(image1[y1: y2, x1: x2], cv2.COLOR_BGR2RGB), extent=(x1, x2, y1, y2), origin="upper")
[ax_ins.spines[axis].set_linewidth(1.05) for axis in ['top', 'bottom', 'left', 'right']]
plt.gca().indicate_inset_zoom(ax_ins, edgecolor="black", linewidth=1.05, alpha=0.8)
plt.axis('off')
# plt.title('Původní fotografie')
plt.tight_layout()

plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
plt.gca().add_patch(plt.Rectangle(top_left, height, width, linewidth=2, edgecolor='r', facecolor='none'))
ax_ins = plt.gca().inset_axes([0.65, 0.65, 0.33, 0.33], xticks=[], xticklabels=[], yticks=[], yticklabels=[])
ax_ins.imshow(cv2.cvtColor(marked_image[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0]], cv2.COLOR_BGR2RGB),
              extent=(top_left[0], bottom_right[0], top_left[1], bottom_right[1]), origin="upper")
[ax_ins.spines[axis].set_linewidth(1.05) for axis in ['top', 'bottom', 'left', 'right']]
plt.gca().indicate_inset_zoom(ax_ins, edgecolor="black", linewidth=1.05, alpha=0.8)
plt.axis('off')
# plt.title('Označená podobná oblast')
plt.tight_layout()

if save_fig:
    plt.savefig("fig_1.pdf", format="pdf", dpi=700, bbox_inches='tight')

########################################################################################################################
########################################################################################################################

data = cv2.resize(template, (20, 20), interpolation=cv2.INTER_LINEAR)

fig, ax = plt.subplots(num="Resized tracked area")
ax.imshow(data, cmap='gray', extent=[0, data.shape[1], 0, data.shape[0]], interpolation='none')

fig, ax = plt.subplots(num="Pixel values of resized tracked area")

data = cv2.normalize(data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)[5:15, 5:15]

# Vytvoření heatmapy
ax.imshow(data, cmap='gray', extent=[0, data.shape[1], 0, data.shape[0]], interpolation='none')

# Nastavení popisků os
ax.set_xlabel('x coordinates')
ax.set_ylabel('y coordinates')

# Zobrazení hodnot v každém čtverci s kontrastní barvou popisků
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        value = data[i, j]
        ax.text((j + 0.5), data.shape[0] - (i + 0.5), f'{value:.2f}',
                color='white' if 0 <= value < 0.35 else 'black', ha='center', va='center', fontsize=8)

ax.set_xticks(np.arange(data.shape[1]) + 0.5, labels=np.arange(data.shape[1]))
ax.set_yticks(np.arange(data.shape[0]) + 0.5, labels=np.arange(data.shape[0]))

# ax.spines[:].set_visible(False)
ax.spines[:].set_color(color_grid)
ax.spines[:].set_linewidth(line_width_grid)

ax.set_xticks(np.arange(data.shape[1] + 1), minor=True)
ax.set_yticks(np.arange(data.shape[0] + 1), minor=True)
ax.grid(which="minor", color=color_grid, linestyle=line_style, linewidth=line_width_grid)
ax.tick_params(which="minor", bottom=False, left=False)

ax.set_facecolor("none")
fig.set_facecolor("none")

ax.set_yticklabels(ax.get_yticklabels()[::-1])
# ax.invert_yaxis()

fig.tight_layout()
ax.set_aspect('equal', adjustable='box')

if save_fig:
    plt.savefig("fig_2.pdf", format="pdf", dpi=700, bbox_inches='tight')

########################################################################################################################
########################################################################################################################
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
plt.figure(figsize=(8, 6), num="Heatmap of similarity of tracked area - zoomed")

plt.imshow(normalized_result[top_left[1] - 100:top_left[1] + 100, top_left[0] - 100:top_left[0] + 100],
           cmap='jet', vmin=0, vmax=1)
plt.colorbar()
plt.title('Tepelná mapa podobnosti')
plt.axis('off')
plt.tight_layout()

if save_fig:
    plt.savefig("fig_3.pdf", format="pdf", dpi=700, bbox_inches='tight')

# Vytvoření grafu tepelné mapy
########################################################################################################################
########################################################################################################################
fig, ax = plt.subplots(figsize=(8, 6), num="Heatmap of similarity of tracked area - with inset")
# plt.title('Tepelná mapa podobnosti')

im = plt.imshow(normalized_result, cmap='jet', vmin=0, vmax=1)

# inset axes
x1, x2, y1, y2 = top_left[0] - 100, top_left[0] + 100, top_left[1] - 100, top_left[1] + 100  # subregion of the image
ax_ins = ax.inset_axes([0.65, 0.65, 0.33, 0.33], xticks=[], xticklabels=[], yticks=[], yticklabels=[])

ax_ins.imshow(normalized_result[y1: y2, x1: x2], extent=(x1, x2, y1, y2), cmap='jet', vmin=0, vmax=1, alpha=1,
              origin="upper")
# change all spines
[ax_ins.spines[axis].set_linewidth(1.05) for axis in ['top', 'bottom', 'left', 'right']]

ax.indicate_inset_zoom(ax_ins, edgecolor="black", linewidth=1.05, alpha=0.8)

"""plt.gca().add_patch(plt.Rectangle((top_left[0] - 100, top_left[1] - 100), 200, 200, edgecolor='blacck', 
facecolor='none', linewidth=2, zorder=5))"""
cax = fig.add_axes((0.85, 0.051, 0.033, 0.89))  # [left, bottom, width, height]
cbar = plt.colorbar(im, cax=cax)  # Přidání colorbaru
# plt.colorbar(aspect=30, shrink=0.75)

ax.axis('off')
# plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.8, top=0.95, bottom=0.05, wspace=0.01, hspace=0.01)

if save_fig:
    plt.savefig("fig_4.pdf", format="pdf", dpi=700, bbox_inches='tight')

# Vytvoření grafu tepelné mapy
########################################################################################################################
########################################################################################################################
fig = plt.figure(figsize=(8, 6), num="3D graph of similarity of tracked area")

fig_ax = plt.gca().axis('off')
# plt.suptitle('Tepelná mapa podobnosti')

ax = fig.add_subplot(111, projection='3d')
# Nastavení polohy osy
ax.set_position([0.05, 0.02, 0.85, 0.92])  # [left, bottom, width, height]

z = normalized_result[top_left[1] - 100:top_left[1] + 100, top_left[0] - 100:top_left[0] + 100]
x = np.arange(0, z.shape[1])
y = np.arange(0, z.shape[0])
x, y = np.meshgrid(x, y)

surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='jet', vmin=0, vmax=1, antialiased=True, linewidth=0)
# alpha=0.8
ax.set_xticks(ax.get_xticks())
ax.set_yticks(ax.get_yticks())

off_set = 1.5

ax.plot_surface(x, y, np.full_like(z, off_set), facecolors=plt.cm.jet(z), rstride=1, cstride=1, antialiased=True,
                alpha=0.15)

# Přidání vrstevnic
contour = ax.contour(x, y, z, levels=7, zdir='z', offset=off_set, linestyles="solid", colors='black', alpha=0.85,
                     antialiased=True)  # colors='black'   //   cmap='jet'
# ax.contour3D(x, y, z, 50, cmap='jet')  # colors='black'   //   cmap='jet'

# fig.autofmt_xdate()
# ax.set_xticklabels(np.int32(ax.get_xticks() + (top_left[1] - 100)), rotation=-30)
# ax.set_yticklabels(np.int32(ax.get_yticks() + (top_left[0] - 100)))

# Vytvoření colorbaru
cax = fig.add_axes((0.8, 0.1, 0.033, 0.8))  # [left, bottom, width, height]
cbar = plt.colorbar(surf, cax=cax, aspect=30, shrink=0.75)  # Přidání colorbaru

# Nastavení os a zobrazení grafu
ax.set_xlabel('x position [pixels]', labelpad=10)
ax.set_ylabel('y position [pixels]', labelpad=10)
ax.zaxis.set_rotate_label(False)  # disable automatic rotation
ax.set_zlabel('Correlation coefficient', labelpad=10, rotation=90)

# ax.set_xticklabels(np.int32(ax.get_xticks() + (top_left[0] - 100)), rotation=-15)
# ax.set_yticklabels(np.int32(ax.get_yticks() + (top_left[1] - 100)))

# ax.set_yticklabels(ax.get_yticklabels()[::-1])
ax.invert_yaxis()

ax.auto_scale_xyz(x.flatten(), y.flatten(), z.flatten())
# Vykreslení 3D grafu
ax.view_init(elev=30, azim=-120, roll=0)
# ax.dist = 1  # Upravte vzdálenost osy podle potřeby
ax.set_zlim(0, 1)
# ax.set_aspect('auto', adjustable='box')
ax.set_box_aspect([1.1, 1.1, 0.755], zoom=0.87)

if save_fig:
    plt.savefig("fig_5.pdf", format="pdf", dpi=700, bbox_inches='tight')

# Vytvoření grafu tepelné mapy
########################################################################################################################
########################################################################################################################
fig, ax = plt.subplots(figsize=(8, 6), num="Heatmap of similarity of tracked area - with inset 3D graph")

fig.set_facecolor('none')
ax.set_facecolor('none')

plt.imshow(normalized_result, cmap='jet', vmin=0, vmax=1)

# inset axes
x1, x2, y1, y2 = top_left[0] - 100, top_left[0] + 100, top_left[1] - 100, top_left[1] + 100  # subregion of the image
ax_ins = ax.inset_axes([0.65, 0.65, 0.33, 0.33], xlim=(x1, x2), ylim=(y1, y2),
                       xticks=[], xticklabels=[], yticks=[], yticklabels=[])
# change all spines
[ax_ins.spines[axis].set_linewidth(1.05) for axis in ['top', 'bottom', 'left', 'right']]

ax_ins_3d = ax_ins.inset_axes([0, 0, 1, 1], projection='3d', xticklabels=[], yticklabels=[],
                              zticks=np.arange(0, 1.2, 0.2), zticklabels=[], zlim=(0, 1))
ax_ins_3d.plot_surface(x, y, z, rstride=1, cstride=1, cmap='jet', vmin=0, vmax=1, antialiased=True)
ax_ins_3d.invert_yaxis()

ax_ins_3d.auto_scale_xyz(x.flatten(), y.flatten(), z.flatten())
ax_ins_3d.view_init(elev=30, azim=-120, roll=0)
# ax.dist = 1  # Upravte vzdálenost osy podle potřeby

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

if save_fig:
    plt.savefig("fig_6.pdf", format="pdf", dpi=700, bbox_inches='tight')

plt.show()
