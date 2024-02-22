from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

min_v = 0
max_v = 360
cmap_name = 'hsv'
label_style = 'degrees'  # degrees_neg, degrees_float, radians, radians_neg, radians_frac1, radians_frac2
circle_r = 0.2

positive_cmap = LinearSegmentedColormap.from_list('positive_cmap',
                                                  ['white', 'red', 'orange', 'yellow', 'lime', 'white'])
negative_cmap = LinearSegmentedColormap.from_list('negative_cmap',
                                                  ['white', 'cyan', 'blue', 'purple', 'magenta', 'white'])

# Generate a figure with a polar projection
fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot(221)

scalar_map = plt.cm.ScalarMappable(cmap=str(cmap_name))
scalar_map.set_clim(vmin=min_v, vmax=max_v)

data = np.random.rand(25, 20) * 360
data[0, :] = 315
# Plot the data using colormesh
cmesh = ax.pcolormesh(data, cmap=scalar_map.cmap, vmin=min_v, vmax=max_v)
# Vykreslení fill between mezi kruhy s různými poloměry

# Set the color map
ax.axis('equal')
# ax.axis('tight')


cax = fig.add_axes(222, projection='polar')
# cax.set_title('Rotations')

# cax.set_theta_direction(-1)
# cax.set_theta_zero_location('N')

n = 800  # the number of secants for the mesh
t = np.linspace(0, 2 * np.pi, n)  # theta values
r = np.linspace(1 - circle_r, 1, 5)  # radius values change 0.6 to 0 for full circle
_, tg = np.meshgrid(r, t)  # create a r,theta meshgrid
cax.pcolormesh(t, r, tg.T, cmap=cmap_name)  # plot the colormesh on axis with colormap

# d = np.pi / len(t)
# [ax.fill_between([t, t + d], [r, r], [1, 1], color=scalar_map.to_rgba(r), zorder=3) for t, r in zip(t, r)]

# Draw lines in octants 1 and 2
edge = np.linspace(0, 2 * np.pi, n)
cax.plot(edge, np.ones_like(edge) * (1 + 0.02), color='black', linewidth=1.75)
cax.plot(edge, np.ones_like(edge) * (1 - circle_r - 0.015), color='black', linewidth=1.75)
cax.set_xticks(cax.get_xticks())

# Draw lines in octants
for i in cax.get_xticks():
    r1 = np.linspace(1 + 0.025, 1 + 0.075, 2)
    cax.plot(np.ones_like(r1) * i, r1, color='black')

# '{:.0f}°'.format(i[1] * 180 / np.pi)
# '{:.2f}π'.format(i[1] / np.pi)

if label_style == 'degrees_float':
    cax.set_xticklabels(['{:.2f}°'.format(i * 180 / np.pi) for i in cax.get_xticks()])
elif label_style == 'degrees_int':
    cax.set_xticklabels(['{:.0f}°'.format(i * 180 / np.pi) for i in cax.get_xticks()])
elif label_style == 'degrees_neg':
    cax.set_xticklabels(
        ['{:.0f}°'.format(i * 180 / np.pi if i <= np.pi else i * 180 / np.pi - 360) for i in cax.get_xticks()])
elif label_style == 'radians':
    cax.set_xticklabels(['{:.2f}π'.format(i / np.pi) for i in cax.get_xticks()])
    cax.tick_params(pad=7)
elif label_style == 'radians_neg':
    cax.set_xticklabels(
        ['{:.2f}π'.format(i / np.pi if i <= np.pi else (i - 2 * np.pi) / np.pi) for i in cax.get_xticks()])
    cax.tick_params(pad=7)
elif label_style == 'radians_frac1':
    from fractions import Fraction

    cax.set_xticklabels([str(Fraction(i / np.pi).limit_denominator()) + 'π' for i in cax.get_xticks()])
elif label_style == 'radians_frac2':
    cax.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4'])
else:
    pass

cax.set_yticklabels([])  # turn of radial tick labels (yticks)
# cax.tick_params(pad=10, labelsize=18)  # cosmetic changes to tick labels
cax.spines['polar'].set_visible(False)  # turn off the axis spine.
# ax.set_axis_off()
cax.grid(False)
# ax.set_rlim([-1, 1])


pos_cax = fig.add_axes(223, projection='polar')
neg_cax = fig.add_axes(224, projection='polar')
neg_cax.set_theta_direction(-1)

circle_r *= 0.6
t = np.linspace(0, 2 * np.pi, n)  # theta values
r = np.linspace(1 - circle_r, 1, 5)  # radius values change 0.6 to 0 for full circle
_, tg = np.meshgrid(r, t)  # create a r,theta meshgrid
cax.pcolormesh(t, r, tg.T, cmap=cmap_name)  # plot the colormesh on axis with colormap

for axis, color_map in zip([pos_cax, neg_cax], (positive_cmap, negative_cmap)):  # ('autumn', 'winter')
    axis.pcolormesh(t, r, tg.T, cmap=color_map, zorder=4)  # plot the colormesh on axis with colormap
    axis.plot(edge, np.ones_like(edge) * (1 + 0.02), color='black', linewidth=1.75, zorder=10)
    axis.plot(edge, np.ones_like(edge) * (1 - circle_r - 0.015), color='black', linewidth=1.75, zorder=10)
    axis.set_xticks(axis.get_xticks())
    # Draw lines in octants
    for i in axis.get_xticks():
        r1 = np.linspace(1 + 0.025, 1 + 0.07, 2)
        axis.plot(np.ones_like(r1) * i, r1, color='black', zorder=10)
    axis.set_yticklabels([])  # turn of radial tick labels (yticks)
    axis.spines['polar'].set_visible(False)  # turn off the axis spine.
    axis.grid(False)
neg_cax.set_xticklabels([f'-{str(i.get_text())}' for i in neg_cax.get_xticklabels()])

t = np.linspace(0, 2 * np.pi, n)  # theta values
r = np.linspace(1 - circle_r - 0.02, 1 - circle_r * 2 - 0.04, 5)
_, tg = np.meshgrid(r, t)  # create a r,theta meshgrid
pos_cax.pcolormesh(t, r, tg.T, cmap=negative_cmap.reversed(), zorder=3)  # 'winter_r'
pos_cax.plot(edge, np.ones_like(edge) * (1 - circle_r * 2 - 0.05), color='black', linewidth=1.75,
             zorder=10)
for i in pos_cax.get_xticks():
    r1 = np.linspace(1 - circle_r * 2 - 0.065, 1 - circle_r * 2 - 0.10, 2)
    pos_cax.plot(np.ones_like(r1) * i, r1, color='black', zorder=10)
    pos_cax.text(i, float(r1[-1] - 0.21), '{:.0f}°'.format(i * 180 / np.pi - 360 if i != 0 else 0),
                 ha='center', va='center', fontsize=10, zorder=10)

pos_cax.fill_between(t, 1 + 0.015, 1 - circle_r * 2 - 0.045, color='black', zorder=1)

cax.set_aspect('equal', adjustable='box')
pos_cax.set_aspect('equal', adjustable='box')
neg_cax.set_aspect('equal', adjustable='box')
ax.set_aspect('equal', adjustable='box')

fig.tight_layout()
# fig.subplots_adjust(right=1 - 0.1, left=0.1, top=1 - 0.1, bottom=0.1, wspace=0.3, hspace=0.3)


fig, ax = plt.subplots()

r_out = 0.9
r_in = 0.7

re, im = np.mgrid[-1:1:500j, -1:1:500j]
angle = np.flip(-np.angle(re + 1j * im), axis=0)
cf = ax.pcolormesh(re, im, angle, shading='gouraud', cmap="hsv", vmin=-np.pi, vmax=np.pi, zorder=5)

ax.add_patch(plt.Circle((0, 0), r_in, edgecolor='black', linewidth=3, facecolor='white', alpha=1, zorder=6))
clip_path = plt.Circle((0, 0), r_out, edgecolor='black', linewidth=3, facecolor='none', alpha=1, zorder=6)
ax.add_patch(clip_path)
cf.set_clip_path(clip_path)

theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
for t in theta:
    x = r_out * np.cos(t)
    y = r_out * np.sin(t)
    ax.plot((x, x * 1.05), (y, y * 1.05), color='black', linewidth=3)
    f = t * 180 / np.pi
    ax.text(x * 1.2, y * 1.2, f'{f:.0f}°' if f.is_integer() else f'{f:.2f}°', ha='center', va='center', fontsize=12,
            zorder=4)

ax.relim()
ax.axis('off')
ax.set_aspect('equal', adjustable='box')
fig.tight_layout()

plt.show()
