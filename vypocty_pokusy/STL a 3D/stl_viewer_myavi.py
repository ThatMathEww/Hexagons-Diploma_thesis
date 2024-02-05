import numpy as np
from mayavi import mlab
from tvtk.api import tvtk
import matplotlib.pyplot as plt
from PIL import Image

# Název souboru STL
stl_filename = "11.stl"
photo_filename = "11.png"

if "_ROT" not in photo_filename:
    import os.path as os_path

    photo_name = photo_filename.split(".")[0] + "_ROT." + photo_filename.split(".")[1]

    if not os_path.exists(photo_name):
        im = Image.open(photo_filename).rotate(90)
        im.save(photo_name)
else:
    photo_name = photo_filename

bmp1 = tvtk.PNGReader()
bmp1.file_name = photo_name  # any jpeg file

my_texture = tvtk.Texture(input_connection=bmp1.output_port, interpolate=1)

fig = mlab.figure(size=(500, 500), bgcolor=(0.16, 0.28, 0.46))

# Načtení scény ze souboru STL
mesh = mlab.pipeline.open(stl_filename)

# Vykreslení 3D scény
surf = mlab.pipeline.surface(mesh)

# Zobrazení os
# mlab.orientation_axes()

surf.actor.enable_texture = True
surf.actor.tcoord_generator_mode = 'plane'
surf.actor.actor.texture = my_texture

# Nastavení pohledu (například pohled shora)
mlab.view(azimuth=45, elevation=45, distance='auto')

"""mlab.options.offscreen = True
# Uložení obrázku bez pozadí do PDF
fig = mlab.figure(None, size=(800, 800))  # Vytvoření nové scény s vlastní velikostí
mlab.draw(fig)
mlab.savefig('output.pdf', figure=fig, magnification=2)  # Upravte podle potřeby

# Vypnutí bezhlavého režimu
mlab.options.offscreen = False

# Uzavření okna Mayavi
mlab.close(fig)"""

# Nastavení pozadí
mlab.get_engine().scenes[0].scene.background = (1, 1, 1)
# Získání aktuálního okna a nastavení vlastností
ren_win = mlab.gcf().scene.render_window
ren_win.alpha_bit_planes = 1  # Zapnutí alfa kanálu pro transparentní pozadí


# Nastavení kvality
ren_win.line_smoothing = True
ren_win.multi_samples = 8  # Zvýší kvalitu vykreslení

# Uložení scény v nejvyšší kvalitě
# mlab.savefig('output_render.png', figure=mlab.gcf())  # Upravte podle potřeby

mlab.savefig(f'./3D outputs/output_object_{photo_filename.split(".")[0]}.obj', figure=mlab.gcf())

# Uložení scény v nejvyšší kvalitě
#  mlab.savefig('output.png', magnification=2)  # Upravte podle potřeby

imgmap = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)

# mlab.close(fig)
# do the matplotlib plot
"""fig2 = plt.figure(figsize=(7, 5))
plt.gcf().set_facecolor((0, 0, 0, 0))
plt.gca().set_facecolor((0, 0, 0, 0))
plt.gca().axis('off')
plt.imshow(imgmap, zorder=4)
# plt.plot(np.arange(0, 480), np.arange(480, 0, -1), 'r-')
plt.savefig('example.png', dpi=1000, bbox_inches='tight')"""

# Zobrazení 3D grafu
mlab.show()
