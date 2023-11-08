import h5py
import blosc
from PIL import Image
import matplotlib.pyplot as plt
import timeit


def h5_save():
    # Uložení fotografií do souboru pomocí knihovny h5py
    with h5py.File('photos.h5', 'w') as file:
        file.create_dataset('photo1', data=image1)
        file.create_dataset('photo2', data=image2)


def h5_load():
    # Načtení fotografií z h5py souboru
    with h5py.File('photos.h5', 'r') as file:
        loaded_photo1 = file['photo1'][()]
        loaded_photo2 = file['photo2'][()]


"""# Zobrazení obou obrázků
plt.figure()
plt.tight_layout()
plt.subplot(121)
plt.imshow(loaded_photo1)
plt.subplot(122)
plt.imshow(loaded_photo2)
plt.show()"""


def h5_save_gzip():
    # Uložení fotografií do souboru pomocí knihovny h5py
    with h5py.File('photos_gzip.h5', 'w') as file:
        file.create_dataset('photo1', data=image1, compression='gzip')
        file.create_dataset('photo2', data=image2, compression='gzip')


def h5_load_gzip():
    # Načtení fotografií z h5py souboru
    with h5py.File('photos_gzip.h5', 'r') as file:
        loaded_photo1 = file['photo1'][()]
        loaded_photo2 = file['photo2'][()]

    return loaded_photo1, loaded_photo2


def h5_save_lzf():
    # Uložení fotografií do souboru pomocí knihovny h5py
    with h5py.File('photos_lzf.h5', 'w') as file:
        file.create_dataset('photo1', data=image1, compression='lzf')
        file.create_dataset('photo2', data=image2, compression='lzf')


def h5_load_lzf():
    # Načtení fotografií z h5py souboru
    with h5py.File('photos_lzf.h5', 'r') as file:
        loaded_photo1 = file['photo1'][()]
        loaded_photo2 = file['photo2'][()]

    return loaded_photo1, loaded_photo2


"""def h5_save_blosc():
    # Uložení fotografií do souboru pomocí knihovny h5py
    with h5py.File('photos_blosc.h5', 'w') as file:
        file.create_dataset('photo1', data=image1, compression='blosc')
        file.create_dataset('photo2', data=image2, compression='blosc')


def h5_load_blosc():
    # Načtení fotografií z h5py souboru
    with h5py.File('photos_blosc.h5', 'r') as file:
        loaded_photo1 = file['photo1'][()]
        loaded_photo2 = file['photo2'][()]


def h5_save_bitshuffle():
    # Uložení fotografií do souboru pomocí knihovny h5py
    with h5py.File('photos_bitshuffle.h5', 'w') as file:
        file.create_dataset('photo1', data=image1, compression='blosc:bitshuffle')
        file.create_dataset('photo2', data=image2, compression='blosc:bitshuffle')


def h5_load_bitshuffle():
    # Načtení fotografií z h5py souboru
    with h5py.File('photos_bitshuffle.h5', 'r') as file:
        loaded_photo1 = file['photo1'][()]
        loaded_photo2 = file['photo2'][()]
"""

"""# Zobrazení obou obrázků
plt.figure()
plt.tight_layout()
plt.subplot(121)
plt.imshow(loaded_photo1)
plt.subplot(122)
plt.imshow(loaded_photo2)
plt.show()
"""


def pil_save():
    # Uložení dvou obrázků do jednoho souboru ve formátu TIFF
    with Image.new('RGB', (image1.width + image2.width, max(image1.height, image2.height))) as tiff_image:
        tiff_image.paste(image1, (0, 0))
        tiff_image.paste(image2, (image1.width, 0))
        tiff_image.save('images.tiff')


def pil_load():
    # Načtení obrázku z formátu TIFF
    with Image.open('images.tiff') as tiff_image:
        width, height = tiff_image.size
        half_width = width // 2

        # Rozdělení obrázku na dva samostatné obrázky
        loaded_photo1 = tiff_image.crop((0, 0, half_width, height))
        loaded_photo2 = tiff_image.crop((half_width, 0, width, height))


"""# Zobrazení obou obrázků
plt.figure()
plt.tight_layout()
plt.subplot(121)
plt.imshow(loaded_photo1)
plt.subplot(122)
plt.imshow(loaded_photo2)
plt.show()"""

# Načtení fotografií jako numpy pole (použijte své vlastní cesty k fotografiím)
image1 = plt.imread('photos/IMG_0385.JPG')
image2 = plt.imread('photos/IMG_0417.JPG')

"""# Zobrazení obou obrázků
plt.figure()
plt.tight_layout()
plt.subplot(121)
plt.imshow(image1)
plt.subplot(122)
plt.imshow(image2)
plt.show()"""

"""time00 = timeit.timeit(lambda: h5_save(), number=5)
time01 = timeit.timeit(lambda: h5_load(), number=5)
time1 = timeit.timeit(lambda: h5_save_gzip(), number=5)
time2 = timeit.timeit(lambda: h5_load_gzip(), number=5)
time3 = timeit.timeit(lambda: h5_save_lzf(), number=5)
time4 = timeit.timeit(lambda: h5_load_lzf(), number=5)


image1 = Image.open('photos/IMG_0385.JPG')
image2 = Image.open('photos/IMG_0417.JPG')

time5 = timeit.timeit(lambda: pil_save(), number=5)
time6 = timeit.timeit(lambda: pil_load(), number=5)

times = [time00, time01, time1, time2, time3, time4]

for i in range(len(times)):
    print(f"Čas {i + 1}: {times[i]}")"""



# Zobrazení obou obrázků
plt.figure()
plt.subplot(121)
plt.imshow(image1)
plt.tight_layout()
plt.subplot(122)
plt.imshow(image2)
plt.tight_layout()

loaded_photo1, loaded_photo2 = h5_load_gzip()
plt.figure()
plt.subplot(121)
plt.imshow(loaded_photo1)
plt.tight_layout()
plt.subplot(122)
plt.imshow(loaded_photo2)
plt.tight_layout()

if loaded_photo1.all() == image1.all() and loaded_photo2.all() == image2.all():
    print("GZIP OK")

loaded_photo1, loaded_photo2 = h5_load_lzf()
plt.figure()
plt.subplot(121)
plt.imshow(loaded_photo1)
plt.tight_layout()
plt.subplot(122)
plt.imshow(loaded_photo2)
plt.tight_layout()

if loaded_photo1.all() == image1.all() and loaded_photo2.all() == image2.all():
    print("LZF OK")


plt.show()
