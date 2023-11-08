import numpy as np
import cv2
import matplotlib.pyplot as plt
from numba import jit
import time

path = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\hex_mer\photos\01_12s\original\IMG_0415.JPG'
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
template1 = cv2.imread(r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\hex_mer\data\scale\12.png',
                       0)[30:-90, 215:-45]
height1, width1 = template1.shape


def mat(img, start, end, h, w, temp):
    pixel_values = np.zeros((img.shape[0] - h, img.shape[1] - w))
    for x in range(start[0], end[1] - w):
        for y in range(start[1], end[0] - h):
            window = img[y:y + h, x:x + w]
            pixel_values[y, x] = cv2.matchTemplate(temp, window, cv2.TM_CCOEFF_NORMED)[0][0]
    return pixel_values


"""start_time = time.time()
pixel_values_np = mat(image, np.array([0, 0]), np.array(image.shape) // 50, height1, width1, template1)
end_time = time.time()

max_position = list(np.unravel_index(np.argmax(pixel_values_np), pixel_values_np.shape))
print(max_position)

print("\t  Délka výpočtu:", round((end_time - start_time), 2), "sekund")
plt.figure()
plt.title("mat")
plt.imshow(pixel_values_np, cmap='jet', interpolation='nearest')
plt.colorbar()
plt.tight_layout()

del pixel_values_np"""


def mat2(img, start, end, h, w, temp):
    min1 = np.min(img)
    mm = np.max(img) - min1
    img2 = (temp - min1) / mm
    img1 = (img - min1) / mm

    block_size = 500  # Velikost bloku pro zpracování

    pixel_values = np.zeros((img.shape[0] - h, img.shape[1] - w))
    img1 = np.lib.stride_tricks.sliding_window_view(img1, (w, h))
    win_y, win_x = img1.shape[:2]

    x = np.linspace(0, win_x - 1, (win_x // block_size), dtype=np.int32)
    y = np.linspace(0, win_y - 1, (win_y // block_size), dtype=np.int32)

    for i in range(len(x) - 1):
        for j in range(len(y) - 1):
            window = img1[y[j]:y[j + 1], x[i]:y[i + 1]]
            distances = np.linalg.norm(window - img2, axis=(2, 3))
            pixel_values[y[j]:y[j + 1], x[i]:y[i + 1]] += distances

    return pixel_values


"""start_time = time.time()
pixel_values_np = mat2(image, np.array([0, 0]), np.array(image.shape) // 30, height1, width1, template1)
end_time = time.time()

max_position = list(np.unravel_index(np.argmin(pixel_values_np), pixel_values_np.shape))
print(max_position)

print("\t  Délka výpočtu:", round((end_time - start_time), 2), "sekund")
plt.figure()
plt.title("mat3")
plt.imshow(pixel_values_np, cmap='jet', interpolation='nearest')
plt.colorbar()
plt.tight_layout()

del pixel_values_np"""


# plt.show()


@jit(nopython=True, cache=True)
def mat3(img, start, end, h, w, temp):
    img = img[start[0]:end[0], start[1]:end[1]]

    pixel_values = np.zeros((end[0] - start[0] - h, end[1] - start[1] - w))

    min1 = np.min(img)
    mm = np.max(img) - min1
    img2 = (temp - min1) / mm
    img1 = (img - min1) / mm

    windows = np.lib.stride_tricks.sliding_window_view(img1, (w, h))
    # TODO try: except numpy.core._exceptions._ArrayMemoryError jeden po jednom     ###############################

    for x in range(start[1], end[1] - w):
        for y in range(start[0], end[0] - h):
            window = windows[y, x]
            pixel_values[y, x] = np.linalg.norm(window - img2)

    return pixel_values


start_time = time.time()
pixel_values_np = mat3(image, np.array([0, 0]), np.array(image.shape) // 30, height1, width1, template1)
end_time = time.time()

max_position = list(np.unravel_index(np.argmin(pixel_values_np), pixel_values_np.shape))
print(max_position)

print("\t  Délka výpočtu:", round((end_time - start_time), 2), "sekund")
plt.figure()
plt.title("mat3")
plt.imshow(pixel_values_np, cmap='jet', interpolation='nearest')
plt.colorbar()
plt.tight_layout()

del pixel_values_np


@jit(nopython=True, cache=True)
def calculate(img, start, end, h, w, templ):
    img = img[start[0]:end[0], start[1]:end[1]]
    pixel_values = np.zeros((end[0] - start[0] - h, end[1] - start[1] - w))

    min1 = np.min(img)
    mm = np.max(img) - min1
    img2 = (templ - min1) / mm

    for x in range(start[1], end[1] - w):
        for y in range(start[0], end[0] - h):
            window = img[y:y + h, x:x + w]

            img1 = (window - min1) / mm

            pixel_values[y, x] = np.linalg.norm(img1 - img2)
    return pixel_values


start_time = time.time()
pixel_values_np = calculate(image, np.array([0, 0]), np.array(image.shape) // 30, height1, width1, template1)
end_time = time.time()

min_position = list(np.unravel_index(np.argmin(pixel_values_np), pixel_values_np.shape))
print(min_position)

print("\t  Délka výpočtu:", round((end_time - start_time), 2), "sekund")
plt.figure()
plt.title("calculate")
plt.imshow(pixel_values_np, cmap='jet', interpolation='nearest')
plt.colorbar()
plt.tight_layout()
plt.show()


@jit(nopython=True, cache=True)
def method_best(img1, img2):
    min1, min2, max1, max2 = np.min(img1), np.min(img2), np.max(img1), np.max(img2)
    img1, img2 = (img1 - min1) / (max1 - min1), (img2 - min2) / (max2 - min2)
    return np.linalg.norm(img1 - img2)


@jit(nopython=True, cache=True)
def calculate(img, start, end, h, w, t):
    pixel_values = np.zeros((img.shape[0] - h, img.shape[1] - w))
    for x in range(start[0], end[0]):
        for y in range(start[1], end[1]):
            pic2 = img[y:y + h, x:x + w]
            pixel_values[y, x] = method_best(t, pic2)
    return pixel_values


"""pixel_values_np = calculate(image, top_left - 100, top_left + 10, height, width, template)
pixel_values_np = pixel_values_np[top_left[1] - 100:top_left[1] + 10, top_left[0] - 100:top_left[0] + 10]
min_position = list(np.unravel_index(np.argmin(pixel_values_np), pixel_values_np.shape))

plt.figure()
plt.imshow(pixel_values_np, cmap='jet', interpolation='nearest')
plt.colorbar()
plt.tight_layout()
plt.show()"""

if False:
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    """rectangle = plt.Rectangle((min_position[0] + top_left[0] - 100, min_position[1] + top_left[1] - 100),
                              width, height, edgecolor='dodgerblue',
                              facecolor='none')
    plt.gca().add_patch(rectangle)"""
    rectangle = plt.Rectangle((top_left1[0], top_left1[1]), width1, height1, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rectangle)
    rectangle = plt.Rectangle((top_left2[0], top_left2[1]), width2, height2, edgecolor='blue', facecolor='none')
    plt.gca().add_patch(rectangle)
    plt.tight_layout()
plt.show()
