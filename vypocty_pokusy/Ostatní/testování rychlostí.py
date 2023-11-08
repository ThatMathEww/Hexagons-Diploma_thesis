import numpy as np
import cv2
import timeit
from line_profiler import LineProfiler
from multiprocessing import Process
import multiprocessing
import threading

# Vytvořte testovací data
image1 = cv2.imread('photos/IMG_0385.JPG')
image2 = cv2.imread('photos/IMG_0417.JPG')

method = cv2.SIFT_create()


# Funkce pro první výpočet
def vypocet1():
    # kód pro první výpočet
    keypoints1, descriptors1 = method.detectAndCompute(image1, None)


# Funkce pro druhý výpočet
def vypocet2():
    # kód pro druhý výpočet
    keypoints2, descriptors2 = method.detectAndCompute(image2, None)


def thread():
    # Spuštění výpočtu v paralelních vláknech
    thread1, thread2 = threading.Thread(target=vypocet1), threading.Thread(target=vypocet2)
    thread1.start(), thread2.start()

    # Čekání na dokončení všech vláken
    thread1.join(), thread2.join()


def multiprocess():
    # Spuštění výpočtu v paralelních procesech
    process1, process2 = Process(target=vypocet1), Process(target=vypocet2)
    process1.start(), process2.start()

    # Čekání na dokončení všech procesů
    process1.join(), process2.join()


def normal_calculate():
    vypocet1(), vypocet2()


def is_multiprocessing_supported():
    num_cpus = multiprocessing.cpu_count()
    print(num_cpus)
    if num_cpus > 1:
        return True
    else:
        return False


if __name__ == '__main__':
    # Test, zda je zařízení vhodné pro multiprocessing
    if is_multiprocessing_supported():
        print("Zařízení je vhodné pro multiprocessing.")
    else:
        print("Zařízení není vhodné pro multiprocessing.")

    print("Začátek měření")

    # Měření času

    # time1 = timeit.timeit(lambda: multiprocess(), number=5)

    print("První hotov")

    # time2 = timeit.timeit(lambda: multiprocess(), number=1)

    print("Druhý hotov")

    # time3 = timeit.timeit(lambda: normal_calculate(), number=5)

    print("Třetí hotov")

    # Výpis výsledků
    # print("Čas 1:", time1)
    # print("Čas 2:", time2)
    # print("Čas 3:", time3)

"""blue, green, red = image1[:, :, 0], image1[:, :, 1], image1[:, :, 2]
blue = blue * 2
red = red * 1.5
green = green * 0.75
saturate, V, temp, contrast, brightness, top, down, left, right, Y, U, V2, L, A, B = 5, 150, 50.62, 10, 12.5, 7, 100, \
    125, 2115, 1463, 1.45, 8.1, 14, 56, 456"""

"""# Metoda s astype

def method_1(image):
    image_1 = image

    image_1[:, :, 0], image_1[:, :, 1], image_1[:, :, 2] = blue, green, red

    image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    image_temp[:, :, 1:] = np.multiply(image_temp[:, :, 1:], np.array([saturate, V]))
    image_temp[:, :, 1:] = np.clip(image_temp[:, :, 1:], 0, 255)
    image_temp[:, :, 0] = image_temp[:, :, 0] + temp
    image_temp[:, :, 0] = np.clip(image_temp[:, :, 0], 0, 179)
    image_2 = cv2.cvtColor(image_temp, cv2.COLOR_HSV2BGR_FULL)

    image_1 = cv2.addWeighted(image_1, 0.5, image_2, 0.5, 0)

    image_temp = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image_temp[:, :, :] = np.multiply(image_temp[:, :, :], np.array([Y, U, V2]))
    image_temp[:, :, :] = np.clip(image_temp[:, :, :], 0, 255)
    image_2 = cv2.cvtColor(image_temp, cv2.COLOR_YUV2RGB)

    image_1 = cv2.addWeighted(image_1, 0.9, image_2, 0.1, 0)

    image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_temp[:, :, :] = np.multiply(image_temp[:, :, :], np.array([L, A, B]))
    image_temp[:, :, :] = np.clip(image_temp[:, :, :], 0, 255)
    image_2 = cv2.cvtColor(image_temp, cv2.COLOR_LAB2BGR)

    image_1 = cv2.addWeighted(image_1, 0.9, image_2, 0.1, 0)

    fin_photo = cv2.convertScaleAbs(image_1, alpha=contrast, beta=brightness)[top:down, left:right]
    fin_photo = cv2.cvtColor(fin_photo, cv2.COLOR_BGR2GRAY)

    return fin_photo


# Metoda bez astype

def method_2(image):
    image_1 = image

    image_1 = np.dstack((blue, green, red)).astype(np.uint8)

    image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    image_temp[:, :, 1:] = (image_temp[:, :, 1:] * np.array([saturate, V])).astype(np.uint8)
    np.clip(image_temp[:, :, 1:], 0, 255, out=image_temp[:, :, 1:])
    image_temp[:, :, 0] = (image_temp[:, :, 0] + temp).astype(np.uint8)
    np.clip(image_temp[:, :, 0], 0, 179, out=image_temp[:, :, 0])
    image_2 = cv2.cvtColor(image_temp, cv2.COLOR_HSV2BGR_FULL)

    image_1 = cv2.addWeighted(image_1, 0.5, image_2, 0.5, 0)

    image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_temp[:, :, :] = (image_temp[:, :, :] * np.expand_dims(np.array([Y, U, V2]), axis=(0, 1))).astype(np.uint8)
    np.clip(image_temp[:, :, :], 0, 255, out=image_temp[:, :, :])
    image_2 = cv2.cvtColor(image_temp, cv2.COLOR_YUV2RGB)

    image_1 = cv2.addWeighted(image_1, 0.9, image_2, 0.1, 0)

    image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_temp[:, :, :] = (image_temp[:, :, :] * np.array([L, A, B])).astype(np.uint8)
    np.clip(image_temp[:, :, :], 0, 255, out=image_temp[:, :, :])
    image_2 = cv2.cvtColor(image_temp, cv2.COLOR_LAB2BGR)

    image_1 = cv2.addWeighted(image_1, 0.9, image_2, 0.1, 0)

    fin_photo = cv2.convertScaleAbs(image_1, alpha=contrast, beta=brightness)[top:down, left:right]

    fin_photo = cv2.cvtColor(fin_photo, cv2.COLOR_BGR2GRAY)

    return fin_photo


# Metoda bez astype

def method_3(image):
    image_1 = image

    image_1[:, :, 0], image_1[:, :, 1], image_1[:, :, 2] = blue, green, red

    image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    image_temp[:, :, 1], image_temp[:, :, 2] = image_temp[:, :, 1] * saturate, image_temp[:, :, 2] * V
    np.clip(image_temp[:, :, 1:], 0, 255, out=image_temp[:, :, 1:])
    image_temp[:, :, 0] = image_temp[:, :, 0] + temp
    np.clip(image_temp[:, :, 0], 0, 179, out=image_temp[:, :, 0])
    image_2 = cv2.cvtColor(image_temp, cv2.COLOR_HSV2BGR_FULL)

    image_1 = cv2.addWeighted(image_1, 0.5, image_2, 0.5, 0)

    image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_temp[:, :, 0], image_temp[:, :, 1], image_temp[:, :, 2] = image_temp[:, :, 0] * Y, \
                                                                    image_temp[:, :, 1] * U, image_temp[:, :, 2] * V2
    np.clip(image_temp[:, :, :], 0, 255, out=image_temp[:, :, :])
    image_2 = cv2.cvtColor(image_temp, cv2.COLOR_YUV2RGB)

    image_1 = cv2.addWeighted(image_1, 0.9, image_2, 0.1, 0)

    image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_temp[:, :, 0], image_temp[:, :, 1], image_temp[:, :, 2] = image_temp[:, :, 0] * L, \
                                                                    image_temp[:, :, 1] * A, image_temp[:, :, 2] * B
    np.clip(image_temp[:, :, :], 0, 255, out=image_temp[:, :, :])
    image_2 = cv2.cvtColor(image_temp, cv2.COLOR_LAB2BGR)

    # Využití paralelního zpracování pomocí OpenCV
    image_1 = cv2.addWeighted(image_1, 0.9, image_2, 0.1, 0)

    fin_photo = cv2.convertScaleAbs(image_1, alpha=contrast, beta=brightness)[top:down, left:right]

    fin_photo = cv2.cvtColor(fin_photo, cv2.COLOR_BGR2GRAY)

    return fin_photo


def method_4(image):
    image_1 = image
    image_1 = np.dstack((blue, green, red)).astype(np.uint8)

    image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    image_temp[:, :, 1:] = np.clip((image_temp[:, :, 1:] * np.array([saturate, V])).astype(np.uint8), 0, 255)
    image_temp[:, :, 0] = np.clip((image_temp[:, :, 0] + temp).astype(np.uint8), 0, 179)
    image_2 = cv2.cvtColor(image_temp, cv2.COLOR_HSV2BGR_FULL)

    image_1 = cv2.addWeighted(image_1, 0.5, image_2, 0.5, 0)

    image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_temp[:, :, :] = np.clip(
        (image_temp[:, :, :] * np.expand_dims(np.array([Y, U, V2]), axis=(0, 1))).astype(np.uint8), 0, 255)
    image_2 = cv2.cvtColor(image_temp, cv2.COLOR_YUV2RGB)

    image_1 = cv2.addWeighted(image_1, 0.9, image_2, 0.1, 0)

    image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_temp[:, :, :] = np.clip((image_temp[:, :, :] * np.array([L, A, B])).astype(np.uint8), 0, 255)
    image_2 = cv2.cvtColor(image_temp, cv2.COLOR_LAB2BGR)

    image_1 = cv2.addWeighted(image_1, 0.9, image_2, 0.1, 0)

    fin_photo = cv2.convertScaleAbs(image_1, alpha=contrast, beta=brightness)[top:down, left:right]

    fin_photo = cv2.cvtColor(fin_photo, cv2.COLOR_BGR2GRAY)

    return fin_photo


def method_5(input_image):
    image_out_1 = input_image
    image_out_1[:, :, 0], image_out_1[:, :, 1], image_out_1[:, :, 2] = blue, green, red

    image_out_1, input_image = image_out_1[top:down, left:right], input_image[top:down, left:right]

    image_temp = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV_FULL)
    np.clip(image_temp[:, :, 0] + temp, 0, 179, out=image_temp[:, :, 0])
    np.clip(image_temp[:, :, 1] * saturate, 0, 255, out=image_temp[:, :, 1])
    np.clip(image_temp[:, :, 2] * V, 0, 255, out=image_temp[:, :, 2])
    image_out_2 = cv2.cvtColor(image_temp, cv2.COLOR_HSV2BGR_FULL)

    image_out_1 = cv2.addWeighted(image_out_1, 0.5, image_out_2, 0.5, 0)

    image_temp = cv2.cvtColor(input_image, cv2.COLOR_BGR2YUV)
    np.clip(image_temp[:, :, 0] * Y, 0, 255, out=image_temp[:, :, 0])
    np.clip(image_temp[:, :, 1] * U, 0, 255, out=image_temp[:, :, 1])
    np.clip(image_temp[:, :, 2] * V2, 0, 255, out=image_temp[:, :, 2])
    image_out_2 = cv2.cvtColor(image_temp, cv2.COLOR_YUV2RGB)

    image_out_1 = cv2.addWeighted(image_out_1, 0.9, image_out_2, 0.1, 0)

    image_temp = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)
    np.clip(image_temp[:, :, 0] * L, 0, 255, out=image_temp[:, :, 0])
    np.clip(image_temp[:, :, 1] * A, 0, 255, out=image_temp[:, :, 1])
    np.clip(image_temp[:, :, 2] * B, 0, 255, out=image_temp[:, :, 2])
    image_out_2 = cv2.cvtColor(image_temp, cv2.COLOR_LAB2BGR)

    # Využití paralelního zpracování pomocí OpenCV
    image_out_1 = cv2.addWeighted(image_out_1, 0.9, image_out_2, 0.1, 0)

    image_out_1 = cv2.cvtColor(cv2.convertScaleAbs(image_out_1, alpha=contrast, beta=brightness), cv2.COLOR_BGR2GRAY)

    return image_out_1


def a():
    image_temp = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV_FULL)
    image_temp[:, :, 0], image_temp[:, :, 1], image_temp[:, :, 2] = image_temp[:, :, 0] * 12.8, \
                                                                    image_temp[:, :, 1] * 0.12587, image_temp[:, :,
                                                                                                   2] * -45
    np.clip(image_temp, 0, 255, out=image_temp)
    np.clip(image_temp, 0, 255, out=image_temp)
    np.clip(image_temp, 0, 255, out=image_temp)
    np.clip(image_temp, 0, 255, out=image_temp)


def b():
    image_temp = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV_FULL)
    image_temp[:, :, 0], image_temp[:, :, 1], image_temp[:, :, 2] = image_temp[:, :, 0] * 12.8, \
                                                                    image_temp[:, :, 1] * 0.12587, image_temp[:, :,
                                                                                                   2] * -45
    np.clip(image_temp[:, :, :], 0, 255, out=image_temp[:, :, :])
    np.clip(image_temp[:, :, :], 0, 255, out=image_temp[:, :, :])
    np.clip(image_temp[:, :, :], 0, 255, out=image_temp[:, :, :])
    np.clip(image_temp[:, :, :], 0, 255, out=image_temp[:, :, :])


def c():
    image_temp = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV_FULL)
    image_temp[:, :, 0], image_temp[:, :, 1], image_temp[:, :, 2] = image_temp[:, :, 0] * 12.8, \
                                                                    image_temp[:, :, 1] * 0.12587, image_temp[:, :,
                                                                                                   2] * -45
    image_temp[:, :, :] = np.clip(image_temp, 0, 255)
    image_temp[:, :, :] = np.clip(image_temp, 0, 255)
    image_temp[:, :, :] = np.clip(image_temp, 0, 255)
    image_temp[:, :, :] = np.clip(image_temp, 0, 255)


def d():  # BEST !!!
    image_temp = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV_FULL)
    image_temp[:, :, 0], image_temp[:, :, 1], image_temp[:, :, 2] = image_temp[:, :, 0] * 12.8, \
                                                                    image_temp[:, :, 1] * 0.12587, image_temp[:, :,
                                                                                                   2] * -45
    image_temp[:, :, :] = np.clip(image_temp[:, :, :], 0, 255)
    image_temp[:, :, :] = np.clip(image_temp[:, :, :], 0, 255)
    image_temp[:, :, :] = np.clip(image_temp[:, :, :], 0, 255)
    image_temp[:, :, :] = np.clip(image_temp[:, :, :], 0, 255)"""

"""# Vytvoření instance profiléru
profiler = LineProfiler()

# Přidání funkcí ke sledování profilu
profiler.add_function(method_1)
profiler.add_function(method_2)
profiler.add_function(method_3)
profiler.add_function(method_4)

profiler.add_function(a)
profiler.add_function(b)
profiler.add_function(c)
profiler.add_function(d)

# Spuštění profilování
profiler.enable_by_count()

a()
print("\n\n")
b()
print("\n\n")
c()
print("\n\n")
d()

# Volání funkcí, které chcete profilovat
print("Začátek měření")
image_1 = cv2.imread('photos/IMG_0417.JPG')
method_1(image_1)
print("První hotov")
image_2 = cv2.imread('photos/IMG_0417.JPG')
method_2(image_2)
print("Druhý hotov")
image_3 = cv2.imread('photos/IMG_0417.JPG')
method_3(image_3)
print("Třetí hotov")
image_4 = cv2.imread('photos/IMG_0417.JPG')
method_4(image_4)
print("Čtvrtý hotov\n\n")

# Zastavení profilování
profiler.disable_by_count()

# Výpis výsledků
profiler.print_stats()"""

"""print("Začátek měření")

# Měření času
# a1 = method_1(image1)
# time1 = timeit.timeit(lambda: method_1(image1), number=10)

print("První hotov")

# a2 = method_2(image1)
# time2 = timeit.timeit(lambda: method_2(image1), number=10)

print("Druhý hotov")

a3 = method_3(image1)
#time3 = timeit.timeit(lambda: method_3(image1), number=30)

print("Třetí hotov")

# a4 = method_4(image1)
# time4 = timeit.timeit(lambda: method_4(image1), number=10)

print("Čtvrtý hotov\n")

a5 = method_5(image1)
time5 = timeit.timeit(lambda: method_5(image1), number=100)

if a3.all() == a5.all():
    print("OK")


# Výpis výsledků
# print("Čas 1:", time1)
# print("Čas 2:", time2)
# print("Čas 3:", time3)
# print("Čas 4:", time4)
print("Čas 5:", time5)  # 20.57999070000369"""

"""print(a1[10:15, 1000:1003])
print(a2[10:15, 1000:1003])
print(a3[10:15, 1000:1003])
print(a4[10:15, 1000:1003])"""
