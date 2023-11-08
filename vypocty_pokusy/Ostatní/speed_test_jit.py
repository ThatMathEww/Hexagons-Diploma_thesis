import timeit
import numpy as np
import cv2
from numba import jit


@jit(nopython=True, cache=True)
def method1(matrix1, matrix2):
    min_value = np.min(matrix1)
    max_value = np.max(matrix1)
    normalized_matrix = (matrix2 - min_value) / (max_value - min_value)
    return normalized_matrix


def method2(matrix1, matrix2):
    min_value = np.min(matrix1)
    max_value = np.max(matrix1)
    normalized_matrix = cv2.normalize(matrix2, None, min_value, max_value, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    return normalized_matrix


matrix1 = np.random.rand(7000, 6000)
matrix2 = np.random.rand(7000, 6000)
time1 = timeit.timeit(lambda: method1(matrix1, matrix2), number=5)
time2 = timeit.timeit(lambda: method2(matrix1, matrix2), number=5)

print("Čas 1:", time1)
print("Čas 2:", time2)

m1 = method1(matrix1, matrix2)
m2 = method2(matrix1, matrix2)

if m1.all() == m2.all():
    print("ano")
