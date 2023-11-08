import numpy as np
import time


x11, y11 = [], []
x12, y12 = np.empty((0,)), np.empty((0,))
x13, y13 = [], []

time1 = 0
time2 = 0
time3 = 0

for _ in range(100):
    # Generování 1000 náhodných dat pro test
    random_data = np.random.rand(1000, 2)
    keypoints1 = random_data
    keypoints2 = random_data

    # Metoda extend
    start_time = time.time()

    for kp in keypoints1:
        x11.append(kp[0])
        y11.append(kp[1])

    extend_time = time.time() - start_time
    time1 += extend_time

    # Metoda np.vstack
    start_time = time.time()

    x12 = np.concatenate([x12, keypoints1[:, 0]])
    y12 = np.concatenate([y12, keypoints1[:, 1]])

    vstack_time = time.time() - start_time
    time2 += vstack_time

    # Metoda np.vstack
    start_time = time.time()

    x13.extend(keypoints1[:, 0])
    y13.extend(keypoints1[:, 1])

    ext_time = time.time() - start_time
    time3 += ext_time

print(f"Metoda extend: {extend_time*1000} s")
print(f"Metoda np.vstack: {vstack_time*1000} s")
