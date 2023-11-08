import numpy as np

matrices = [np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]]),
            np.array([[9, 8, 7],
                      [6, 5, 4],
                      [3, 2, 1]])]

# Uložení matic
np.savez('matrices.npz', **{f"var_{i+1}": matrix for i, matrix in enumerate(matrices)})

# Načítání matic
loaded_data = np.load('matrices.npz')
m1, m2 = [loaded_data[f"var_{i+1}"] for i in range(len(loaded_data))]

print(m1,"\n\n", m2)

"""for i, matrix in enumerate(loaded_matrices):
    print(f"Matrix {i+1}:\n{matrix}")"""
