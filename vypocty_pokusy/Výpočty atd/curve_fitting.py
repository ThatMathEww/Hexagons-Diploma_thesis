import numpy as np
import matplotlib.pyplot as plt

# Připravte data x a y
x = np.array([1, 2, 3, 4, 5])
y = np.array([1.1, 4.2, 9.2, 16.4, 26.8])

# Definujte matici A a vektor b pro model y = a1*x**2 + a2*x + a3
A = np.vstack([x ** 4, x ** 3, x ** 2, x, np.ones_like(x)]).T
b = y

# Použijte metodu nejmenších čtverců k nalezení koeficientů
coefficients, resid, rank, s = np.linalg.lstsq(A, b, rcond=None)

coefficients_ = np.polyfit(x, y, 4)

# Koeficienty a1, a2 a a3
a1, a2, a3, a4, a5 = coefficients
a1_, a2_, a3_, a4_, a5_ = coefficients_

# Vytvořte body pro fitovanou kvadratickou funkci
x_fit = np.linspace(min(x), max(x), 100)
y_fit = (a1 * x_fit ** 4) + (a2 * x_fit ** 3) + (a3 * x_fit ** 2) + (a4 * x_fit) + a5

residuals = y - ((a1 * x ** 4) + (a2 * x ** 3) + (a3 * x ** 2) + (a4 * x) + a5)
r_mse = np.sqrt(np.mean(residuals**2))


# Výsledky
print(f'a1: {a1}, a2: {a2}, a3: {a3}')
print("přesnost:", r_mse)

# Vykreslení dat a fitované funkce
plt.plot(x, y, label='Data')
plt.plot(x_fit, y_fit, 'r', label='Fitovaná kvadratická funkce')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
