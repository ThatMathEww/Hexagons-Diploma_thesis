import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Dense


# Vytvoření dat pro sinus a exponenciální křivku
time_steps = 100
t = np.linspace(0, 10, time_steps)
sinus_data = np.sin(t)
exp_data = np.exp(-0.1*t)

# Připravení dat
combined_data = np.stack((sinus_data, exp_data), axis=-1)
X_train = combined_data[:-1]
y_train = combined_data[1:]

# Definice modelu
model = Sequential()
model.add(LSTM(64, input_shape=(None, 2)))
model.add(Dense(2))

# Kompilace modelu
model.compile(optimizer='adam', loss='mean_squared_error')

# Trénink modelu
model.fit(X_train.reshape(-1, 1, 2), y_train, epochs=50)

# Zvolení, kterou křivku chcete předpovídat (sinus nebo exponenciálu)
prediction_type = "sinus"  # Můžete změnit na "exp" pro exponenciální křivku

# Vyberete příslušný vstup pro predikci na základě zvoleného typu křivky
input_data = np.array([[np.sin(t[i]), np.exp(-0.1*t[i])] for i in range(time_steps)])
if prediction_type == "exp":
    input_data = input_data[:, ::-1]

# Predikce
predictions = model.predict(input_data)

# Vizualizace predikcí
import matplotlib.pyplot as plt

# plt.plt(t[1:], predictions[:, 0], label='Original')
plt.plot(t[1:], predictions[:, 1], label='Predicted')
plt.legend()
plt.show()
