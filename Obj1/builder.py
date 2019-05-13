# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras import layers
import numpy as np


def shuffle(tab: np.array) -> np.array:
    indices = np.arange(len(tab))
    np.random.shuffle(indices)
    return tab[indices]


print('=' * 50)
print('Generating data')


x = np.array([[i/100, j/100] for i in range(10) for j in range(10)])
x = shuffle(x)
y = np.array(list(map(sum, x)))


validation_length = 0.9
split_at = int(validation_length * len(x))
x_train, x_val = x[:split_at], x[split_at:]
y_train, y_val = y[:split_at], y[split_at:]

for i in range(len(y_val)):
    print("{} -> {}".format(x_val[i], y_val[i]))

print('=' * 50)
print('Building network')


model = Sequential()

model.add(layers.Dense(1, activation='linear', input_dim=2, use_bias=False))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=1, validation_data=(x_val, y_val))
model.save_weights('weights.h5')

"""
print('=' * 50)
with open('weights.h5') as f:
    print(f.read())

print('=' * 50)
"""
