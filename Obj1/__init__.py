from keras.models import Sequential
from keras.layers import Dense
import numpy


def builder() -> Sequential:
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=100))
    model.add(Dense(units=10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model


def demo() -> None:
    print("It works :)")
