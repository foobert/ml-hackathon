from keras.models import Sequential
from keras.layers import Dense
import numpy
from numpy import asarray
import csv
from keras.utils import to_categorical

def label(s):
    if s == 'F':
        return 0
    if s == 'S':
        return 1
    if s == 'L':
        return 2
    if s == 'R':
        return 3

def read_data():
    track_pics = []
    directions = []
    with open("track_data.csv") as file:
        reader = csv.reader(file, delimiter=";")
        for sample in reader:
            track_pics.append(asarray(sample[:-1]).astype(int))
            directions.append(label(sample[-1].upper()))
    return asarray(track_pics), asarray(directions)

track_pics, directions = read_data()

y_binary = to_categorical(directions)

print(track_pics)
print(directions)
print(y_binary)

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=448))
model.add(Dense(units=4, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(track_pics, y_binary, epochs=5, batch_size=32)

score = model.evaluate(track_pics, y_binary, verbose=0)
print(score)

prediction = model.predict(asarray([track_pics[0]]))
print(prediction)
