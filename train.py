from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy
from numpy import asarray
import csv
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

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

print(track_pics)
print(directions)

train_track_pics, test_track_pics, train_directions, test_directions = train_test_split(track_pics, directions, train_size=0.8)

train_catagories = to_categorical(train_directions)
test_catagories = to_categorical(test_directions)

model = Sequential()
model.add(Dense(units=128, activation='relu', input_dim=448))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=4, activation='softmax'))
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(train_track_pics, train_catagories, epochs=5, batch_size=32)

score = model.evaluate(test_track_pics, test_catagories, verbose=1)
print(score)

with open("model.json", "w") as json_file:
    json_file.write(model.to_json())
model.save_weights("model.h5")
