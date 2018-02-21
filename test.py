from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy
from numpy import asarray
import csv
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import model_from_json

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

def parse_output(cat):
    if cat[0] > 0.5:
        return 0
    if cat[1] > 0.5:
        return 1
    if cat[2] > 0.5:
        return 2
    if cat[3] > 0.5:
        return 3
    return -1

track_pics, directions = read_data()

print(track_pics)
print(directions)

train_track_pics, test_track_pics, train_directions, test_directions = train_test_split(track_pics, directions, train_size=0.2)

train_catagories = to_categorical(train_directions)
test_catagories = to_categorical(test_directions)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
score = loaded_model.evaluate(test_track_pics, test_catagories, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

prediction = loaded_model.predict(test_track_pics)
print(prediction)
prediction = numpy.fromiter((parse_output(xi) for xi in prediction), prediction.dtype)
print(prediction)
print(numpy.mean(numpy.equal(prediction, test_directions)))
print(numpy.nanmax(prediction))

# for x in numpy.nditer(prediction):
    # print(x)
