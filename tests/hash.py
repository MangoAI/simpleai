import numpy as np
from random import shuffle
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping

bits = 4
subset = 1

def binarize(n):
    return [int(c) for c in list(('{0:0' + str(bits) + 'b}').format(n))]

numbers = range(2**bits)
train = list(numbers)[:round(subset*len(numbers))]
shuffle(train)
train = np.array([binarize(n) for n in train])

model = Sequential()
model.add(Dense(units=int(bits* (1/2)), activation='linear', input_dim=bits))
# model.add(Dense(units=int(bits* (1/2)), activation='linear'))
model.add(Dense(units=bits, activation='sigmoid'))
model.compile(loss=keras.losses.mean_squared_error, optimizer=Adam())
model.fit(train, train, batch_size=2**int(bits/2), epochs=2**12, verbose=1, callbacks=[
    EarlyStopping(monitor='loss', min_delta=0.0001, patience=3, verbose=0, mode='auto')
])

accuracies = [0]*bits
for i in range(2**bits):
    prediction_p = [float(str(n)[:5]) for n in np.round(model.predict(np.array([binarize(i)]))[0], 3)]
    prediction = np.round(prediction_p)
    actual = binarize(i)
    for j in range(bits):
        if prediction[j] == actual[j]:
            accuracies[j] += 1
    print(i, prediction, prediction_p)

print(model.layers[0].get_weights())
print(model.layers[1].get_weights())
print(np.array(accuracies)/(2**bits))
print(np.average(np.array(accuracies)/(2**bits)))