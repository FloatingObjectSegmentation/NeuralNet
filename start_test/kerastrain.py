import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.optimizers import SGD
import scipy.io
import os


# Generate dummy data
x_train = np.random.random((3700, 50, 50, 50, 1))
y_train = np.random.randint(2, size=(3700, 1))

# Load real shit data

folder = '/Users/km/Desktop/playground/mag/magneuron/result/'
for i in range(3700):
    A = scipy.io.loadmat(folder + 'examples/' + str(i) + '.mat')
    A = A['objectField']
    x_train[i,:,:,:,0] = A

    y = scipy.io.loadmat(folder + 'targets/' + str(i) + '.mat')
    y = y['target']
    y = 1 if y == 1 else 0 # DRONE IS 1, OTHER IS 0
    y_train[i] = y

    if i % 100 == 0:
        print('Loaded ' + str(i) + ' examples')

x_test = x_train[2700:]
y_test = y_train[2700:]
x_train = x_train[:2700]
y_train = y_train[:2700]

# network
model = Sequential()
model.add(Conv3D(filters=32, kernel_size=(5,5,5), strides=(2,2,2),
                 padding='same', data_format='channels_last', use_bias=True,
                 input_shape=(50,50,50,1)))
model.add(Conv3D(filters=32, kernel_size=(3,3,3),
                 padding='same', data_format='channels_last', use_bias=True))
# Valid question: Should I use stride during maxpooling?
model.add(MaxPooling3D(pool_size=(2,2,2), strides=None, padding='valid', data_format='channels_last'))
model.add(Flatten(data_format='channels_last'))
model.add(Dense(units=128,activation=None,use_bias=True))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=20, epochs=5)
score = model.evaluate(x_test, y_test, batch_size=20)