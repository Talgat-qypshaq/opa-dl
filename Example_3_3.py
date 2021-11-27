from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras.datasets import reuters

from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np

#UEs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
UEs = [3]
#number of all distances 682
training_size = 618 # 80 %
testing_size = 64 #10 %

f_x = open('data/distances.txt', 'r')
x = f_x.readlines()
f_x.close()
examples = len(x)

f_d = open('data/mean_distances.txt', 'r')
d = f_d.readlines()
f_d.close()

fileName_2 = 'results/DL_time.txt'
s = open(fileName_2, "w")
s.write('')
s.close()

for UE in UEs:
    file_path = 'data/coefficients_'+str(UE);
    file_path = file_path+'.txt';
    f_y = open(file_path, 'r')
    y = f_y.readlines()
    f_y.close()
    #initialize array for all data
    X_data = np.zeros((UE, examples), dtype=int)
    Y_data = np.zeros((UE, examples))
    #initialize array for train data
    X_train = np.zeros((UE, training_size), dtype=int)
    Y_train = np.zeros((UE, training_size))
    #initialize array for test data
    X_test = np.zeros((UE, testing_size), dtype=int)
    Y_test = np.zeros((UE, testing_size))
    mean_distances = np.zeros((UE, 1))
    #initialize array for current Y
    Y_current_train = np.zeros((1, training_size), dtype=int)
    Y_current_test = np.zeros((1, testing_size))

    coefficients = np.zeros((UE, 1))

    for c in range(UE):
        for b in range(examples):
            X_data[c][b] = int(x[b].split()[c])
            Y_data[c][b] = float(y[b].split()[c])
            if(b<training_size):
                X_train[c][b] = int(x[b].split()[c])
                Y_train[c][b] = float(y[b].split()[c])
            if(b>=training_size and b<examples):
                X_test[c][b-training_size] = int(x[b].split()[c])
                Y_test[c][b-training_size] = float(y[b].split()[c])
    mean_distances[c] = int(d[0].split()[c])

    fileName_1 = 'results/DL_coefficients_'+str(UE)
    fileName_1 = fileName_1 + '.txt'
    r = open(fileName_1, "w")
    r.write('')
    r.close()
    r = open(fileName_1, "a")

    elapsed_time = 0
    train_accuracy = 0
    test_accuracy = 0

#print(X_train.ndim)
#print(X_train.shape)
#print(len(X_train))
#print(X_train.dtype)

#print(Y_train.ndim)
#print(Y_train.shape)
#print(Y_train.dtype)

    X_train = X_train.reshape(618, UE)
    #X_train = X_train.astype('float32') / 525
    X_test = X_test.reshape((64, UE))
    #X_test = X_test.astype('float32') / 525

    Y_train = Y_train.reshape(618, UE)
    Y_test = Y_test.reshape(64, UE)

    mean_distances = mean_distances.reshape(1, UE)

#Y_train = to_categorical(Y_train)
#Y_test = to_categorical(Y_test)
#y_train = np.asarray(Y_train).astype('float32')
#y_test = np.asarray(Y_test).astype('float32')
#print(mean_distances.shape)
#print(X_test.shape)
#print(Y_train.shape)
#print(Y_test.shape)
#print(y_train.shape)
#print(y_test.shape)

    model = models.Sequential()
    #model.add(layers.Dense(16, activation='relu', input_shape=(UE,)))
    #model.add(layers.Dense(16, activation='relu'))
    #model.add(layers.Dense(1, activation='sigmoid'))
    model.add(layers.Dense(1, activation='sigmoid', input_shape=(UEs[0],)))
    opt = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_absolute_error'])

    x_val = X_train[:64, :]
    partial_x_train = X_train[554:, :]
#print(x_val.shape)
#print(partial_x_train.shape)
    for each_UE in range(UE):
#y_val = Y_train[:10, :]
#partial_y_train = Y_train[10:, :]
        y_val = Y_train[:64, each_UE]
        partial_y_train = Y_train[554:, each_UE]
#print(y_val)
#print(y_val.shape)
#print(partial_y_train.shape)
#print(len(y_val))
#print(len(partial_y_train))
        history = model.fit(partial_x_train, partial_y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val))
        test_loss, test_acc = model.evaluate(X_test, Y_test[:, 0])
        print("1. Test Accuracy:"+str(test_acc)+"\n")
        model = models.Sequential()
        #model.add(layers.Dense(64, activation='relu', input_shape=(UEs[0],)))
        #model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid', input_shape=(UEs[0],)))

        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_absolute_error'])
        model.fit(partial_x_train, partial_y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val))
        test_loss, test_acc = model.evaluate(X_test, Y_test[:, 0])
        print("2. Test Accuracy:"+str(test_acc)+"\n")
        predictions = model.predict(mean_distances)
        #print('3. Predictions:', predictions)
        #for prediction in predictions:
        print('4. Predictions:', each_UE, np.round(predictions, 5))
