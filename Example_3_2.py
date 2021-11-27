from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras.datasets import reuters

from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np

#(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
#print("train_data shape:"+str(train_data.shape))
#word_index = reuters.get_word_index()
#reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
#decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

#def vectorize_sequences(sequences, dimension=10000):
#    results = np.zeros((len(sequences), dimension))
#    for i, sequence in enumerate(sequences):
#        results[i, sequence] = 1.
#    return results

#x_train = vectorize_sequences(train_data)
#x_test = vectorize_sequences(test_data)

#y_train = np.array(train_labels)
#y_test = np.array(test_labels)

#one_hot_train_labels = to_categorical(train_labels)
#one_hot_test_labels = to_categorical(test_labels)

#x_val = x_train[:1000]
#partial_x_train = x_train[1000:]

#y_val = one_hot_train_labels[:1000]
#partial_y_train = one_hot_train_labels[1000:]

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])

history = model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)

predictions = model.predict(x_test)
print(predictions)
