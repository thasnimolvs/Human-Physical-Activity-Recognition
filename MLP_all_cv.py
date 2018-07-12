from numpy.random import seed
seed(54)

import numpy as np
from sklearn.metrics import accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
import time
from load_dataset import load_dataset
from keras.callbacks import History
from keras import callbacks
from sklearn.model_selection import train_test_split
import random


# loading data
X_train, y_train, X_test, y_test = load_dataset()


# combining training and testing for CV
X_data = np.concatenate((X_train, X_test), axis=0)
y_data = np.concatenate((y_train, y_test), axis=0)


all_test_acc = []
all_train_acc = []

# Classification with various splitsof training and testing data

for iter in range(1,21):
    rand = random.randint(1, 100)
    # performing train-test split on training data to then add the test to existing test data
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.30,random_state=rand)
    # one hot target for neural network
    y_train_onehot = keras.utils.to_categorical(y_train, num_classes=6)
    print(y_train_onehot.shape)

    # input feature dimension
    no_dim = X_train.shape[1]

    # keras callback history
    history = History()

    start = time.time()
    print("Training... ")

    model = Sequential()
    # MLP with two hidden layers
    model.add(Dense(32, activation='relu', input_dim=no_dim))
    model.add(Dropout(0.25))
    model.add(Dense(6, activation='softmax'))
    optimizer = Adam(lr=0.001)

    # optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    no_epochs = 500
    callback = callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='max')
    model.fit(X_train, y_train_onehot,epochs=no_epochs, validation_split = 0.05, verbose=1, callbacks=[callback])


    pred_test = model.predict_classes(X_test)  # creating lda classifier
    pred_train = model.predict_classes(X_train)

    test_acc = accuracy_score(y_test, pred_test)
    train_acc = accuracy_score(y_train, pred_train)

    all_test_acc.append(test_acc)
    all_train_acc.append(train_acc)




all_train_acc = np.array(all_train_acc)
all_test_acc = np.array(all_test_acc)

print(all_test_acc)
print(all_test_acc.shape)
print('Std of training acc: ', np.std(all_train_acc))
print('Mean of training acc: ', np.mean(all_train_acc))

print('Std of testing acc: ', np.std(all_test_acc))
print('Mean of testing acc: ', np.mean(all_test_acc))












