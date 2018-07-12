from numpy.random import seed
seed(54)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import History
from keras import callbacks

from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
from plot_confmat import plot_confusion_matrix
from load_dataset import load_dataset

from sklearn.model_selection import train_test_split
import json
import os
import numpy as np
from sklearn.metrics import confusion_matrix


# loading datset
X_train, y_train, X_test, y_test = load_dataset()

# combining training and testing to shuffle it and change traintest split
X_data = np.concatenate((X_train, X_test), axis=0)
y_data = np.concatenate((y_train, y_test), axis=0)

# performing train-test split on training data to then add the test to existing test data
X_train, X_test_temp, y_train, y_test_temp = train_test_split(
         X_train, y_train, test_size=0.30, random_state=32)

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

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

no_epochs=500
callback = callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='max')

# training the model
history = model.fit(X_train, y_train_onehot,
          epochs=no_epochs, validation_split = 0.05, verbose=1, callbacks=[callback])


# saving model and training history

model_path = os.getcwd() + '/MLP Models/model1(MLP_all).h5'
model.save(model_path)

history_dict = history.history
# Save it under the form of a json file
history_path = os.getcwd() + '/MLP Models/history(MLP_all).txt'
json.dump(history_dict, open(history_path, 'w'))


end = time.time()
print("training took: " + str(end - start) + 'seconds')

pred = model.predict_classes(X_test)

print('Testing accuracy is ' + str(accuracy_score(y_test, pred)))
classes = ['walking', 'walking up', 'walking down', 'sitting', 'standing', 'laying']
classes = np.array(classes)

conf_mat_test = confusion_matrix(y_test,pred, labels=[0,1,2,3,4,5])
plt.figure(1)
plot_confusion_matrix(conf_mat_test, classes,normalize=True,title='Confusion matrix')

fname = os.getcwd() + '/Confusion Matrices/MLP_test_all.png'
plt.savefig(fname, dpi=256, facecolor='w', edgecolor='b',
        orientation='portrait', papertype=None, format='png',
        frameon=True)

plt.figure(2)
pred_train = model.predict_classes(X_train)
print('Training accuracy is ' + str(accuracy_score(y_train, pred_train)))

conf_mat_train = confusion_matrix(y_train,pred_train, labels=[0,1,2,3,4,5])
plot_confusion_matrix(conf_mat_train, classes,normalize=True,title='Confusion matrix')

fname = os.getcwd() + '/Confusion Matrices/MLP_train_all.png'
plt.savefig(fname, dpi=256, edgecolor='b',format='png',
        frameon=True)

# summarize history for loss
plt.figure(3)
plt.plot(history.history['loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model loss')
plt.ylabel('Loss/Acc')
plt.xlabel('Epoch')
plt.legend(['Training loss', 'Training accuracy', 'Valdiation accuracy'], loc='right')
fname = os.getcwd() + '/Confusion Matrices/MLP_all_training_plot.png'
plt.savefig(fname, dpi=256, edgecolor='b',format='png',
        frameon=True)

plt.show()

del model

