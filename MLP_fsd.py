# This program trains a multilayer percetron on the 10 features selected by FFS with LDA as the performance metric

from numpy.random import seed
seed(100)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import History
from keras import callbacks
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
from plot_confmat import plot_confusion_matrix
from load_dataset import load_dataset
from sklearn.model_selection import train_test_split
import json
import os
from feature_indices import get_features
from sequential_FS import sequential_foward_selection

# loading data
X_train, y_train, X_test, y_test = load_dataset()

# combining training and testing for CV
X_data = np.concatenate((X_train, X_test), axis=0)
y_data = np.concatenate((y_train, y_test), axis=0)

# performing train-test split on training data to then add the test to existing test data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.30,
                                                              random_state=32)

y_train_onehot = keras.utils.to_categorical(y_train, num_classes=6)


# obtains the features such as min, max, std, energy, which require less computation as compared
# to the other functions used to create features
feature_indices, feature_names = get_features()
print(feature_indices.shape)

X_train_t = X_train[:,feature_indices]
X_test_t = X_test[:,feature_indices]

# selecting k features. Note: The features selected are same each time, they're not randomized
k = 10
k_features = sequential_foward_selection(X_train_t, y_train, k)
selected_features = feature_names[k_features]
print('The selected features are ', selected_features)

# training and testing data with only selected features
X_train_new = X_train_t[:,k_features]
X_test_new = X_test_t[:,k_features]
no_dim = X_test_new.shape[1]
print('Number of features after FSS: ', no_dim)

## Training MLP ##

# keras callback history
history = History()

start = time.time()
print("Training... ")

# MLP with one hidden layer
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=no_dim))
model.add(Dropout(0.25))
model.add(Dense(6, activation='softmax'))
optimizer = Adam(lr=0.001)

# compiling model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# max number of epochs
no_epochs=500
# early stopping with validation accuracy
callback = callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='max')

print(X_train_new.shape)
# training the model
history = model.fit(X_train_new, y_train_onehot,
          epochs=no_epochs, validation_split = 0.05, verbose=1, callbacks=[callback])


# saving model and training history
model_path = os.getcwd() + '/MLP Models/model1(MLP_fsd).h5'
model.save(model_path)
history_dict = history.history

# Save it under the form of a json file
history_path = os.getcwd() + '/MLP Models/history(MLP_fsd).txt'
json.dump(history_dict, open(history_path, 'w'))

end = time.time()
print("training took: " + str(end - start) + 'seconds')

pred = model.predict_classes(X_test_new)

##### Printing and plotting results #####
print('Testing accuracy is ' + str(accuracy_score(y_test, pred)))
classes = ['walking', 'walking up', 'walking down', 'sitting', 'standing', 'laying']
classes = np.array(classes)

conf_mat_test = confusion_matrix(y_test,pred, labels=[0,1,2,3,4,5])
plt.figure(1)
plot_confusion_matrix(conf_mat_test, classes,normalize=True,title='Confusion matrix')
fname = os.getcwd() + '/Confusion Matrices/MLP_fsd_test.png'
plt.savefig(fname, dpi=256, facecolor='w', edgecolor='b',
        orientation='portrait', papertype=None, format='png',
        frameon=True)

plt.figure(2)
pred_train = model.predict_classes(X_train_new)
print('Training accuracy is ' + str(accuracy_score(y_train, pred_train)))

conf_mat_train = confusion_matrix(y_train,pred_train, labels=[0,1,2,3,4,5])
plot_confusion_matrix(conf_mat_train, classes,normalize=True,title='Confusion matrix')
fname = os.getcwd() + '/Confusion Matrices/MLP_fsd_train.png'
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
plt.legend(['Training loss', 'Training accuracy', 'Valdiation accuracy'], loc='upper right')
fname = os.getcwd() + '/Confusion Matrices/MLP_fsd_training_plot.png'
plt.savefig(fname, dpi=256, edgecolor='b',format='png',
        frameon=True)

plt.show()

del model

