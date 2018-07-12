# LDA applied to all the features. Confusion matrices of test and train are displayed

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import sklearn
import matplotlib.pyplot as plt
from plot_confmat import plot_confusion_matrix
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from load_dataset import load_dataset
import os
from sklearn.model_selection import train_test_split



# loading data
X_train, y_train, X_test, y_test = load_dataset()


# combining training and testing for CV
X_data = np.concatenate((X_train, X_test), axis=0)
y_data = np.concatenate((y_train, y_test), axis=0)



# performing train-test split on training data to then add the test to existing test data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.30,
													random_state=33)
# creating lda classifier
lda2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None)

# fitting model
lda2.fit(X_train, y_train)

# predicting
pred_train = lda2.predict(X_train)
pred_test = lda2.predict(X_test)

# test confusion matrix
conf_mat_test = confusion_matrix(y_test, pred_test, labels=[0, 1, 2, 3, 4, 5])
# confusion matrix of training results
conf_mat_train = confusion_matrix(y_train, pred_train, labels=[0, 1, 2, 3, 4, 5])

classes = ['walking', 'walking up', 'walking down', 'sitting', 'standing', 'laying']
classes = np.array(classes)

### Printing and plotting operation of results ###

# Test accuracy
print('Test Accuracy: ', accuracy_score(y_test, pred_test))
plt.figure(1)
plot_confusion_matrix(conf_mat_test, classes,normalize=True,title='Confusion matrix')
fname_test = os.getcwd() + '/Confusion Matrices/LDA_test_all.png'
plt.savefig(fname_test, dpi=256, edgecolor='b',format='png',
        frameon=True)

# calculating train accuracy score
print('Train Accuracy: ', accuracy_score(y_train, pred_train))
plt.figure(2)
plot_confusion_matrix(conf_mat_train, classes,normalize=True,title='Confusion matrix')
fname = os.getcwd() + '/Confusion Matrices/LDA_train_all.png'
plt.savefig(fname, dpi=256, edgecolor='b',format='png',
        frameon=True)
plt.show()







