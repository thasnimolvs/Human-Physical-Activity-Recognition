# LDA applied on 10 high performing features, selected by heuristic forward feature selection

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from plot_confmat import plot_confusion_matrix
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from load_dataset import load_dataset
import os
from sklearn.model_selection import train_test_split
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

# obtains the features such as min, max, std, energy, which require less computation as compared
# to the other functions used to create features
feature_indices, feature_names = get_features()
print(feature_indices.shape)

X_train_t = X_train[:,feature_indices]
X_test_t = X_test[:,feature_indices]

# selecting k features
k = 10
k_features = sequential_foward_selection(X_train_t, y_train, k)
selected_features = feature_names[k_features]
print('The selected features are ', selected_features)

# training and testing data with only selected features
X_train_new = X_train_t[:,k_features]
X_test_new = X_test_t[:,k_features]
no_dim = X_test_new.shape[1]
print('Number of features after FSS: ', no_dim)


# creating instance
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None)
# fitting model
lda.fit(X_train_new,y_train)

# predicting on training and test 
pred_train = lda.predict(X_train_new)
pred_test = lda.predict(X_test_new)


### Printing and plotting operation of results ###
print('Testing: ')
print(accuracy_score(y_test,pred_test))

classes = ['walking', 'walking up', 'walking down', 'sitting', 'standing', 'laying']
classes = np.array(classes)

conf_mat_test = confusion_matrix(y_test,pred_test, labels=[0,1,2,3,4,5])
plt.figure(1)
plot_confusion_matrix(conf_mat_test, classes,normalize=True,title='Confusion matrix')

fname = os.getcwd() + '/Confusion Matrices/LDA_fsd(test)='
plt.savefig(fname + str(k) + '.png', dpi=256, edgecolor='b',format='png',
        frameon=True)

print('Training: ')
print(accuracy_score(y_train, pred_train))
plt.figure(2)
conf_mat_train = confusion_matrix(y_train,pred_train, labels=[0,1,2,3,4,5])
plot_confusion_matrix(conf_mat_train, classes,normalize=True,title='Confusion matrix')

fname = os.getcwd() + '/Confusion Matrices/LDA_fsd(train)='
plt.savefig(fname + str(k) + '.png', dpi=256, edgecolor='b',format='png',
        frameon=True)
plt.show()




