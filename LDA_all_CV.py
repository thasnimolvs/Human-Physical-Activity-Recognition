# This code performs LDA classification 20 times and records the prints and standard deviation for different train-
# test splits



from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from load_dataset import load_dataset
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
	X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.30,
														random_state=rand)
	# creating lda classifier
	lda2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None)

	# fitting model
	lda2.fit(X_train, y_train)

	# predicting
	pred_train = lda2.predict(X_train)
	pred_test = lda2.predict(X_test)

	test_acc = accuracy_score(y_test, pred_test)
	train_acc = accuracy_score(y_train, pred_train)
	all_test_acc. append(test_acc)
	all_train_acc.append(train_acc)


all_train_acc = np.array(all_train_acc)
all_test_acc = np.array(all_test_acc)
print(all_test_acc)
print(all_test_acc.shape)

print('Std of training acc: ', np.std(all_train_acc))
print('Mean of training acc: ', np.mean(all_train_acc))

print('Std of testing acc: ', np.std(all_test_acc))
print('Mean of testing acc: ', np.mean(all_test_acc))





