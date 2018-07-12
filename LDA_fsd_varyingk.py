# LDA applied on various feature sets, incrimenting from 3 features to 12 features

from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from load_dataset import load_dataset
from sklearn.model_selection import train_test_split
from feature_indices import get_features
from sequential_FS import sequential_foward_selection



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


train_score = []
test_score = []

for k in range(3,13):
	# loading data

	# selecting k features
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

	a = accuracy_score(y_test,pred_test)
	test_score.append(a)


	b = accuracy_score(y_train, pred_train)
	train_score.append(b)


print('Testing: ', test_score)
print('Training ', train_score)








