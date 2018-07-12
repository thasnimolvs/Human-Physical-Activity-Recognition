# this function uses foward feature selection of the mlxtend library and
# returns selected feature indices


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mlxtend.feature_selection import SequentialFeatureSelector 
import numpy as np


def sequential_foward_selection(X_train_t, y_train, k):


	lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None)
	sfs = SequentialFeatureSelector(lda, 
	           k_features=k, 
	           forward=True, 
	           floating=False, 
	           verbose=1,
	           scoring='accuracy',
	           cv=2)

	sfs.fit(X_train_t, y_train)

	k_features = np.array(sfs.k_feature_idx_)

	return k_features

