# input: None
# output: 
# 1. indices of features when mean, min, std, and energy functions were applied only to time domain signals
# 2. names of those features

import os
import numpy as np

def get_features():

	current_dir = os.getcwd()
	dir_X_train = current_dir  + '/UCI HAR Dataset/features.txt'

	all_feature_names = np.loadtxt(dir_X_train, dtype = 'str',delimiter="\n")
	
	# time domain features are the first 266 features
	feature_names = all_feature_names[0:265]

	feature_index=0
	# mean, min, std, energy are the less complex mathemical functions to calculate, among others
	# in the list such as correlection, sma etc.

	f_indices = []
	for i in feature_names:
	    if ('mean' in i or 'min' in i or 'std' in i or 'energy' in i):
	        f_indices.append(feature_index)
	    
	    feature_index+=1

	f_indices = np.array(f_indices)

	selected_feature_names = feature_names[f_indices]
	return f_indices, selected_feature_names


