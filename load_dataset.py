# This functions loads the data and returns the features and labels of test and training data

import os
import numpy as np


def load_dataset():

	current_dir = os.getcwd()
	# directories of dataset
	dir_X_train = current_dir  + '/UCI HAR Dataset/train/X_train.txt'
	dir_y_train = current_dir  + '/UCI HAR Dataset/train/y_train.txt'
	dir_X_test = current_dir  + '/UCI HAR Dataset/test/X_test.txt'
	dir_y_test = current_dir  + '/UCI HAR Dataset/test/y_test.txt'
	# loading data
	X_train = np.loadtxt(dir_X_train)
	y_train = np.loadtxt(dir_y_train).astype(int)
	X_test = np.loadtxt(dir_X_test)
	y_test = np.loadtxt(dir_y_test).astype(int)

	# changing labels from 1 to 6 to 0 to 5. This is be useful when target labels are one-hot encoded 
	y_train = y_train - np.ones(y_train.shape)
	y_test = y_test - np.ones(y_test.shape)
	y_train = y_train.astype(int)
	y_test = y_test.astype(int)

	return X_train, y_train, X_test, y_test