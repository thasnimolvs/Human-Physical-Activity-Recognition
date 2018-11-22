# Human-Physical-Activity-Recognition

This repo reports the results of a linear discriminant classifier and a multilayer perceptron classifying 6 human activities using 516 handcrafted features created from accelerometer and gyroscope triaxial data on a smartphone, along with a subset of features selected from it. Test accuracies of 97.83% and 95.18% were obtained using linear discriminant classifier and the MLP respectively. However, this will not be computationally feasible on a smartphone. So heuristic forward feature selection was used with the LDA classifier to select 10 high performing features. These hand-crafted features were constraint to the time domain and ease of computation. Using these 10 features, the MLP obtained the higher test accuracy of 92.23% as compared to 90.84% using LDA. 

The dataset is available at (move into root before running):
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones


Classification using all 561 handcrafted features (Multilayer peceptron): 

![alt text](https://github.com/sid-sundrani/Human-Physical-Activity-Recognition/blob/master/Confusion%20Matrices/MLP_train_all.png)

Classification using  10 handcrafted features (Multilayer peceptron): 
![alt text] (https://github.com/sid-sundrani/Human-Physical-Activity-Recognition/blob/master/Confusion%20Matrices/MLP_fsd_test.png)


FILE DESCRIPTIONS:

1. load_dataset is a function that loads the data. 
2. LDA_all.py is the code that applies a linear discriminant classifier to all the features. Confusion matrices are displayed and saved
3. MLP_all.py is the code that trains a multilayer perceptron on all the features. Confusion matrices are displayed and saved 
4. plot_confmat.py is a function that plots the confusion matrices. 
5. LDA_all_CV.py runs the LDA classifier 19 times to obtain mean and standard deviation of accuracies by shuffling the training and testing data. The results are printed
6. MLP_all_CV.py runs the multilayer perceptron 20 times to obtain mean and standard deviation of the classification accuracies of by shuffling the training and testing data. The results are printed
7. feature_indices.py is a function that outputs only the features corresponding to the time domain which used min, mean, std, and energy (mean of sum of squares) to compute its features
8. Sequential_FS.py is a function that uses the MLxtend library to perform feature selection
9. LDA_fsd_varyingk.py	varies the number of features to select using LDA  and prints the performance (training accuracy). (fsd - feature selected)
10. LDA_fsd.py uses only the 10 features for classification (LDA_fsd.py calls sequential_FS which produces the same features each time)
11. MLP_fsd.py trains and tests a MLP using only the selected 10 features. 

FOLDER DESCRIPTIONS:
1. Confusion matrices: contains confusion matrices of results and MLP training plots (epoch, val_acc etc)
2. UCI HAR Dataset: The dataset (download the dataset from the link and replace with this)
3. MLP models: Contains MLP models and training history. Names correspond to python file names which created them


