"""
This is the main file to run train and test of models.

The original data is as follows:
----------------------------------
       f0   f1  ......  f63  Label
Row 0  50   100 ......   20    0
Row 1  ......                  1
.
.
.
Row N ......                   0
----------------------------------
The original data will be scaled and transformed by PCA for training and testing.
There total data contains 80000 rows with a class weight (True: False = 1:3.5).

Four different machine learning models are selected and tested using the default models
provided in sklearn except the unbalanced class weight mentioned above:
1. Logistic Regression
2. Decision Tree
3. Neural Network (MLP)
4. Support Vector Machine

Both the accuracy and prediction time of four models were tested.
"""

from data_preperation import *
from train_functions import *


if __name__ == '__main__':
    num_pca_components = 10  # the number of principal components
    num_cameras = 8  # the number of total cameras

    positive_data_original = np.loadtxt(
        './data/vptv/positive_data_partial.txt')  # the training data with positive label
    negative_data_original = np.loadtxt(
        './data/vptv/negative_data_partial.txt')  # the training data with negative label
    shuffle = True

    # get the total dataFrame (train + test) from txt files
    data_frame = prepare_data(positive_data_original, negative_data_original, num_cameras, shuffle)

    # perform train and test of models
    train_test(data_frame, num_pca_components)