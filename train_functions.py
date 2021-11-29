"""
This file contains the training and testing of models.

The original total dataFrame is passed to train_test function. In this function,
    the total dataFrame is split into train and test data sets.

The train_test_helper is the function that performs the actual training and testing.
In train_test_helper, a PCA transformation is first performed to reduce the dimension of features.
"""


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import time
from pickle import dump

'''
helper function of train_test
@ train             The ndarray of train data
@ train_label       The labels of train
@ test              The ndarray of test data
@ test_label        The labels of test
@ n_components      The number of principal components
@ fraction_x_train  The fraction used for training, used when the train array is too large
'''
def train_test_helper(train, train_label,
                      test, test_label,
                      n_components,
                      fraction_x_train=1):
    number_rows = int(fraction_x_train * len(train))    # calculate number of rows of training data based on fraction
    pca = PCA(n_components=n_components)                # pca object

    train[np.isnan(train)] = 0                          # eliminate nan in train as 0
    test[np.isnan(test)] = 0                            # eliminate nan in test as 0

    pca.fit(train)                                      # pca fit of train
    dump(pca, open('./model_parameters/pca.pkl', 'wb')) # save pca parameters

    train = pca.transform(train)                        # pca transformation of train
    test = pca.transform(test)                          # pca transformation of test

    # initiate four different classification objects
    log_clf = LogisticRegression(random_state=0, class_weight={0: 3.5, 1: 1}, C=1)  # logistic regression
    tree_clf = DecisionTreeClassifier(class_weight={0: 3.5, 1: 1})                  # decision tree
    nn_clf = MLPClassifier(random_state=1, max_iter=500)                            # neural network
    svm_clf = SVC(gamma='scale', class_weight={0: 3.5, 1: 1})                       # support vector machine

    train = train[:number_rows, :]                      # select part of train data
    train_label = train_label[:number_rows]             # select part of train labels

    # fit four classifiers using train and train_label
    log_clf.fit(train, train_label.ravel())
    tree_clf.fit(train, train_label.ravel())
    nn_clf.fit(train, train_label.ravel())
    svm_clf.fit(train, train_label.ravel())

    # save models
    dump(log_clf, open('./model_parameters/log_clf.pkl', 'wb'))
    dump(tree_clf, open('./model_parameters/tree_clf.pkl', 'wb'))
    dump(nn_clf, open('./model_parameters/nn_clf.pkl', 'wb'))
    dump(svm_clf, open('./model_parameters/svm_clf.pkl', 'wb'))

    # following blocks calculate the accuracies and prediction time of different classifiers
    current_time = time.perf_counter()
    test_predicted_labels_log = np.array(log_clf.predict(test)).reshape(-1, 1)      # logistic regression
    print(f"log prediction in {time.perf_counter() - current_time:0.4f} seconds")
    print("Accuracy of LOG test dataset is: {}".format(accuracy_score(test_label, test_predicted_labels_log)))

    current_time = time.perf_counter()
    test_predicted_labels_tree = np.array(tree_clf.predict(test)).reshape(-1, 1)
    print(f"tree prediction in {time.perf_counter() - current_time:0.4f} seconds")  # decision tree
    print("Accuracy of TREE test dataset is: {}".format(accuracy_score(test_label, test_predicted_labels_tree)))

    current_time = time.perf_counter()
    test_predicted_labels_nn = np.array(nn_clf.predict(test)).reshape(-1, 1)
    print(f"nn prediction in {time.perf_counter() - current_time:0.4f} seconds")    # neural network
    print("Accuracy of NN test dataset is: {}".format(accuracy_score(test_label, test_predicted_labels_nn)))

    current_time = time.perf_counter()
    test_predicted_labels_svm = np.array(svm_clf.predict(test)).reshape(-1, 1)
    print(f"svm prediction in {time.perf_counter() - current_time:0.4f} seconds")   # support vector machine
    print("Accuracy of SVM test dataset is: {}".format(accuracy_score(test_label, test_predicted_labels_svm)))


'''
Train and test models
@ df                    The total dataFrame before splitting to train and test data sets
@ num_pca_components    Number of principal components
'''
def train_test(df, num_pca_components):
    total_data_df = df.drop(df.columns[[-1]], axis=1)       # drop the "label" column
    total_label_df = df["label"]                            # select the label column as different dataFrame

    # split the total dataFrame as train data set (X_train, X_test) and test data set (y_train, y_test)
    X_train, X_test, y_train, y_test = train_test_split(total_data_df, total_label_df, test_size=0.2, random_state=42)

    scaler = StandardScaler()                                        # scaler for normalization
    scaler.fit(X_train)                                              # fit the scaler using X_train
    dump(scaler, open('./model_parameters/scaler.pkl', 'wb'))        # save the scale using pickle

    X_train_scaled = scaler.transform(X_train)      # normalize X_train using scaler
    X_test_scaled = scaler.transform(X_test)        # normalize X_test using scaler

    train_test_helper(X_train_scaled, y_train, X_test_scaled, y_test, num_pca_components)   # call helper function
