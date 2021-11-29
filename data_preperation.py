"""
This file prepares total dataFrame for train and test of models
"""


import numpy as np
import pandas as pd

'''
This function creates total dataFrame according to original txt files
@ positive_data_original    The ndarray containing only samples with true label
@ negative_data_original    The ndarray containing only samples with false label
@ num_cameras               Number of cameras, used to create some features' names
@ shuffle                   If the returning dataFrame is shuffled
'''
def prepare_data(positive_data_original, negative_data_original, num_cameras, shuffle=True):
    column_names = ["camA_id", "camB_id",
                    "xca", "yca", "xcb", "ycb",
                    "area_a", "area_b",
                    "r_a", "r_b",
                    "rou_a", "rou_b",
                    "peri_a", "peri_b",
                    "I_a", "I_b", "MI_a", "MI_b", "MEI_a", "MEI_b", "MI_SHI_xa", "MI_SHI_ya", "MI_SHI_xb",
                    "MI_SHI_yb",
                    "epipolar_error",
                    "X", "Y", "Z",
                    "xca_repo", "yca_repo", "xcb_repo", "ycb_repo",
                    "repo_error_a", "repo_error_b", "repo_error_total", "repo_SHI_xa", "repo_SHI_ya",
                    "repo_SHI_xb", "repo_SHI_yb"
                    ]

    # add the rest of feature names, there features are related to pixel intensities
    column_names_continue = [["RI_" + str(i), "RMI_" + str(i), "RMEI_" + str(i)] for i in range(num_cameras)]
    flatten = lambda t: [item for sublist in t for item in sublist]
    column_names_continue = flatten(column_names_continue)

    for item in column_names_continue:
        column_names.append(item)

    # add the label column
    column_names.append("label")

    # construct and shuffle the training data numpy array
    positive_data = positive_data_original
    positive_label = np.ones((positive_data.shape[0], 1))
    positive_data = np.c_[positive_data, positive_label]

    negative_data = negative_data_original
    negative_label = np.zeros((negative_data.shape[0], 1))
    negative_data = np.c_[negative_data, negative_label]

    total_data = np.vstack([positive_data, negative_data])

    if shuffle:
        np.random.shuffle(total_data)

    # construct the pandas dataFrame from the numpy array
    data_frame = pd.DataFrame(data=total_data, columns=column_names)

    return data_frame