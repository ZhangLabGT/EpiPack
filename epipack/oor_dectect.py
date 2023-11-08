import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def gaussian_init(train_data, label):

    init_mu = []
    init_cov_inv = []
    for i in label:
        class_i_data = train_data[np.where(label == i)]
        class_i_mean = np.mean(class_i_data, axis=0, keepdims=True)
        class_i_norm = class_i_data - class_i_mean
        class_i_cov = np.matmul(class_i_norm, class_i_norm.T)/(class_i_norm.shape[0]-1)

        init_mu.append(class_i_mean)
        init_cov_inv.append(np.linalg.inv(class_i_cov))

    return init_mu, init_cov_inv


def maha_dist(train_data, label, query_data, confidence_threshold):

    ## transform label(chr) to label(int)
    le = LabelEncoder()
    y=le.fit_transform(label)

    label_set = np.unique(y)

    init_mu, init_cov_inv = gaussian_init(train_data, label_set)

    maha_score = []

    for i in len(label_set):
        query_norm = query_data - init_mu[i]
        

    


    

