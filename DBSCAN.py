import pandas as pd
from sklearn import datasets
# from jqm_cvi import base
import numpy as np

# loading the dataset
# X = datasets.load_iris()
# df = pd.DataFrame(X.data)
#
# print(df)
# # K-Means
# from sklearn import cluster
#
# k_means = cluster.KMeans(n_clusters=3)
# k_means.fit(df)  # K-means training
# y_pred = k_means.predict(df)
#
# print(y_pred)
#
# # We store the K-means results in a dataframe
# pred = pd.DataFrame(y_pred)
# pred.columns = ['Species']
# print(pred)
#
# # we merge this dataframe with df
# prediction = pd.concat([df, pred], axis=1)
# print(prediction)
#
# # We store the clusters
# clus0 = prediction.loc[prediction.Species == 0]
# clus1 = prediction.loc[prediction.Species == 1]
# clus2 = prediction.loc[prediction.Species == 2]
# cluster_list = [clus0.values, clus1.values, clus2.values]
#
# print(cluster_list)
#
# # print(base.dunn(cluster_list))

# class dunner():
def delta(ck, cl):
    values = np.ones([len(ck), len(cl)]) * 10000

    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i] - cl[j])

    return np.min(values)


def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])

    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i] - ci[j])

    return np.max(values)

def dunn(k_list):
    """ Dunn index [CVI]

    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    """
    deltas = np.ones([len(k_list), len(k_list)]) * 1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))

    for k in l_range:
        for l in (l_range[0:k] + l_range[k + 1:]):
            deltas[k, l] = delta(k_list[k], k_list[l])

        big_deltas[k] = big_delta(k_list[k])

    di = np.min(deltas) / np.max(big_deltas)
    return di

# print(dunn(k_list=cluster_list))

# def calc_dunn_index()