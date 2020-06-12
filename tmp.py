
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics












#############################################################################
import colorsys
from itertools import permutations
def perm_given_index(alist, apermindex):
    alist = alist[:]
    for i in range(len(alist)-1):
        apermindex, j = divmod(apermindex, len(alist)-i)
        alist[i], alist[i+j] = alist[i+j], alist[i]
    return alist
############################################################################

labels_true = [0]*100+[1]*100+[2]*100+[3]*100+[4]*100+[5]*100+[6]*100+[7]*100
labels_true = np.array(labels_true)


data_2d = np.genfromtxt ('./test/AE_reduced_data.txt', delimiter=",").reshape(800,2)
core_samples_mask = np.array(open('./test/AE_mask.txt').readline().split(","))=='True'
labels = np.array(open('./test/AE_labelss.txt').readline().split(","))



def r(d):
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    unique_labels = set(labels)
    fig = plt.figure(figsize=(6.4, 4.8))
    colors = perm_given_index([colorsys.hsv_to_rgb(each,np.random.random()*.3+.7,np.random.random()*.05+.95)+(1,)     for each in np.linspace(0, .8, len(unique_labels))],989458348234234281828545688234238584564638*d)
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = data_2d[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor=tuple(col), markersize=3)
        # plt.scatter(xy[:, 0], xy[:, 1], s=6, c=tuple(col), marker= 'o',
        #          edgecolors='k')
        xy = data_2d[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor=tuple(col), markersize=3)

        from scipy.spatial import ConvexHull
        points = np.vstack((data_2d[class_member_mask & core_samples_mask],data_2d[class_member_mask & ~core_samples_mask])) # 30 random points in 2-D
        hull = ConvexHull(points)

        #import matplotlib.pyplot as plt
        plt.plot(points[:, 0], points[:, 1], 'o')
        for simplex in hull.simplices:  plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
        plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'black', lw=0)
        #plt.plot(points[hull.vertices[0], 0], points[hull.vertices[0], 1], 'ro')
        #plt.show()

        # plt.scatter(xy[:, 0], xy[:, 1], s=6, c=tuple(col), marker= 'o',
        #          edgecolors='k')
    homo = metrics.homogeneity_score(labels_true, labels)
    comp = metrics.completeness_score(labels_true, labels)
    v = metrics.v_measure_score(labels_true, labels)
    rand = metrics.adjusted_rand_score(labels_true, labels)
    infor = metrics.adjusted_mutual_info_score(labels_true, labels)
    silhouette = metrics.silhouette_score(data_2d, labels)
    plt.title('clusters: %d, noise points: %d' % (n_clusters_, n_noise_))
    plt.savefig('rockbell.png', format="png", dpi=300, bbox_inches='tight')
    plt.show()

r(456)