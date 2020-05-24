import keras
import numpy as np
from celluloid import Camera
from keras.models import Sequential
from keras.layers.core import Dense
from keras import backend as K
from keras.models import load_model
import random
import vectorizer
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from pathlib import Path
from subprocess import call
from keras.models import load_model
from sklearn.cluster import DBSCAN
from sklearn import metrics
import pandas as pd

env = {}
_normalized_frequencies_for_file_cache = {}
picolo = {}
files = []
model_info = None
data_files_info = None
normalized_frequencies = []
model = None
env['id'] = str(random.randint(1000000,9000000))
env['norm_ex'] = 1
fig = None
photo_index = 0
X=[]
Y=[]

labels_true = [0]*100+[1]*100+[2]*100+[3]*100+[4]*100+[5]*100+[6]*100+[7]*100
labels_true = np.array(labels_true)


########################################################################################################################
def load_files(data_files_info):
    data_files_count = len(data_files_info)
    last_data_point_id = 0
    files = []

    for i in range(data_files_count):
        data_file = data_files_info[i]
        raw_data_points = vectorizer.split_n_count_fasta(
            data_file['file'],
            data_file['len_mean'],
            data_file['len_sd'],
            data_file['count']
        )
        print("sad")
        for c in range(len(raw_data_points)):
            raw_data_points[c] = (last_data_point_id,i) + raw_data_points[c]
            last_data_point_id += 1

        files.append(raw_data_points)

    return files

def normalize_over_axis1(arr2d):
    for i in range(arr2d.shape[0]):
        hori_sum = sum(arr2d[i]**env['norm_ex'])
        if hori_sum!= 0 :arr2d[i] = (arr2d[i]**env['norm_ex'])/hori_sum

def get_normalized_frequencies_for_file(file):
    freqs = []
    for raw_data_point in file:
        freqs.append(raw_data_point[4])
    np_freqs =  np.array(freqs).astype(np.float32)
    normalize_over_axis1(np_freqs)
    return np_freqs

def get_normalized_frequencies_for_files(files):
    tpl = tuple([get_normalized_frequencies_for_file(file) for file in files])
    return np.concatenate(tpl, axis=0)

def reload_n_process_data(data_files_info):
    global files
    global normalized_frequencies
    global _normalized_frequencies_for_file_cache

    files = load_files(data_files_info)
    normalized_frequencies = get_normalized_frequencies_for_files(files)
    np.random.shuffle(normalized_frequencies)
    _normalized_frequencies_for_file_cache = {}

def build_model(model_info):
    layers_info = model_info[0]
    model = Sequential()
    model.add(Dense(layers_info[1], input_dim=layers_info[0],activation = 'sigmoid'))
    for i in range(2,len(layers_info)):
        model.add(Dense(layers_info[i], activation = 'sigmoid'))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam())
    return model

def train(model,normalize_frequencies,epochs):
    model.fit(normalize_frequencies,normalize_frequencies,epochs=epochs)

def auto_train(model_info, data_files_info, epochs, plot, new_model=False, reload=True,peek = False, current_epoch = 0, use_model = "",plot_perc = .1):
    global model
    global fig
    global camera

    global  photo_index
    photo_index = 0

    env['id'] = str(int(time.time() * 1000))

    if (new_model):
        model = build_model(model_info)
    epoch_count = current_epoch

    if use_model:
        model = load_model("./models/"+use_model)
        env['id'] = use_model + "_" + str(int(time.time() * 1000))

    fig = plt.figure()
    for i in epochs:

        caption = "epoch:"+str(epoch_count)+" reload:"+str(reload)+" norm_ex:"+str(env['norm_ex'])

        if peek and  reload:
            plot_files(files,data_files_info,model,model_info,plot_perc,"old_data "+caption)
        if reload:
            reload_n_process_data(data_files_info) # files, normalized_frequencies assigned
        if plot:
            plot_files(files,data_files_info,model,model_info,plot_perc,"new_data "+caption)
        if(i != 0):
            train(model, normalized_frequencies, i)
            epoch_count+=i

    if plot:
        caption = "epoch:" + str(epoch_count) + " reload:" + str(reload) + " norm_ex:" + str(env['norm_ex'])
        plot_files(files,data_files_info,model,model_info,plot_perc,caption)

    call("ffmpeg -framerate 1 -i "+env['id']+"_%12d.png "+env['id']+".mp4", cwd="./figures", shell=True)
    #call("rm *.png", cwd="./figures", shell=True)

    model.save("./models/"+env['id'])


def plot_files(files,data_files_info,model,model_info,perc = .1,caption = "", loc=""):
    global photo_index
    global picolo
    global X, Y

    mapping_function = K.function([model.layers[0].input], [model.layers[model_info[1]-1].output])
    for file_index in range(len(files)):
        if(file_index not in _normalized_frequencies_for_file_cache):
            _normalized_frequencies_for_file_cache[file_index] = get_normalized_frequencies_for_file(files[file_index][0:int(len(files[file_index])*perc)])
        points = mapping_function(_normalized_frequencies_for_file_cache[file_index])[0]
        X,Y = [i[0] for i in points], [i[1] for i in points]
        plt.scatter(X, Y, color = data_files_info[file_index]['color'], s=scatter_point_size, alpha=.8)
    picolo = _normalized_frequencies_for_file_cache

    Path(loc).mkdir(parents=True, exist_ok=True)
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
    # plt.savefig(loc+env['id'] +"_"+ str(photo_index).zfill(12))
    photo_index +=1
    # plt.show()

def retract_input_freqs():
    input_freqs = []

    for i in range(len(picolo)):
        # print(picolo[i])
        for j in picolo[i]:
            input_freqs.append(j.tolist())
    input_freqs = np.array(input_freqs)

    return input_freqs


def AE_reduce(model, model_info):

    input_freqs = retract_input_freqs()
    mapping_function = K.function([model.layers[0].input], [model.layers[model_info[1] - 1].output])
    data = mapping_function(input_freqs)

    return data

def run_DBSCAN(data, eps):

    db = DBSCAN(eps=eps, min_samples=10).fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # print(core_samples_mask)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # print(db.core_sample_indices_)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(data, labels))

    return data, labels, core_samples_mask, n_clusters_

def plot_DBSCAN(data_2d, labels, core_samples_mask):

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = data_2d[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = data_2d[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    homo = metrics.homogeneity_score(labels_true, labels)
    comp = metrics.completeness_score(labels_true, labels)
    v = metrics.v_measure_score(labels_true, labels)
    rand = metrics.adjusted_rand_score(labels_true, labels)
    infor = metrics.adjusted_mutual_info_score(labels_true, labels)
    silhouette = metrics.silhouette_score(data_2d, labels)

    plt.title('clusters: %d, noise points: %d, homogeneity: %d, completeness: %d, v: %d, rand: %d, infor: %d, silhouette: %d' % (n_clusters_, n_noise_ ,homo,comp, v, rand, infor, silhouette))
    plt.savefig('figures/myfig/1_AE_eboladengue_eps=0.05.png', format="png", dpi=300, bbox_inches='tight')
    plt.show()

########################################################################################################################

data_files_info = [
                dict    (file ="data/bat_sars.fasta", len_mean = 5000, len_sd = 1000, count = 1000, color ='r'),
                dict    (file ="data/covid_19.fasta", len_mean = 5000, len_sd = 1000, count = 1000, color ='g'),
                dict    (file ="data/sars.fasta",     len_mean = 5000, len_sd = 1000, count = 1000, color ='b'),

                dict(file="data/escherichia coli.fasta", len_mean=5000, len_sd=1000, count=1000, color='y'),
                dict(file="data/pseudomonas aeruginosa.fasta", len_mean=5000, len_sd=1000, count=1000, color='k'),
                dict(file="data/staphylococcus aureus.fasta", len_mean=5000, len_sd=1000, count=1000, color='c'),

                dict(file="data/Chimeric_dengue_-virus.fasta", len_mean=5000, len_sd=1000, count=1000, color='#777777'),
                dict(file="data/ebola_1976.fasta", len_mean=5000, len_sd=1000, count=1000, color='m'),
]

model_info = (
                #[32,16,6,2,6,16,32],
                # [128,32,2,32,128],      # layers
                [32,16,2,16,32],
                2               # autoencoder output layer index
)

scatter_point_size = 8
########################################################################################################################


# g_files = load_files(data_files_info)

reload_n_process_data(data_files_info)

my_model = load_model("./models/1589714612647")
plot_files(files,data_files_info, my_model, model_info, caption="my_model_plot", loc="./figures/myfig/")
print("chakabakabawa")
print(np.shape(retract_input_freqs()))

data = AE_reduce(my_model,model_info)

data_2d,labelss,mask, clusters= run_DBSCAN(data[0], 0.05)


# plot_DBSCAN(data_2d,labelss,mask)










#
# from sklearn.metrics import davies_bouldin_score
# score = davies_bouldin_score(data_2d, labelss)
# print(score)


# reload_n_process_data(data_files_info)
#
# model = build_model(model_info)
#
#
# env['norm_ex'] = 1
# epochs = [0,4,10,6]+[10]*8+[10]*30+[250]*50#+[1000]*200+[10]*10+[4000]*25+[10]*10#+[50]*200
# auto_train(model_info, data_files_info, epochs=epochs,current_epoch=0, plot=True, new_model=True, reload=False, peek=True)
#
#
# auto_train(model_info, data_files_info, epochs=epochs,current_epoch=0, plot=True, new_model=True, reload=False, peek=True)