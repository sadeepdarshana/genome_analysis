import keras
import numpy as np
from celluloid import Camera
from keras.models import Sequential
from keras.layers.core import Dense
from keras import backend as K
from keras.models import load_model
import os
import random
import vectorizer
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from pathlib import Path
from subprocess import call
from keras.models import load_model
from itertools import groupby

env = {}
_normalized_frequencies_for_file_cache = {}
files = []
model_info = None
data_files_info = None
normalized_frequencies = []
model = None
env['id'] = str(random.randint(1000000,9000000))
env['norm_ex'] = 1
fig = None
photo_index = 0

colors = [
 '#6BB930',
 '#C6C032',
 '#EDD2DB',
 '#A3BCDB',
 '#6D6D8A',
 '#1481A9',
 '#E68A9A',
 '#03DBC4',
 '#8C6E47',
 '#E05C54',
 '#BD871C',
 '#428F57',
 '#9BDA86',
 '#9FDA0D',
 '#3D3161',
 '#ACA450',
 '#E60D69',
 '#818A5A',
 '#188025',
 '#3995F0',
 '#3DEDF6',
 '#6FB0D7',
 '#2FD362',
 '#A90805',
 '#76DE44',
 '#1E858E',
 '#AE6F25',
 '#AF9DF4',
 '#2E3E70',
 '#A86A61']

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

        if i == 'm':
            save_model(epoch_count)
            continue

        caption = "epoch:"+str(epoch_count)+" reload:"+str(reload)#+" norm_ex:"+str(env['norm_ex'])

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

    save_video(env['id'])
    save_model()

def delete_pngs():call("rm *.png", cwd="./figures", shell=True)

def save_model(caption=""):model.save("./models/"+env['id']+" "+str(caption))

def save_video(id):  call("ffmpeg -framerate 1 -i " + str(id) + "_%12d.png " + str(id) + ("_[" + "-".join([str(x) for x in model_info[0]]) + "]_") +str(int(time.time() * 1000))+ ".mp4", cwd="./figures", shell=True)

def plot_files(files,data_files_info,model,model_info,perc = .1,caption = ""):
    global photo_index

    mapping_function = K.function([model.layers[0].input], [model.layers[model_info[1]-1].output])
    for file_index in range(len(files)):
        if(file_index not in _normalized_frequencies_for_file_cache):
            _normalized_frequencies_for_file_cache[file_index] = get_normalized_frequencies_for_file(files[file_index][0:int(len(files[file_index])*perc)])
        points = mapping_function(_normalized_frequencies_for_file_cache[file_index])[0]
        X,Y = [i[0] for i in points], [i[1] for i in points]
        plt.scatter(X, Y, color = data_files_info[file_index]['color'], s=scatter_point_size, alpha=.8)

    Path("./figures").mkdir(parents=True, exist_ok=True)
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig("figures/"+env['id'] +"_"+ str(photo_index).zfill(12))
    photo_index +=1
    plt.show(block=False)

def sum_epochs(epo = None):
    if epo is None:epo =epochs
    return sum(x for x in epo if x != 'm')

def build_data_files_info(dir):
    len_mean = 5000
    len_sd = 1000
    count = 500

    files = os.listdir(dir)
    data_files_info = [dict(file=dir+"/"+files[i], len_mean=len_mean, len_sd=len_sd, count=count, color=colors[i]) for i in range(len(files))]
    return data_files_info
########################################################################################################################

data_files_info = [
                # dict    (file ="data/bat_sars.fasta", len_mean = 5000, len_sd = 1000, count = 1000, color ='r'),
                # dict    (file ="data/covid_19.fasta", len_mean = 5000, len_sd = 1000, count = 1000, color ='g'),
                # dict    (file ="data/sars.fasta",     len_mean = 5000, len_sd = 1000, count = 1000, color ='b'),
                #
                # dict(file="data/escherichia coli.fasta", len_mean=5000, len_sd=1000, count=1000, color='y'),
                # dict(file="data/pseudomonas aeruginosa.fasta", len_mean=5000, len_sd=1000, count=1000, color='k'),
                # dict(file="data/staphylococcus aureus.fasta", len_mean=5000, len_sd=1000, count=1000, color='c'),
                #
                # dict(file="data/Chimeric_dengue_-virus.fasta", len_mean=5000, len_sd=1000, count=1000, color='#777777'),
                # dict(file="data/ebola_1976.fasta", len_mean=5000, len_sd=1000, count=1000, color='m'),
]

model_info = (
                #[32,16,6,2,6,16,32],
                [32,16,2,16,32],      # layers
                2               # autoencoder output layer index
)

scatter_point_size = 8
########################################################################################################################


data_files_info = build_data_files_info("data/21s")

reload_n_process_data(data_files_info)

model = build_model(model_info)


env['norm_ex'] = 1

epochs = [0,4,10,6,'m']+[10]*8+[10]*30+[250]*50+['m']+[1000]*30+['m']+[1000]*20#+[10]*10+[4000]*3+['m']+[10]*10+['m']#+[50]*200


#auto_train(model_info, data_files_info, epochs=epochs,current_epoch=0, plot=True, new_model=True, reload=False, peek=True,plot_perc=.2)
