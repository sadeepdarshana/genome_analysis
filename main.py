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
import seaborn as sns; sns.set()
import pandas as pd

env = {}
_normalized_frequencies_for_file_cache = {}
files = []
model_info = None
data_files_info = None
normalized_frequencies = []
model = None
env['id'] = str(random.randint(1000000,9000000))
env['norm_ex'] = 1
photo_index = 0


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
    layers_info = model_info
    model = Sequential()
    model.add(Dense(layers_info[1], input_dim=layers_info[0],activation = 'sigmoid'))
    for i in range(2,len(layers_info)):
        model.add(Dense(layers_info[i], activation = 'sigmoid'))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam())
    return model

def train(model,normalize_frequencies,epochs):
    model.fit(normalize_frequencies,normalize_frequencies,epochs=epochs)

def auto_train(model_info, data_files_info, epochs, plot=True, new_model=False, reload=False,peek = False, current_epoch = 0, use_model = "",plot_perc = .1):
    global model
    global camera
    global  photo_index
    photo_index = 0

    reload_n_process_data(data_files_info)

    env['id'] = str(int(time.time() * 1000))

    if (new_model):
        model = build_model(model_info)
    epoch_count = current_epoch

    if use_model:
        model = load_model("./models/"+use_model)
        env['id'] = use_model + "_" + str(int(time.time() * 1000))

    for i in epochs:

        if i == 'm':
            save_model(epoch_count,photo_index)
            continue

        caption =  ("[" + "-".join([str(x) for x in model_info]) + "]   ")+"epoch: "+str(epoch_count)

        if peek and  reload:
            plot_files(files, model, model_info, plot_perc, ("old_data " if reload else "") + caption)
        if reload:
            reload_n_process_data(data_files_info) # files, normalized_frequencies assigned
        if plot:
            plot_files(files, model, model_info, plot_perc, ("new_data " if reload else "") + caption)
        if i != 0:
            train(model, normalized_frequencies, i)
            epoch_count+=i

    if plot:
        caption = ("[" + "-".join([str(x) for x in model_info]) + "]   ") + "epoch: " + str(epoch_count)
        plot_files(files, model, model_info, plot_perc, caption)

    save_video(env['id'],epoch_count)
    save_model(epoch_count,photo_index)
    #delete_pngs()

def delete_pngs():call("rm *.png", cwd="./figures", shell=True)


def pd_vstack(dfs):
    big_list = None
    for i in dfs:
        if big_list is None: big_list = i
        else:big_list = big_list.append(i, ignore_index=True)
    return big_list

def save_model(epochs="", next_photo_index=""):model.save("./models/" + env['id'] + (("_" + str(epochs).zfill(12)) if epochs else "") + (("_" + str(next_photo_index).zfill(12)) if next_photo_index else ""))

def save_video(id,epochs=""):  call("ffmpeg -framerate 1 -i " + str(id) + "_%12d.png " + str(id) + ("_[" + "-".join([str(x) for x in model_info]) + "]_") +str(int(time.time() * 1000))+"_"+str(epochs)+ ".mp4", cwd="./figures", shell=True)

def plot_files(files, model, model_info, perc = .1, caption =""):
    global photo_index

    mapping_function = K.function([model.layers[0].input], [model.layers[len(model_info)//2-1].output])

    plot_dfs_list = []

    for file_index in range(len(files)):
        if(file_index not in _normalized_frequencies_for_file_cache):
            _normalized_frequencies_for_file_cache[file_index] = get_normalized_frequencies_for_file(files[file_index][0:int(len(files[file_index])*perc)])
        points = mapping_function(_normalized_frequencies_for_file_cache[file_index])[0]
        X,Y,F = [i[0] for i in points], [i[1] for i in points], get_file_1st_names()[file_index]

        plot_dfs_list.append(pd.DataFrame(         {'X': X,      'Y': Y,         'Sequence Name': F     }))

    Path("./figures").mkdir(parents=True, exist_ok=True)
    plt.figure()
    df = pd_vstack(plot_dfs_list).sample(frac=1).reset_index(drop=True)
    ax = sns.scatterplot(x='X', y='Y', hue='Sequence Name', data=df, edgecolor="none",alpha=1,hue_order=get_file_1st_names())
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.grid(False)
    plt.title(caption)
    plt.savefig("figures/"+env['id'] +"_"+ str(photo_index).zfill(12), bbox_inches='tight')
    photo_index +=1
    plt.show()

def sum_epochs(epo = None): return sum(x for x in (epochs if epo is None else epo) if x != 'm')

def get_file_1st_names():return [x['file'].split("/")[-1].split(".")[0] for x in data_files_info]

def build_data_files_info(dir):
    len_mean = 5000
    len_sd = 1000
    count = 500

    files = os.listdir(dir)
    data_files_info = [dict(file=dir+"/"+files[i], len_mean=len_mean, len_sd=len_sd, count=count) for i in range(len(files))]
    return data_files_info
########################################################################################################################

model_info = (
                #[32,16,6,2,6,16,32]
                [32,16,2,16,32]
                #[32,2,32]
)


data_files_info = build_data_files_info("data/the_selected_8")


epochs = [20,'m']*10+[50]*10+[100]*10+[200,'m']*10+[500,'m']*10+['m']+[1000,'m']*20


auto_train(model_info, data_files_info, epochs=epochs,current_epoch=0, new_model=True)
