import keras
import numpy as np
from celluloid import Camera
from keras.models import Sequential
from keras.layers.core import Dense
from keras import backend as K
from keras.models import load_model
import random
import vectorizer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from subprocess import call

env = {}
_normalized_frequencies_for_file_cache = {}
files = []
model_info = None
data_files_info = None
normalized_frequencies = []
model = None
env['id'] = str(random.randint(1000000,9000000))
fig = None
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
        hori_sum = sum(arr2d[i])
        if hori_sum!= 0 :arr2d[i] = arr2d[i]/hori_sum

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

def auto_train(model_info, data_files_info, epochs, plot, new_model=False, reload=True,peek = False, current_epoch = 0):
    global model
    global fig
    global camera

    global  photo_index
    photo_index = 0

    if (new_model):
        model = build_model(model_info)
    epoch_count = current_epoch
    fig = plt.figure()
    for i in epochs:
        if peek:
            plot_files(files,data_files_info,model,model_info,.1,"old_data "+str(epoch_count))
        if reload:
            reload_n_process_data(data_files_info) # files, normalized_frequencies assigned
        if plot:
            plot_files(files,data_files_info,model,model_info,.1,"new_data "+str(epoch_count))
        if(i != 0):
            train(model, normalized_frequencies, i)
            epoch_count+=i

    if plot:
        plot_files(files,data_files_info,model,model_info,.1,str(epoch_count))


    call("ffmpeg -framerate 1 -i "+env['id']+"_%12d.png "+env['id']+".mp4", cwd="./figures", shell=True)


def plot_files(files,data_files_info,model,model_info,perc = .1,caption = ""):
    global photo_index

    mapping_function = K.function([model.layers[0].input], [model.layers[model_info[1]-1].output])
    for file_index in range(len(files)):
        if(file_index not in _normalized_frequencies_for_file_cache):
            _normalized_frequencies_for_file_cache[file_index] = get_normalized_frequencies_for_file(files[file_index][0:int(len(files[file_index])*perc)])
        points = mapping_function(_normalized_frequencies_for_file_cache[file_index])[0]
        X,Y = [i[0] for i in points], [i[1] for i in points]
        plt.scatter(X, Y, color = data_files_info[file_index]['color'], s=scatter_point_size, alpha=.8)


    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig("figures/"+env['id'] +"_"+ str(photo_index).zfill(12))
    photo_index +=1
    plt.show()
########################################################################################################################

data_files_info = [
                dict    (file ="data/bat_sars.fasta", len_mean = 5000, len_sd = 1000, count = 1849, color ='r'),
                dict    (file ="data/covid_19.fasta", len_mean = 5000, len_sd = 1000, count = 1000, color ='g'),
                dict    (file ="data/sars.fasta",     len_mean = 5000, len_sd = 1000, count = 1000, color ='b'),
]

model_info = (
                [32,2,32],      # layers
                1               # autoencoder output layer index
)

scatter_point_size = 8
########################################################################################################################

reload_n_process_data(data_files_info)

model = build_model(model_info)

#epochs = [0,0,1,10,100,1,1,1,500,1,1,1,1,11,1000,1,1,1,1,1001,1500,10000,1,50000,1,1,1,1,1,1,1,1,1,50000,1,50000,50000,50000,50000,50000]


env['id']  = str(int(time.time()*1000))
epochs = [0,4,10,6]+[10]*8+[10]*90+[25]*200
auto_train(model_info, data_files_info, epochs=epochs, plot=True, new_model=True, reload=False, peek=True)


env['id']  = str(int(time.time()*1000))
epochs = [0,4,10,6]+[10]*8+[10]*90+[25]*200
auto_train(model_info, data_files_info, epochs=epochs, plot=True, new_model=True, reload=True, peek=True)


env['id']  = str(int(time.time()*1000))
epochs = [0,4,10,6]+[10]*8+[10]*90+[25]*200
auto_train(model_info, data_files_info, epochs=epochs, plot=True, new_model=True, reload=True, peek=True)


env['id']  = str(int(time.time()*1000))
epochs = [0,4,10,6]+[10]*8+[10]*90+[25]*200
auto_train(model_info, data_files_info, epochs=epochs, plot=True, new_model=True, reload=True, peek=True)


env['id']  = str(int(time.time()*1000))
epochs = [0,4,10,6]+[10]*8+[10]*90+[25]*200
auto_train(model_info, data_files_info, epochs=epochs, plot=True, new_model=True, reload=True, peek=True)


env['id']  = str(int(time.time()*1000))
epochs = [0,4,10,6]+[10]*8+[10]*90+[25]*200
auto_train(model_info, data_files_info, epochs=epochs, plot=True, new_model=True, reload=True, peek=True)

