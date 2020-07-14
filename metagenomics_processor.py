import keras
from keras import Sequential
from keras.layers import Dense
import numpy as np
import pyVectorizer
from keras import backend as K
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


def normalize_over_axis1(arr2d):
    for i in range(arr2d.shape[0]):
        hori_sum = sum(arr2d[i])
        if hori_sum!= 0 :arr2d[i] = (arr2d[i])/hori_sum


def build_model(layers_sizes, activation ):
    model = Sequential()
    model.add(Dense(layers_sizes[1], input_dim=layers_sizes[0],activation = activation))
    for i in range(2,len(layers_sizes)):
        model.add(Dense(layers_sizes[i], activation = activation))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam())
    return model

def train(model,epochs,vectors):
    model.fit(vectors,vectors,epochs=epochs)

def get_points(model, vectors):
    middle_layer=model.layers[int(len(model.layers) / 2 - 1)]
    input_layer=model.layers[0]
    mapping_function = K.function([input_layer.input], [middle_layer.output])
    points = mapping_function(vectors)[0]
    return points

def process(input_path, k,epochs, layers_sizes, output_path = None):
    model = build_model(layers_sizes,'sigmoid')
    vectors = np.array(pyVectorizer.vectorize_file(input_path, k)).astype(np.float32)
    normalize_over_axis1(vectors)
    train(model,epochs,vectors)
    r=get_points(model,vectors)
    dataset = pd.DataFrame({'x': r[:, 0], 'y': r[:, 1]}) if (r.shape[1] == 2) else pd.DataFrame({'x': r[:, 0], 'y': r[:, 1], 'z': r[:, 2]})

    # ax1 = dataset.plot.scatter(x='x', y='y', c='DarkBlue')
    # plt.show()

    if output_path:
        dataset.to_csv(output_path)

#process("./data/AAGA01.1.fsa_nt",3, [32,2,32], "./out.csv")