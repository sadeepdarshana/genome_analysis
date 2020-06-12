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
from decimal import Decimal
import glob


def build_model(model_info):
    layers_info = model_info
    model = Sequential()
    model.add(Dense(layers_info[1], input_dim=layers_info[0],activation = 'sigmoid'))
    for i in range(2,len(layers_info)):
        model.add(Dense(layers_info[i], activation = 'sigmoid'))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam())
    return model


model = build_model([32,16,2,16,32])


# the only parameters you need to edit----------------------------------------------------------------------------------
filename = "data/3mers"
epochs = 1000
# ----------------------------------------------------------------------------------------------------------------------


r=np.loadtxt(filename)
model.fit(r,r,epochs=epochs)
model.save("model"+str(int(time.time() * 1000)))