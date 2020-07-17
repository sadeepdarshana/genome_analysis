import json

import numpy as np
import pyVectorizer
import pandas as pd

def process(input_path, k, output_path):
    vectors = np.array(pyVectorizer.vectorize_file(input_path, k)).astype(np.float32)
    acgt = pyVectorizer.count_acgt_file(input_path)

    vectorSum = np.zeros(len(pyVectorizer.all_kmers(k)),dtype=np.int64)

    for i in vectors:
        vectorSum += np.array(i).astype(np.int64)

    with open(output_path+"_freqs.json", 'w') as outfile:
        json.dump((k,vectorSum.tolist()), outfile)

    with open(output_path+"_acgt.json", 'w') as outfile:
        json.dump(acgt, outfile)
