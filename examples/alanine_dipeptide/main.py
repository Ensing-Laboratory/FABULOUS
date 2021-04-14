from __future__ import print_function
import random
import os
import pandas as pd
import mdtraj as md
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split
from fabulous import read_MD, read_CV
from fabulous import build_model
from fabulous import Optimizer

# to force run on CPU:
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# If using GPU, use dynamic memory allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

random.seed(10)

# loading the CV-data
print("Loading CV data...")
fns = glob("./data/AD/CV/CV_data/*")
cv_data = [read_CV(fn) for fn in fns]
cv_data = pd.concat(cv_data, ignore_index=True)
cv_data = cv_data.drop(['time'], axis=1)
# min-max normalization
cv_data = (cv_data - cv_data.min()) / (cv_data.max() - cv_data.min())
print(cv_data.describe())
print(cv_data.columns)

cv_train, cv_test = train_test_split(cv_data, train_size=0.7, shuffle=True)
print(cv_train)
print(cv_test)

# loading the TPS-data
print("Loading TPS data...")
fns = glob("./data/AD/TPS/trjs/*")
ref = md.load("./data/AD/c7ax_input.pdb")
backbone = [atom.index for atom in ref.top.atoms if atom.is_backbone]
md_data = []
for i, fn in enumerate(fns):
    md_data.append(read_MD(fn, ref, backbone))
    if i % 100 == 0:
        print(i)
md_data = pd.concat(md_data, ignore_index=True)
print(md_data.describe())
print(md_data.head())

md_train = md_data.reindex(cv_train.index)
md_test = md_data.reindex(cv_test.index)

# constraints for architecture optimization
param_constraints_input = {
    'network_config': {'n_layers': [2, 3, 4, 5],
                       'batch_size': [5000],
                       'optimizer': ['adam'],
                       'epochs': [500]
                       },

    'layer_config': {'layer_type': ['dense', 'dropout', 'batch_norm', 'batch_norm_dropout'],
                     'n_nodes': [4, 8, 16, 32],
                     'activation': ['relu', 'selu', 'tanh', 'elu', 'sigmoid', 'linear', 'exponential'],
                     'dropout': [0.1, 0.2]
                     },

    'io_config': {'input_shape': [7],
                  'inputs': cv_train.columns.tolist(),
                  'output_shape': [md_train.shape[1]],
                  'outputs': ['custom']
                  }
}

n_gen = 50
run = '1'
if not os.path.exists('./results'):
    os.makedirs('./results')

optimizer = Optimizer(param_constraints_input, build_model, cv_train, log_dir="./results/run_{}".format(run),
                      pop_size=50, train_verbose=0, reject_select_chance=0.05, retain=0.02, early_stopping=True,
                      mutation_rate=0.1, penalty=0.15, parent_frac=0.1, cache=True, train_chance=0.5,
                      test_data=cv_test, val_split=0.3, y_train=md_train, y_test=md_test)

pop = optimizer.create_pop()

gen_scores = []
for i in range(0, n_gen):
    pop = optimizer.evolve(pop)
    current_score = optimizer.get_gen_scores(i)
    optimizer.pickle_gen(i)
    gen_scores.append(current_score)
    optimizer.pickle_all_gen()
