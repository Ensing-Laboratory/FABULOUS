from __future__ import print_function
import random
import os
import yaml
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
def check_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                  "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


# loading the CV-data
def prepare_cv(cv_dir):
    """
    Returns
    -------
    pandas.DataFrame
        single dataframe with all CV information
    """
    print("Loading CV data...")
    fns = glob(os.path.join(cv_dir, "*"))
    cv_data = [read_CV(fn) for fn in fns]
    cv_data = pd.concat(cv_data, ignore_index=True)
    cv_data = cv_data.drop(['time'], axis=1)
    return cv_data


def finish_cv_data(cv_data, train_size=0.7):
    # min-max normalization
    cv_data = (cv_data - cv_data.min()) / (cv_data.max() - cv_data.min())
    print(cv_data.describe())
    print(cv_data.columns)
    cv_train, cv_test = train_test_split(cv_data, train_size=train_size,
                                         shuffle=True)
    print(cv_train)
    print(cv_test)
    return cv_train, cv_test


# loading the TPS-data
def prepare_md(md_dir, ref_file, keep_atoms):
    """
    Returns
    -------
    pandas.DataFrame
        single dataframe with all (aligned) posisiton of selected atoms
    """
    print("Loading TPS data...")
    fns = glob(os.path.join(md_dir, "*"))
    ref = md.load(ref_file)
    if isinstance(keep_atoms, str):
        keep_atoms = ref.topology.select(keep_atoms)
    md_data = []
    for i, fn in enumerate(fns):
        md_data.append(read_MD(fn, ref, keep_atoms))
        if i % 100 == 0:
            print(i)
    md_data = pd.concat(md_data, ignore_index=True)
    return md_data


def finish_md_data(md_data, cv_train, cv_test):
    print(md_data.describe())
    print(md_data.head())

    md_train = md_data.reindex(cv_train.index)
    md_test = md_data.reindex(cv_test.index)
    return md_train, md_test

# constraints for architecture optimization
PARAM_CONSTRAINT_DEFAULTS = {
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
}


def io_config(cv_train, md_train):
    cols = cv_train.columns
    return {
        'input_shape': list(range(len(cols)))[1:],
        'inputs': cols.tolist(),
        'output_shape': [md_train.shape[1]],
        'outputs': ['custom']
    }


def load_param_dict(yaml_file):
    with open(yaml_file, mode='r') as f:
        param_dict = yaml.load(f, Loader=yaml.FullLoader)

    optimizer_params = param_dict.pop('optimizer', {})
    keep_atoms = param_dict.pop('keep_atoms')
    print(optimizer_params)
    return param_dict, optimizer_params, keep_atoms

def run_fabulous(cv_train, cv_test, md_train, md_test,
                 param_constraints_input, optimizer_params, n_gen=50,
                 results_dir="results", label="1"):
    keys = [key for key in PARAM_CONSTRAINT_DEFAULTS
            if key in param_constraints_input]
    # make a copy of the defaults
    param_constraints = dict(PARAM_CONSTRAINT_DEFAULTS)
    for key in keys:
        param_constraints[key].update(param_constraints_input[key])

    param_constraints['io_config'] = io_config(cv_train, md_train)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    log_dir = os.path.join(results_dir, "run_{}".format(label))

    optimizer = Optimizer(
        param_constraints, build_model, cv_train, log_dir=log_dir,
        test_data=cv_test, y_train=md_train, y_test=md_test,
        **optimizer_params
    )

    pop = optimizer.create_pop()

    gen_scores = []
    for i in range(n_gen):
        pop = optimizer.evolve(pop)
        current_score = optimizer.get_gen_scores(i)
        optimizer.pickle_gen(i)
        gen_scores.append(current_score)
        optimizer.pickle_all_gen()


def main(cv_dir, md_dir, ref_file, yaml_file, n_gen, results_dir, label):
    check_gpus()
    param_constraints_input, optimizer_params, keep_atoms = \
            load_param_dict(yaml_file)
    cv_data = prepare_cv(cv_dir)  # this will be different for OPS
    cv_train, cv_test = finish_cv_data(cv_data)
    md_data = prepare_md(md_dir, ref_file, keep_atoms)  # and this
    md_train, md_test = finish_md_data(md_data, cv_train, cv_test)
    run_fabulous(cv_train, cv_test, md_train, md_test,
                 param_constraints_input, optimizer_params, n_gen,
                 results_dir, label)


if __name__ == "__main__":
    # hard coded for now, but better to take from a argparse parser
    cv_dir = "./data/AD/CV/CV_data/"
    md_dir = "./data/AD/TPS/trjs/"
    ref_file = "./data/AD/c7ax_input.pdb"
    yaml_file = "ad_setup.yml"
    n_gen = 50
    results_dir = "results"
    label = "1"
    keep_atoms = "backbone"
    random.seed(10)
    main(cv_dir, md_dir, ref_file, yaml_file, n_gen, results_dir, label)
