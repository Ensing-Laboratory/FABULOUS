from __future__ import print_function

import random
import os
import re
import time
import pickle
import itertools
from datetime import datetime
from copy import deepcopy
from dict_hash import sha256

from fabulous.keras_utilities import *

from tensorflow.keras.callbacks import TensorBoard, CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.backend import clear_session

class Optimizer:
    """
    Class for optimizing the hyperparameters of a keras model using a genetic algorithm.
        
    Data is supplied in the form of a pandas dataframe. 
    The dataframe for train/val/test should contain both the inputs (features) and the outputs (labels):
    the column names of the dataframe should match the specified in/outputs in the param_constraints.
    So, if 'psi' and 'phi' are the inputs specified in param_constraints, 
    dataframe[['psi', 'psi']] is expected to return correctly (same for outputs).
    In the case that no test data is suplied, either validation data or a validation split above 0 must be supplied.
    
    The logs are kept in the specified log_dir and have the following structure:
        csv_logs
        tensorboard
        pickles
        checkpoints
    
    Example:
        
    
    Args:
        param_constraints (dict): dictionary containing all parameter contraints.
        build_model (function): function used to build the model; it should return a compiled keras model.
        train_data (pandas dataframe): data used to train the network; configured as described above.
        val_data (pandas dataframe, optional): data used for validation of the network training.
        val_split (float, optional): split from training data to be used for validation; 
            ignored when val_data is given.
        test_data (pandas dataframe, optional): data used to test the performance of the network; 
            if none is given, the last validation score is used instead.
        retain (float): the size of the parent pool; the number of networks that are allowed to
            directly advance to the next generation.
        n_children_per_couple (int): the number of children that are created per couple.
        reject_select_chance (float): the chance that a rejected network is still able to go to advance
            to the next generation.
        mutation_rate (float): the rate of mutation; the chance a child is chosen to be mutated.
        pop_size (int): the total population size; the number of networks in a generation.
        log_dir (str): the path to the log directory.
        log_path_params (list of str): the parameters to be used in the naming of the logs.
        train_verbose (bool): whether to train verbose.
        individual (bool): whether all layers in a network are identical or not.
        early_stopping (bool): wheter to apply the early stopping callback from the keras API.
        early_stopping_patience (int, optional): the number of epochs a network is allowed to make no progress.
        tensorboard (bool): wheter to apply the tensorboard callback from the keras API.
        model_checkpoint (bool): wheter to apply the checkpoint callback from the keras API; 
            saves the model parameters.
        csv_logger (bool): wheter to apply the csv_logger callback from the keras API.
        user_callbacks (list of callbacks): list of user specified callbacks.
        y_train (dataframe): dataframe containing the y values to be trained on.
        y_val (dataframe): dataframe containing the y values used during validation.
        y_test (dataframe): dataframe containing y values during testing.
    """

    def __init__(self, param_constraints, build_model, train_data, val_data=None, val_split=0.3,
                 test_data=None, retain=0.1, parent_frac=0.3, n_children_per_couple=2, train_chance=0.5,
                 reject_select_chance=0.1, mutation_rate=0.3, force_mutate=True, pop_size=20, log_dir="./logs",
                 log_path_params=['optimizer', 'batch_size'], train_verbose=True, individual=True, cache=False,
                 early_stopping=True, early_stop_patience=10, tensorboard=False, model_checkpoint=False,
                 csv_logger=True, user_callbacks=[], y_train=None, y_val=None, y_test=None, penalty=0.1):
        """
        Constructor of the optimizer.
        """

        self.param_constraints = param_constraints
        self.build_model = build_model

        self.train_data = train_data
        self.val_data = val_data
        self.val_split = val_split
        self.test_data = test_data

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.retain = retain
        self.parent_frac = parent_frac
        self.n_children_per_couple = n_children_per_couple
        self.reject_select_chance = reject_select_chance
        self.mutation_rate = mutation_rate
        self.force_mutate = force_mutate
        self.pop_size = pop_size
        self.individual = individual
        self.cb_early_stop = early_stopping
        self.cb_tensorboard = tensorboard
        self.cb_model_checkpoint = model_checkpoint
        self.cb_csv_logger = csv_logger
        self.penalty = penalty
        self.cache = cache
        self.trained_networks = {}
        self.train_chance = train_chance

        self.early_stop_patience = early_stop_patience
        self.user_callbacks = user_callbacks

        self.generation_history = {}
        self.current_generation = 0
        self.log_dir = log_dir
        self.log_path_params = log_path_params
        self.train_verbose = train_verbose
        self.train_verbose = train_verbose

        log_dir_exists = False

        if os.path.exists(self.log_dir):
            if os.path.isdir(self.log_dir):
                assert not os.listdir(
                    self.log_dir), "log_dir is not empty, please provide an empty or non existing directory."
                log_dir_exists = True
            else:
                assert False, "log_dir exists and is not a directory, please provide an empty or non existing directory."

        assert self.pop_size * self.parent_frac > 1.5, "Population must be able to have at least two parents: current min = {}".format(
            self.pop_size * self.parent_frac)

        # if requested create cache
        if self.cache:
            self.trained_networks = {}

        # check if all data specified in inputs is in train_data
        test_params = self.param_constraints['io_config']['inputs']
        if self.y_train is None:
            test_params = test_params + self.param_constraints['io_config']['outputs']
        for i in test_params:
            try:
                self.train_data[i]
            except:
                assert False, '{} in param_constraints is not found in train_data'.format(i)

        # check if test_data has the same columns as train_data
        if self.test_data is not None:
            assert set(self.test_data.columns) == set(self.train_data.columns), 'Test and train columns do not match'

        # check if val_data has the same columns as train_data
        if self.val_data is not None:
            assert set(self.val_data.columns) == set(
                self.train_data.columns), 'Validation and train columns do not match'
        else:
            # check if val_split is above 0 when no val_data is given
            assert self.val_split > 0, "No validation data available, either provide val_data or set val_split>0."

        # check if only a single output_shape is given in param_constraints
        assert len(self.param_constraints['io_config']['output_shape']) == 1, 'output_shape can only be a single entry.'

        if any([y_train is not None, y_val is not None, y_test is not None]):
            assert y_train is not None, "Please provide y_train"
            if val_data is not None:
                assert y_val is not None, "Please provide y_val"
            if test_data is not None:
                assert y_test is not None, "Please provide y_test"

        # check if the following are in the param_constraints:
        # 'network_config' - n_layers
        # 'layer_config'
        # 'io_config' - 'input_shape', 'inputs', 'output_shape', 'outputs'

        necessary_entries = ['network_config', 'layer_config', 'io_config']
        assert all(x in self.param_constraints.keys() for x in necessary_entries), \
            "One or more of the following is missing from param_constraints: {}".format(necessary_entries)

        assert 'n_layers' in self.param_constraints['network_config'], \
            "n_layers is missing from network_config in param_constraints."

        io_entries = ['input_shape', 'inputs', 'output_shape', 'outputs']
        assert all(x in self.param_constraints['io_config'] for x in io_entries), \
            "One or more of the following is missing from param_constraints: {}".format(io_entries)

        if not log_dir_exists:
            os.makedirs(self.log_dir)
        self.head_log = open(os.path.join(self.log_dir, 'optimizer.log'), 'a')
        print(
            'Started operation at ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' with the following parameters:',
            file=self.head_log)
        print('retain : {}'.format(self.retain), file=self.head_log)
        print('parent fraction: {}'.format(self.parent_frac), file=self.head_log)
        print('number of children per couple: {}'.format(self.n_children_per_couple), file=self.head_log)
        print('reject selection chance: {}'.format(self.reject_select_chance), file=self.head_log)
        print('mutation rate: {}'.format(self.mutation_rate), file=self.head_log)
        print('forced mutation: {}'.format(self.force_mutate), file=self.head_log)
        print('population size: {}'.format(self.pop_size), file=self.head_log)
        print('individual layers: {}'.format(self.individual), file=self.head_log)
        print('early stopping: {}'.format(self.cb_early_stop), file=self.head_log)
        print('tensorboard: {}'.format(self.cb_tensorboard), file=self.head_log)
        print('model_checkpoint: {}'.format(self.cb_model_checkpoint), file=self.head_log)
        print('csv logging: {}'.format(self.cb_csv_logger), file=self.head_log)
        print('percentage penalty per input: {}'.format(self.penalty), file=self.head_log)
        print('early stopping patience: {}\n'.format(self.early_stop_patience), file=self.head_log)
        print('penalty applied after cache', file=self.head_log)

        print('The networks are restricted by the following parameter constraints:', file=self.head_log)
        print_dict(self.param_constraints, print_file=self.head_log)
        print('\n', file=self.head_log)

        self.head_log.flush()

    def __dir__(self):
        return self.keys()

    def create_random(self):
        """
        Creates a random network from param_constraints and returns it.
        
        Returns: 
            dictionary: a random network.
        """

        network = {'network_config': {},
                   'layer_config': {},
                   'io_config': {}
                   }
        for key in self.param_constraints['network_config'].keys():
            network['network_config'][key] = random.choice(self.param_constraints['network_config'][key])

        for key in self.param_constraints['layer_config'].keys():
            config = []

            if self.individual:
                for i in range(network['network_config']['n_layers']):
                    config.append(random.choice(self.param_constraints['layer_config'][key]))
                network['layer_config'][key] = config

            else:
                n_nodes = random.choice(self.param_constraints['layer_config'][key])
                for i in range(network['network_config']['n_layers']):
                    config.append(n_nodes)
                network['layer_config'][key] = config

        input_shape = random.choice(self.param_constraints['io_config']['input_shape'])
        network['io_config']['input_shape'] = input_shape
        network['io_config']['inputs'] = random.sample(self.param_constraints['io_config']['inputs'], input_shape)

        # take output_shape out of list for consistency
        network['io_config']['output_shape'] = self.param_constraints['io_config']['output_shape'][0]
        network['io_config']['outputs'] = self.param_constraints['io_config']['outputs']

        return network

    def create_pop(self):
        """
        Create a population of self.pop_size random networks using self.create_random().
        
        Returns: 
            a list of dictionaries: list of random networks.
        
        """
        population = []
        for i in range(self.pop_size):
            population.append(self.create_random())
        return population

    def compile_network(self, network):
        """
        Compiles the network using the provided model_build function using the paramaters specified by network.
        
        Arguments:
            network (dictionary): dictionary containing the network parameters. 
        
        Returns:
            model: as build by provided model_build.
        """

        model = self.build_model(network)

        return model

    def train_and_score(self, network, network_id=None):
        """
        Compiles the network and trains it on the training data with the enabled callbacks.
        The network is compiled by self.compile_network() which in turn uses the supplied build_model function.
        In the case no val_data is supplied, a val_split section of train_data is used for this purpose instead. 
        When no test_data is specified, the test score will be identical to the last validation score.
        
        Arguments:
            network (dictionary): dictionary containing the network parameters. 
            network_id: (int) Id of network in generation; if None, no logpath is created, blocking tensorboard, checkpoint and csvlogging
        
        Returns:
            keras.callbacks.History: History returned from training.
            float: test score obtained as described above.            
        """

        clear_session()

        if network_id is not None:
            model_save_path = os.path.join(self.log_dir,
                                           "gen_" + "{:03d}".format(self.current_generation),
                                           "id_" + "{:03d}".format(network_id))

            os.makedirs(model_save_path)

        if self.cache:
            network_hash = sha256(network)
            if network_hash in self.trained_networks.keys():
                if random.uniform(0, 1) > self.train_chance:  # skip training based on train_chance
                    # grab score from cache
                    test_score = self.trained_networks[network_hash]['score']
                    total_score = test_score * (1 + self.penalty * network['io_config']['input_shape'])
                    old_save_path = self.trained_networks[network_hash]['save_path']

                    with open(os.path.join(model_save_path, 'cached_score.txt'), 'w') as fp:
                        print(f'Test_score: {test_score}', file=fp)
                        print(f'Total_score: {total_score}', file=fp)
                        print(old_save_path, file=fp)
                        print(network, file=fp)
                        print(network_hash, file=fp)

                    return total_score

        callbacks = []
        if self.cb_early_stop:
            early_stop = EarlyStopping(patience=self.early_stop_patience)
            callbacks.append(early_stop)

        if network_id is not None:
            if self.cb_tensorboard:
                tensorboard = TensorBoard(os.path.join(model_save_path, "tensorboard"),
                                          write_graph=True,
                                          histogram_freq=5)
                callbacks.append(tensorboard)
            if self.cb_model_checkpoint:
                model_checkpoint = ModelCheckpoint(os.path.join(model_save_path, "checkpoints",
                                                                "model.{epoch:02d}-{val_loss:.5f}.hdf5"),
                                                   monitor='val_loss',
                                                   verbose=0,
                                                   save_best_only=True,
                                                   save_weights_only=False,
                                                   mode='auto',
                                                   period=1)
                callbacks.append(model_checkpoint)
            if self.cb_csv_logger:
                csv_logger = CSVLogger(os.path.join(model_save_path, "train_log.csv"),
                                       separator=',',
                                       append=False)
                callbacks.append(csv_logger)

        callbacks = callbacks + self.user_callbacks

        model = self.compile_network(network)

        # create data based on in/outputs
        x_train = self.train_data[network['io_config']['inputs']]
        if self.y_train is not None:
            y_train = self.y_train
        else:
            y_train = self.train_data[network['io_config']['outputs']]

        if self.val_data is not None:
            x_val = self.val_data[network['io_config']['inputs']]
            if self.y_train is not None:
                y_val = self.y_val
            else:
                y_val = self.val_data[network['io_config']['outputs']]
                total_val_data = [x_val, y_val]
        else:
            total_val_data = None

        start_time = time.time()

        history = model.fit(x_train,
                            y_train,
                            verbose=self.train_verbose,
                            epochs=network['network_config']['epochs'],
                            batch_size=network['network_config']['batch_size'],
                            callbacks=callbacks,
                            validation_data=total_val_data,
                            validation_split=self.val_split
                            )

        print('Training time: {}, training val_loss: {}'.format(time.time() - start_time,
                                                                history.history['val_loss'][-1]), file=self.head_log)

        # if test data is specified, use it. Otherwise use last val_loss from training
        if self.test_data is not None:
            x_test = self.test_data[network['io_config']['inputs']]
            if self.y_train is not None:
                y_test = self.y_test
            else:
                y_test = self.test_data[network['io_config']['outputs']]

            start_time = time.time()
            test_score = model.evaluate(x_test, y_test, verbose=0)[0]
            print('Test time: {}, testing loss: {}'.format(time.time() - start_time, test_score), file=self.head_log)
            print(network, file=self.head_log)
        else:
            test_score = history.history['val_loss'][-1]

        # serialize model to JSON
        model_json = model.to_json()
        json_save_path = os.path.join(model_save_path, "json_model")
        os.makedirs(json_save_path)
        with open(os.path.join(json_save_path, "model.json"), "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(os.path.join(json_save_path, "model_weigths.h5"))

        self.head_log.flush()

        # return best score instead of newest
        if self.cache:
            if network_hash in self.trained_networks.keys():
                old_score = self.trained_networks[network_hash]['score']
                if old_score > test_score:
                    # update cached score if new score is better (lower)
                    self.trained_networks[network_hash]['score'] = test_score
                    self.trained_networks[network_hash]['save_path'] = model_save_path
                else:
                    # get hashed score if better
                    test_score = old_score
            else:
                # if the networks was not cached before
                self.trained_networks[network_hash] = {}
                self.trained_networks[network_hash]['score'] = test_score
                self.trained_networks[network_hash]['save_path'] = model_save_path

        # modify test_score to include penalty for number of inputs
        total_score = test_score * (1 + self.penalty * network['io_config']['input_shape'])
        return total_score

    def train_and_score_pop(self, pop):
        """
        Applies the self.train_and_score() method to every network in the current population.
        
        Arguments:
            pop (list of dictionaries): list of dictionaries containing the parameters of 
                the networks in the population.
        
        Returns:
            list of tuples: (network, test_score, history) the latter obtained from self.train_and_score().     
        """

        scores = []

        for i, network in enumerate(pop):
            network_id = i

            # print parameter networks
            config_str = "_".join("{}".format(str(find_param(network, param)).replace(", ", "-"))
                                  for param in self.log_path_params)
            config_str = config_str = re.sub("[\[\]\'\'\"\"\ ]", "", config_str)
            print("Training network {} of {}: {}".format(i, len(pop) - 1, config_str), file=self.head_log)

            # train network
            score = self.train_and_score(network, network_id)
            print("Completed testing with {} val_loss\n".format(score), file=self.head_log)
            scores.append((network, score, network_id))

        self.generation_history[self.current_generation] = scores
        self.current_generation = self.current_generation + 1
        return scores

    def create_new_pop(self, scores):
        """
        Creates a new population of networks using the scores.
        
        Arguments:
            scores (list of tuples): (network, test_score, history) the latter two obtained 
                from self.train_and_score().
        
        Returns:
            list of dictionaries: new population of networks.      
        """

        scores = [x[0] for x in sorted(scores, key=lambda x: x[1])]
        retain_len = int(round(len(scores) * self.retain))
        parent_len = int(round(len(scores) * self.parent_frac))
        assert parent_len > 1, "Not enough parents to propegate next generation. Adapt parent_len or pop_size"

        # get best performing networks
        retained = scores[:retain_len]
        parents = scores[:parent_len]

        # randomly select rejects anyway
        for network in scores[parent_len:]:
            if random.random() < self.reject_select_chance:
                parents.append(network)

        # fill out pop with children
        new_pop = retained[:]
        while len(new_pop) < self.pop_size:
            father = random.randint(0, len(parents) - 1)
            mother = random.randint(0, len(parents) - 1)
            if father == mother:
                continue
            children = self.breed(parents[father], parents[mother])
            for child in children:
                if len(new_pop) < self.pop_size:
                    new_pop.append(child)
        return new_pop

    def breed(self, father, mother, verbose=0):
        """
        Combines the paramaters from the father and mother parameters randomly
        and returns self.n_children_per_couple. Mutates childs randomly with
        the self.mutate() method based on self.mutation_rate.
        
        Arguments:
            father (dictionary): network 1 to be used in breeding.
            mother (dictionary): network 2 to be used in breeding.
            verbose (bool): whether or not to print debug info.
            
        Returns:
            list of dictionaries: the self.n_children_per_couple networks obtained by 
                combining the mother and father network
        """

        children = []

        for i in range(self.n_children_per_couple):
            child = {}
            for param_dict in father.keys():
                child[param_dict] = {}

                if param_dict == 'network_config':
                    # pick randomly from mother or father
                    for param in father[param_dict].keys():
                        child[param_dict][param] = random.choice([father[param_dict][param], mother[param_dict][param]])

                elif param_dict == 'layer_config':
                    # randomly select mixing coefficient
                    # combine len(father)*(mixing) + len(mother)*(1-mixing)
                    mixing = float(random.randint(10, 90)) / 100

                    # determine where to stop (father) and where to start (mother) copying
                    father_stop = int(round(father['network_config']['n_layers'] * mixing))
                    mother_start = 0
                    if father_stop == 0:
                        # guarantee the first layer of father is always copied to child
                        father_stop = 1
                        # mother network will start copy from one index further (so one layer smaller)
                        mother_start += 1
                    # add the length determined by mixing
                    mother_start += int(round(mother['network_config']['n_layers'] * mixing))

                    # apply mixing for every entry in layer_config
                    for param in father[param_dict].keys():
                        father_param = father[param_dict][param]
                        mother_param = mother[param_dict][param]

                        child[param_dict][param] = father_param[:father_stop]

                        # If no mother would be mixed in, replace the last layer with the mother layer
                        # otherwise copy over the length determined by the mixing
                        if mother_start == mother['network_config']['n_layers']:
                            child[param_dict][param][-1] = mother_param[-1]
                        else:
                            child[param_dict][param].extend(mother_param[mother_start:])

                elif param_dict == 'io_config':
                    # TODO?: make general, now requires 4 entries

                    # pick either mother or father input_length
                    child[param_dict]['input_shape'] = random.choice([father[param_dict]['input_shape'],
                                                                      mother[param_dict]['input_shape']])

                    # create list with all unique inputs of both mother and father combined
                    diff = list(set(mother[param_dict]['inputs']) - set(father[param_dict]['inputs']))
                    total_list = father[param_dict]['inputs'] + diff

                    # create list with inputs_shape different unique entries
                    child[param_dict]['inputs'] = random.sample(total_list, child[param_dict]['input_shape'])

                    # just copy outputs from father, as they should be consistent
                    child[param_dict]['outputs'] = father[param_dict]['outputs']
                    child[param_dict]['output_shape'] = father[param_dict]['output_shape']

            # correct the number of layers in network_config
            total_child_len = father_stop + mother['network_config']['n_layers'] - mother_start
            child['network_config']['n_layers'] = total_child_len

            if verbose:
                # print father['layer_config']['n_nodes'][:father_stop], mother['layer_config']['n_nodes'][mother_start:]
                # print child['layer_config']['n_nodes']
                # print
                pass

            # if the child has been selected, mutate it
            if random.random() <= self.mutation_rate:
                self.mutate(child)
            children.append(child)

        return children

    def mutate(self, network):
        """
        Mutates one of the parameters of the provided network (if selected paramater is mutable).
        
        Arguments:
            network (dictionary): dictionary containing the network parameters.
        
        Returns:
            dictionary: mutated network.
            
        """
        old_network = deepcopy(network)

        mutated = False

        while mutated == False:
            # Select a feature to mutate
            param_dict = random.choice(list(network.keys()))
            param = random.choice(list(network[param_dict].keys()))

            if param_dict == 'layer_config':
                n = random.randint(0, len(network[param_dict][param]) - 1)
                new = random.choice(self.param_constraints[param_dict][param])
                network[param_dict][param][n] = new

            elif param_dict == 'network_config':
                if param == 'n_layers':

                    # Add or substract a layer if it does not exceed the min and max layers
                    n_layers = network['network_config']['n_layers']
                    delta = random.choice([-1, 1])
                    # if not (((n_layers + delta) < 1) or ((n_layers + delta) > max(self.param_constraints['network_config']['n_layers']))):
                    if (n_layers + delta) in self.param_constraints['network_config']['n_layers']:
                        # select where to add or sub; 1 and n_layers -2 so the in/output layers are not altered (CHANGED)

                        if delta == -1:
                            pos = random.randint(0, network['network_config']['n_layers'] - 1)
                            for key in network['layer_config'].keys():
                                del (network['layer_config'][key][pos])

                        if delta == 1:
                            pos = random.randint(0, network['network_config']['n_layers'])
                            for key in network['layer_config'].keys():
                                network['layer_config'][key].insert(pos, random.choice(
                                    self.param_constraints['layer_config'][key]))

                        network['network_config']['n_layers'] = n_layers + delta

                else:
                    new = random.choice(self.param_constraints[param_dict][param])
                    network[param_dict][param] = new

            elif param_dict == 'io_config':
                if param == 'input_shape':
                    if len(self.param_constraints['io_config'][
                               'input_shape']) != 1:  # only try to modify the input shape if it is longer than 1
                        input_shape = network[param_dict][param]
                        delta = random.choice([-1, 1])

                        # check if resulting input_shape is valid; if not try again
                        new_shape = input_shape + delta
                        while new_shape not in self.param_constraints['io_config']['input_shape']:
                            delta = random.choice([-1, 1])
                            new_shape = input_shape + delta

                        if delta == -1:
                            pos = random.randint(0, input_shape - 1)
                            del (network['io_config']['inputs'][pos])

                        elif delta == 1:
                            not_in_list = list(set(self.param_constraints['io_config']['inputs']) - set(
                                network['io_config']['inputs']))
                            new_random = random.choice(not_in_list)
                            network['io_config']['inputs'].append(new_random)

                        network[param_dict][param] = new_shape


                elif param == 'inputs':
                    # select random input and remove, find new unique in its place (could be the same)
                    pos = random.randint(0, len(network[param_dict][param]) - 1)
                    del (network[param_dict][param][pos])
                    not_in_list = list(set(self.param_constraints[param_dict][param]) - set(network[param_dict][param]))
                    new_random = random.choice(not_in_list)
                    network[param_dict][param].append(new_random)

            if self.force_mutate == True:
                mutated = not network == old_network
            else:
                mutated = True  # always allow to continue

        return

    def evolve(self, pop):
        """
        Executes one cycle of evolution by first training and scoring every network and 
        then creating a new population based on the results.
        
        Arguments:
            pop (list of dictionaries): list of dictionaries containing the parameters of 
                the networks in the pop.
                
        Returns:
            list of dictionaries: new (evolved) population
        """

        scores = self.train_and_score_pop(pop)
        new_pop = self.create_new_pop(scores)
        return new_pop

    def get_gen_scores(self, gen_id):
        """
        Returns the scores of the specified generation.
        
        Arguments:
            gen_id (int): generation id.
        
        Returns:
            list of tuples: (network, test_score, history) note the history object is broken
                due to clearing the tensorflow backend.
        """

        history = self.generation_history[gen_id]
        score = []
        for network in history:
            score.append(network[1])
        return score

    def give_unique_log_file(self, file_name):
        """
        Creates a unique string for log files by append a number so no data is lost.
        
        Arguments:
            file_name (string): name of the log file.
        
        Returns:
            string: new unique file_name.
        """

        for i in itertools.count():
            log_file = os.path.join(self.log_dir, file_name + "-{}".format(i))
            if os.path.exists(log_file):
                continue
            else:
                return log_file

    def pickle_all_gen(self):
        """
        Uses the pickle library to pickle all generations to file. The file is located in
        log_dir/pickles/pickle_all-*.
        """

        file_name = self.give_unique_log_file("pickles/pickle_all")
        try:
            os.makedirs(file_name.rsplit("/", 1)[0])
        except:
            pass

        with open(file_name, 'wb') as fp:
            pickle_list = self.generation_history
            pickle.dump(pickle_list, fp)
            print("Pickled all generations to file: {}, at time:".format(file_name), file=self.head_log)
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '/n', file=self.head_log)
        return

    def pickle_gen(self, gen_id):
        """
        Uses the pickle library to pickle all generations to file. The file is located in
        log_dir/pickles/pickle_gen*.
        
        Argurments:
            gen_id (int): the generation data to be pickled.
        
        """

        file_name = self.give_unique_log_file("pickles/pickle_gen{}".format(gen_id))
        try:
            os.makedirs(file_name.rsplit("/", 1)[0])
        except:
            pass

        with open(file_name, 'wb') as fp:
            pickle_list = self.generation_history[gen_id]
            pickle.dump(pickle_list, fp)
            print("Pickled generation {} to file: {} \n".format(gen_id, file_name), file=self.head_log)
        return

    def test_pop_creation(self, n_gen):
        """
        Creates random population and propegates it n_gen times using random scores.
        
        Arguments:
            n_gen (int): the number of times to propegate the population. 
            
        Returns:
            list of dictionaries: final population after n_gens of propegation.
        """

        pop = self.create_pop()
        for i in range(n_gen):
            score = [[network, random.random()] for network in pop]
            pop = self.create_new_pop(score)
        return pop

    def test_compile(self, pop):
        """
        Compiles every network in pop; Used to test if pop is valid.
        
        Arguments:
            pop (list of dictionaries): list of dictionaries containing the parameters of 
                the networks in the pop.
        """

        for i, p in enumerate(pop):
            print(i, p, '\n')
            self.compile_network(p)
        return

    def test_mutate(self, n_mutations):
        """
        Mutates a new random network n_mutations times and returns it.
        
        Argurments:
            n_mutations (int): the amount of times a network must be mutated.
        
        Returns:
            dictionary: n_mutations mutated network.
        
        """

        network = self.create_random()
        for i in range(n_mutations):
            self.mutate(network)
        return network
