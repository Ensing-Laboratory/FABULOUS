from __future__ import print_function
import pandas as pd
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt


def get_dense_weights(model):
    total_weigths = []
    for layer in model.layers:
        if 'dense' in layer.get_config()['name']:
            weigths = layer.get_weights()[0]
            total_weigths.append(weigths)
    return total_weigths


def read_CV(CVfile):
    with open(CVfile) as fp:
        buffer = []
        for line in fp.readlines():
            if line.startswith('#! FIELDS'):
                columns = line.split()[2:]
            if line.startswith('#!'):
                continue
            buffer.append([float(x) for x in line.split()])
    return pd.DataFrame(buffer, columns=columns)


def read_MD(MDfile, ref, slice_idx, XTC=False):
    if XTC:
        trj = md.load_xtc(MDfile, ref.top)
    else:
        trj = md.load_pdb(MDfile)

    ref = ref.atom_slice(slice_idx)
    trj = trj.atom_slice(slice_idx)

    trj = trj.superpose(ref)

    xyzfile = trj.xyz
    n_atoms = trj.n_atoms
    n_frames = trj.n_frames
    trj = np.reshape(xyzfile, (n_frames, n_atoms * 3))
    return pd.DataFrame(trj)


def get_layer_sizes(model):
    layers = [model.input.shape.as_list()[1]]
    for layer in model.layers:
        if 'dense' in layer.get_config()['name']:
            n_nodes = layer.get_config()['units']
            layers.append(n_nodes)
    return layers


# adapted from https://gist.github.com/craffel/2d727968c3aaebd10359
def draw_neural_net(model, ax=None, figsize=(12, 12), left=0.1, right=0.9, bottom=0.1, top=0.9, use_weigths=True,
                    line_colors=None, circle_colors=None):
    '''
    Draw a representantion of the dense layers in a neural network using matplotilb.

    parameters:
        - model: keras model
            The keras model (neural network) to be drawn
        - ax: figure axis
            The matplotlib figure axis in which to draw
        - figsize: tuple
            The size of the figure
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
    '''

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        ax.axis('off')

    layer_sizes = get_layer_sizes(model)
    weights = get_dense_weights(model)
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in xrange(layer_size):
            if circle_colors is not None:
                circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                    color=circle_colors[n], ec='k', zorder=4)
            else:
                circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                    color='w', ec='k', zorder=4)
            ax.add_artist(circle)

    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in xrange(layer_size_a):
            for o in xrange(layer_size_b):
                if use_weigths == True:
                    weight = weights[n][m][o]
                    if weight > 0:
                        line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                          [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c=[0, 0, 1, 1],
                                          linewidth=weight)
                    else:
                        line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                          [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c=[1, 0, 0, 1],
                                          linewidth=weight)
                else:
                    if line_colors is not None:
                        l_color = line_colors[n]
                    else:
                        l_color = [0, 0, 0, 1]

                    line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                      [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c=l_color,
                                      linewidth=1)
                ax.add_artist(line)
    return fig


def print_dict(my_dict, print_file=None):
    # prints all keys and there values in dicts including deeper dicts
    for key in my_dict.keys():
        try:
            print_dict(my_dict[key], print_file=print_file)
        except:
            if print_file is not None:
                print(key, my_dict[key], file=print_file)
            else:
                print(key, my_dict[key])


def find_param(my_dict, key):
    if key in my_dict: return my_dict[key]
    for k, v in my_dict.items():
        if isinstance(v, dict):
            item = find_param(v, key)
            if item is not None:
                return item