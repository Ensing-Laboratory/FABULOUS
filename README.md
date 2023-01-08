![scheme](fabulous.jpeg?raw=true)

# Ferry, Alberto, and Bernd's Ultimate Learning Of Useful Slow-coordinates

The Fabulous package is a machine learning tool to find an optimal set of collective variables (CVs) from a pool of candidates.
Here, CVs are geometric descriptors of a molecular transition, such as bond distances, angles, coordination numbers, etc.

Fabulous combines a neural network that is trained to provide the atomistic positions given as input a set of CV values. 
An optimimal set of CVs is found by using a genetic algorithm that selects the fittest neural network from a pool of instances, each with
a different starting set of CVs, randomly chosen from a pool of candidate CVs. The fitness of each network is determined from the loss 
of the trained neural network. A good set of input CVs will give a better desciption of the molecular configuration, and thus result in
a small training loss of the network, and a higher fitness in the genetic algorithm.

As described in Ref. [1], Fabulous can be used in different ways, depending on the available datasets for training.
Ideally, the dataset consists of MD trajectories of the molecular transition that start and end in the stable reactant and product states.
However, such trajectories are often not available or very expensive to compute.

A second option is to first apply the Fabulous algorithm to MD trajectories from the stable reactant and product states, i.e. without
information of the molecular configurations along the connecting transition. The optimimal set of CVs obtained by Fabulous can 
then be used to bias the molecular system and obtain trajectories along the transition. With these trajectories, Fabulous can 
optain a better set of CVs, which can subsequently be used in a second biased simulation, and so forth, until convergence.


### Example
To run the example as is, unpack the following archives before running main.py:
```bash
.examples/alanine_dipeptide/data/AD/CV/CV_data.zip
.examples/alanine_dipeptide/data/AD/TPS/trjs.zip
```


A method of analysing the results is given in:
```bash
.examples/alanine_dipeptide/analysis/
```

### References

[1] Discovering Collective Variables of Molecular Transitions via Genetic Algorithms and Neural Networks.
Ferry Hooft, Alberto Pérez de Alba Ortíz, and Bernd Ensing. _J. Chem. Theory Comput._ __17__ (2021), 2294–2306.
[DOI: 0.1021/acs.jctc.0c00981](https://doi.org/10.1021/acs.jctc.0c00981).
