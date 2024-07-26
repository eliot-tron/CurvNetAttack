# CurvNetAttack
A playtest example of the impact of curvature on adversarial attacks.

This repository contains the code used for the article *Adversarial attacks on neural networks through canonical Riemannian foliations* by Eliot Tron, Nicolas Couellan and Stéphane Puechmorel, available at [https://arxiv.org/abs/2203.00922](https://arxiv.org/abs/2203.00922)

We propose a two-step method based on the eigenvectors of the Fisher Information Matrix (FIM).

# How to use CurvNetAttack
## Prerequisites
The code has been written under python 3.9.18. You can find the requirements in [requirements.txt](./requirements.txt). 
## [main.py](./main.py)
```
usage: main.py [-h] [--dataset {MNIST,XOR,XOR-old,CIFAR10}] [--nsample N]
               [--task {plot-attack,plot-attacks-2D,fooling-rates,plot-leaves,plot-curvature,plot-contour,inf-norm,save-attacks}]
               [--nl f] [--startidx STARTIDX] [--random] [--run attack [attack ...]]
               [--attacks path [path ...]] [--range a b] [--savedirectory path] [--cpu]
               [--double]

Compute a One-Step or Two-Step spectral attack, and some visualizations.

optional arguments:
  -h, --help            show this help message and exit
  --dataset {MNIST,XOR,XOR-old,CIFAR10}
                        Dataset name to be used.
  --nsample N           Number of points to compute the attack on.
  --task {plot-attack,plot-attacks-2D,fooling-rates,plot-leaves,plot-curvature,plot-contour,inf-norm,save-attacks}
                        Task.
  --nl f                Non linearity used by the network.
  --startidx STARTIDX   Start index of the input points
  --random              Permutes randomly the inputs.
  --run attack [attack ...]
                        List of attacks to run on the task.
  --attacks path [path ...]
                        Path to (budget, test_point, attack_vectors) if you had them
                        precomputed.
  --range a b           Range for the generated datapoints for the Xor task. Datapoints will
                        lie in [a,b)².
  --savedirectory path  Path to the directory to save the outputs in.
  --cpu                 Force device to be cpu, even if cuda is available.
  --double              Use double precision (1e-16) for the computations (recommanded).
```

### Task:
- `plot-attack`: Plot the result of the attacks for one input image. 
- `plot-attacks-2D`: Plot the attack vectors for 2D datasets (XOR and OR).
- `fooling-rates`: Plot the graph of the attacks' fooling rates with respect to the Euclidean budget.
- `plot-leaves`: Plot the leaves (kernel or transverse) for 2D datasets (XOR and OR).
- `plot-curvature`: Plot the curvature for 2D datasets (XOR and OR).
- `plot-contour`: Plot the contour of the neural network for 2D datasets (XOR and OR).
- `inf-norm`: Plot the infinity norm of the attacks with respect to the Euclidean budget.
- `save-attacks`: Save the attack vectors for the specified adversarial attacks. It will save the list of the budgets, the tensor of the input points the attacks were computed on and the attack vectors (one file per attack procedure). These .pt files can then be passed to the `--attacks` option. It allows to do all the previous plots and computations without recomputing the attacks each time.

### Example
The following computes the fooling rates for OSSA, TSSA and APGD on 100 points from MNIST with a sigmoid as activation function and double precision.
```
python3 main.py --dataset MNIST --nsample 100 --task fooling-rates --nl sigmoid --random --run OSSA TSSA APGD --double
```

## [adversarial_attack.py](./adversarial_attack.py)
- `AdversarialAttack`: Base class for any adversarial attack.
- `TwoStepSpectralAttack`: Class implementing our two-step attack, with a fixed `first_step_size` of 60% of the total Euclidean budget.
- `OneStepSpectralAttack`: Class implementing the one-step attack found in [The Adversarial Attack and Detection under the Fisher Information Metric](https://arxiv.org/abs/1810.03806) by Zhao et al. 
- `AdversarialAutoAttack`: Class calling the API [AutoAttack](https://github.com/fra31/auto-attack) with settings: the $L_2$ norm and the *standard* version of the attack.
- `APGDAttack`: Class calling the API [AutoAttack](https://github.com/fra31/auto-attack) with settings: the $L_2$ norm and the *curstom* version with only *apgd-ce* for the attack.
- `GeodesicSpectralAttack`: Class implementing an experimental true geodesic attack. **Warning**: very slow.

## [adversarial_attack_plots.py](./adversarial_attack_plots.py)
This file contains all the functions to plot the results of the experiments with the adversarial attacks.

## [geometry.py](./geometry.py)
This file contains the functions used to compute the different geometric notions used by some adversarial attack procedures.

