# MultiparameterBayesianEstimation
Bayesian quantum thermometry and bath characterisation of a single qubit and multilevel system.

## Overview

This repo considers two different systems:

1) The steady-state of a two-level system coupled to a cold heat bath in addition to an unknown number of (parasitic) hot baths.

2) Non-equilibrium Cs atom probes (7 level system) immersed in an ultracold Rb bath.
This follows the experimental setup of Bouton et al. (2019) and Feß et al. (2026). 

This repository computes Bayesian estimates, mean logarithmic error, and information gain under various priors.
We use a number of different (scan, _a priori_ optimised, and adaptive) protocols.

## Requirements
The examples require standard scientific Python packages:
- `numpy`
- `scipy`
- `matplotlib`

## License

[MIT](https://choosealicense.com/licenses/mit/)