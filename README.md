python 3.9.12

This file explains how to utilize maximal cliques to classify labels of nodes in a semi-supervised learning.

More specifically, for a given network, we employ the set of maximal cliques which is a subset of all higher order cliques to classify label of each node by using probability based objective function.

The objective function is motivated by the intuition that nodes densely interconnected with edges in a given graph are likely to exhibit similar labels.

A detail explanation of the function is presented in https://arxiv.org/abs/2310.10114.

Files demonstrate the structure of the objective function and applying method of maximal cliques as well as optimization procedure. 

Balanced and Imbalanced experiments.ipynb : Experiment on balanced and imbalanced generated model using the planted partition model

function.py : utils and function
