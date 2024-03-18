python 3.9.12

This file explains how to utilize maximal cliques to classify labels of nodes in a semi-supervised learning.

More specifically, for a given network, we employ the set of maximal cliques which is a subset of all higher order cliques to classify label of each node by using probability based objective function.

The objective function is motivated by the intuition that nodes densely interconnected with edges in a given graph are likely to exhibit similar labels.

A detail explanation of the function is presented in https://arxiv.org/abs/2310.10114.

Files demonstrate the structure of the objective function and applying method of maximal cliques as well as optimization procedure. 

Balanced and Imbalanced experiments.ipynb : Experiment on balanced and imbalanced generated model using the planted partition model

function.py : utils and function

Abstract
In network analysis with higher-order interactions, utilizing all of the cliques in the network appears natural and intuitive. However, this strategy frequently experiences computational inefficiencies due to overlapping information in both higherorder and lower-order cliques. This paper describes and validates a strategy based on the maximal cliques for semi-supervised node classification tasks that takes advantage of higher-order network structure. The findings indicate that the maximal clique approach performs similarly while training on significantly fewer cliques than the allcliques strategy, and furthermore, maximal cliques outperform pairwise interactions in both balanced and imbalanced networks. This implies that the network structure can be adequately extracted using only maximal cliques, despite utilizing significantly fewer
cliques than the all-cliques approach, resulting in substantial computational reduction.
