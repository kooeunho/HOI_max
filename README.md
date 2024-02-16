python 3.9.12

This file explains how to utilize maximal cliques to classify labels of nodes in a semi-supervised learning.

More specifically, for a given network, we employ the set of maximal cliques which is a subset of all higher order cliques to classify label of each node by using probability based objective function.

The objective function is motivated by the intuition that nodes densely interconnected with edges in a given graph are likely to exhibit similar labels.

A detail explanation of the function is presented in https://arxiv.org/abs/2310.10114
