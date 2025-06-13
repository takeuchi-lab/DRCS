#!/bin/bash

# datalist=(data/australian.lsvm data/breast-cancer.lsvm data/heart.lsvm data/ionosphere_scale.lsvm data/splice_scale.lsvm cifar10)
# kernellist=("linear" "rbf" "cntk")
# methodlist=("greedy1" "greedy2" "greedy3")
# `lam` and `weightchange` can be set to any value.

for f in cifar10; do
    for kernel in rbf; do
		for method in greedy3; do
			for lam in 1.5; do
				for weightchange in 1.05; do
					data="$(basename "$f" .lsvm)"
					python experiment_drcs.py $f $data $kernel $method $weightchange $lam
				done
			done
		done
    done
done