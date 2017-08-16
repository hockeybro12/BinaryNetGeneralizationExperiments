# BinaryNet Generalization Experiments


This repository provides the code to reproduce generalization experiments on [BinaryNet](https://arxiv.org/abs/1602.02830)

The experiments are done with random labels on both the CIFAR-10 and SVHN Dataset. Experiments are run in [Theano](http://deeplearning.net/software/theano/install.html) and Lasagne with Python 2.7. You can download the datasets using the scripts from [PyLearn2](https://github.com/lisa-lab/pylearn2/tree/master/pylearn2/scripts/datasets)

These experiments are similar to the ones used in paper [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530). We randomize the labels of the input dataset and observe what happens to the training and test error.

## SVHN

Train the standard model with random labels with the following command. 

`python svhn_random_labels.py`

After you have trained the CNN, you can evaluate it. This will load the first 7000 entries of the SVHN dataset and evaluate the train error based on the parameters that have been saved in the file `model_parameters_random_labels_svhn.npz`. The parameters are saved by the training file, and you can change the name there. Due to the fact that the labels are random, you must save the dataset and load it in order to get the same results when evaluating.

`python svhn_evaluation_random_labels.py`

We also provide files to train with dropout and l2 regularization. Run them with the following commands.

`python svhn_random_labels_dropout.py`

`python svhn_random_labels_l2_regularization.py` 

## CIFAR-10

Train the model with the following command. 

`python cifar10_random_labels.py`

For dropout, use:

`cifar10_train_acc_dropout.py`

We build on the code from [Matthieu Courbariaux](https://github.com/MatthieuCourbariaux/BinaryNet) for our experiments.
