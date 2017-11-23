# BinaryNet Generalization Experiments

## Paper 

Paper: [Memorization in Binarized Neural Networks](https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnx0aW55bWwyMDE3fGd4OjZmZTRhYTU5YzcyNzNiYzc)

Presented at: [TinyML Workshop at ICML 2017](https://sites.google.com/site/tinyml2017/accepted-papers)

## Information

This repository provides the code to reproduce generalization experiments on [BinaryNet](https://arxiv.org/abs/1602.02830)

The experiments are done with random labels on both the CIFAR-10 and SVHN Dataset. Experiments are run in [Theano](http://deeplearning.net/software/theano/install.html) and Lasagne with Python 2.7. You can download the datasets using the scripts from [PyLearn2](https://github.com/lisa-lab/pylearn2/tree/master/pylearn2/scripts/datasets)

We use the same models as used in the original BinaryNet paper for each of the respective datasets.  

These experiments are similar to the ones used in the paper [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530). We randomize the labels of the input dataset and observe what happens to the training and test error.

## SVHN

Train the standard model with random labels with the following command. 

`python svhn_random_labels.py`

After you have trained the CNN, you can evaluate it. This will load the first 7000 entries of the SVHN dataset and evaluate the train error based on the parameters that have been saved in the file `model_parameters_random_labels_svhn.npz`. The parameters are saved by the training file, and you can change the name there. Due to the fact that the labels are random, you must save the dataset and load it in order to get the same results when evaluating.

`python svhn_evaluation_random_labels.py`

We also provide files to train with dropout, l2 regularization, and stochastic binarization/sigmoid activation function. Run them with the following commands.

`python svhn_random_labels_dropout.py`

`python svhn_random_labels_l2_regularization.py` 

`python svhn_stochastic_sigmoid_random_labels.py`

## CIFAR-10

Train the model with the following command.

`python cifar10_random_labels.py`

For dropout, use:

`python cifar10_train_acc_dropout.py`

For stochastic binarization and sigmoid activation, use:

`python cifar10_stochastic_sigmoid_random_labels.py`

You can also train the model with gaussian noise as the x input, using the following command.

`python cifar10_randX_input.py`


We build on the code from [Matthieu Courbariaux](https://github.com/MatthieuCourbariaux/BinaryNet) for our experiments.
