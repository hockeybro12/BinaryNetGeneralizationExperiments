
from __future__ import print_function

import sys
import os
import time

import numpy as np
np.random.seed(1234) # for reproducibility?

# specifying the gpu to use
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu0') 
import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import binary_net_train_acc_random_labels as binary_net

from pylearn2.datasets.svhn import SVHN
from pylearn2.utils.string_utils import preprocess
import logging
import shutil
from pylearn2.utils import serial

from collections import OrderedDict

import random

import binary_ops

import cPickle as pickle

if __name__ == "__main__":
    
    # Batch Normalization parameters
    batch_size = 30
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))
    
    # BinaryOut
    activation = binary_net.binary_tanh_unit
    print("activation = binary_net.binary_tanh_unit")
    # activation = binary_net.binary_sigmoid_unit
    # print("activation = binary_net.binary_sigmoid_unit")
    
    # BinaryConnect    
    binary = True
    print("binary = "+str(binary))
    stochastic = False
    print("stochastic = "+str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = "+str(H))
    # W_LR_scale = 1.    
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(W_LR_scale))
    
    # Training parameters
    num_epochs = 400
    print("num_epochs = "+str(num_epochs))
    
    # Decaying LR 
    LR_start = 0.001
    print("LR_start = "+str(LR_start))
    LR_fin = 0.000001
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    
    # for SVHN, depending on available CPU memory
    # 1, 2, 4, 7 or 14
    shuffle_parts = 1 
    # shuffle_parts = 2 # does not work on bart5
    # shuffle_parts = 4 # seems to work on bart5
    # shuffle_parts = 7 # just to be safe
    print("shuffle_parts = "+str(shuffle_parts))
    
    print('Loading SVHN dataset')

    train_set = SVHN(which_set= 'train', axes= ['b', 'c', 0, 1])
        
#    test_set = SVHN(
#        which_set= 'train',
#        axes= ['b', 'c', 0, 1])

    
    # bc01 format
    # Inputs in the range [-1,+1]
    # print("Inputs in the range [-1,+1]")
#    train_set.X = np.reshape(np.subtract(np.multiply(2./255.,train_set.X),1.),(-1,3,32,32))
#    test_set.X = np.reshape(np.subtract(np.multiply(2./255.,test_set.X),1.),(-1,3,32,32))
    # print(np.max(train_set.X))
    # print(np.min(train_set.X))
#    test_set.y = np.reshape(test_set.y, (-1))
#    train_set.y = np.reshape(train_set.y, (-1))
 
    # replace the train, validation, and test set with random labels
#    for i in range(0, len(test_set.y)):
#        test_set.y[i] = random.randint(0, 9)

#    for i in range(0, len(train_set.y)):
#        train_set.y[i] = random.randint(0, 9)
    
    # one hot the targets
 #   test_set.y = np.float32(np.eye(10)[test_set.y])    
 #   train_set.y = np.float32(np.eye(10)[train_set.y])   
   
 # for hinge loss (targets are already onehot)
 #   test_set.y = np.subtract(np.multiply(2,test_set.y),1.)
 #   train_set.y = np.subtract(np.multiply(2,train_set.y),1.)

    print('Building the CNN...') 
  
    train_set.X = np.load('X_values_SVHN.npy')  
    train_set.y = np.load('Y_values_SVHN.npy')
  
    print(train_set.X.shape)
    train_set.X = train_set.X[:7000,:,:,:]
    train_set.y = train_set.y[:7000,:]
    print(train_set.X.shape)
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    cnn = lasagne.layers.InputLayer(
            shape=(None, 3, 32, 32),
            input_var=input)
    
    # 128C3-128C3-P2             
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=64, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
            
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=64, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
            
    # 256C3-256C3-P2             
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=128, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
            
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=128, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
            
    # 512C3-512C3-P2              
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=256, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
                  
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=256, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
    
    # print(cnn.output_shape)
    
    # 1024FP-1024FP-10FP            
    cnn = binary_net.DenseLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=1024)      
                  
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
            
    cnn = binary_net.DenseLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=1024)      
                  
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
    
    cnn = binary_net.DenseLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=10)      
    
    print(cnn.get_params())
    viewParametersLayer = cnn
                  
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)

    
    train_output = lasagne.layers.get_output(cnn, deterministic=False)

    # squared hinge loss 
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    
    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
   
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], test_err, allow_input_downcast=True)

    # binarize the weights
    with np.load('model_parameters_random_labels_svhn.npz') as f:
        param_values = [f['arr_%d' %i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(cnn, param_values)

    params = lasagne.layers.get_all_params(cnn)
    for param in params:
        if param.name == "W":
            param.set_value(binary_ops.SignNumpy(param.get_value()))
            print(binary_ops.SignNumpy(param.get_value()))
#    netInfo = {'network': cnn, 'params': lasagne.layers.get_all_param_values(cnn)}
#    Net_FileName = 'svhn_random_labels.pkl'
#    pickle.dump(netInfo, open(os.path.join('/notebooks/summer_projectsM/nikhil/BinaryNet/svhn', Net_FileName), 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
    print("running")

    test_error = val_fn(train_set.X, train_set.y)*100.
    print("test_error = " + str(test_error) + "%")

