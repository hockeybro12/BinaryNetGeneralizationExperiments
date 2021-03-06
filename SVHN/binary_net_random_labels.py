
import time

from collections import OrderedDict

import numpy as np

# specifying the gpu to use
import theano.sandbox.cuda
#theano.sandbox.cuda.use('gpu0') 
import theano
import theano.tensor as T

import lasagne

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise

# Our own rounding function, that does not set the gradient to 0 like Theano's
class Round3(UnaryScalarOp):
    
    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = round(%(x)s);" % locals()
    
    def grad(self, inputs, gout):
        (gz,) = gout
        return gz, 
    
# round 3 is a scalar operation rounding function    
round3_scalar = Round3(same_out_nocomplex, name='round3')
# Generalizes a scalar op to tensors. Basically round the tensors using our function
round3 = Elemwise(round3_scalar)

# only 2 cut points, speeds up training vs normal sigmoid function
def hard_sigmoid(x):
    # clip - limit the values in an array (self, min, max)
    return T.clip((x+1.)/2.,0,1)

# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1 
# during back propagation
# hard tanh = -1 if z < -1, 1 if z > 1, else z
def binary_tanh_unit(x):
    return 2.*round3(hard_sigmoid(x))-1.
    
def binary_sigmoid_unit(x):
    return round3(hard_sigmoid(x))
    
# The weights' binarization function, 
# taken directly from the BinaryConnect github repository 
# (which was made available by his authors)
def binarization(W,H,binary=True,deterministic=False,stochastic=False,srng=None):
    
    # (deterministic == True) <-> test-time <-> inference-time
    if not binary or (deterministic and stochastic):
        print("not binary")
        Wb = W
    
    else:
        
        # [-1,1] -> [0,1]
        Wb = hard_sigmoid(W/H)
        # Wb = T.clip(W/H,-1,1)
        
        # Stochastic BinaryConnect
        if stochastic:
        
            # print("stoch")
            # .cast -> same shape, differnt type. float.
            # random numbers binomial distribution. Weight is gotten from the distribution
            Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)

        # Deterministic BinaryConnect (round to nearest)
        else:
            # print("det")
            Wb = T.round(Wb)
        
        # 0 or 1 -> -1 or 1
        # H = 1. 
        Wb = T.cast(T.switch(Wb,H,-H), theano.config.floatX)
 
    return Wb

# This class extends the Lasagne DenseLayer to support BinaryConnect
class DenseLayer(lasagne.layers.DenseLayer):
    
    def __init__(self, incoming, num_units, 
        binary = True, stochastic = True, H=1.,W_LR_scale="Glorot", **kwargs):

        self.binary = binary
        self.stochastic = stochastic
        
        self.H = H
        if H == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.H = np.float32(np.sqrt(1.5/ (num_inputs + num_units)))
            # print("H = "+str(self.H))
            
        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.W_LR_scale = np.float32(1./np.sqrt(1.5/ (num_inputs + num_units)))

        # generate a random stream of numbers     
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        
        if self.binary:
            # pass the weight as a uniform (rectangle) distribution from -H to H as an extra arg. 
            super(DenseLayer, self).__init__(incoming, num_units, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)
            # add the binary tag to weights            
            self.params[self.W]=set(['binary'])
            
        else:
            # otherwise the DenseLayer init as normal
            super(DenseLayer, self).__init__(incoming, num_units, **kwargs)
        
    # compute an expression for the output for a single layer given its input
    def get_output_for(self, input, deterministic=False, **kwargs):
        
        # get the weights that are binarized
        self.Wb = binarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)
        Wr = self.W
        self.W = self.Wb
            
        rvalue = super(DenseLayer, self).get_output_for(input, **kwargs)
        
        self.W = Wr
        
        return rvalue

# This class extends the Lasagne Conv2DLayer to support BinaryConnect
class Conv2DLayer(lasagne.layers.Conv2DLayer):
    
    def __init__(self, incoming, num_filters, filter_size,
        binary = True, stochastic = True, H=1.,W_LR_scale="Glorot", **kwargs):
        
        self.binary = binary
        self.stochastic = stochastic
        
        self.H = H
        if H == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))
            # print("H = "+str(self.H))
        
        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.W_LR_scale = np.float32(1./np.sqrt(1.5 / (num_inputs + num_units)))
            # print("W_LR_scale = "+str(self.W_LR_scale))
            
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
            
        if self.binary:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)   
            # add the binary tag to weights            
            self.params[self.W]=set(['binary'])
        else:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, **kwargs)    
    
    # convolves input with self.W producing an output
    def convolve(self, input, deterministic=False, **kwargs):
        
        self.Wb = binarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)
        Wr = self.W
        self.W = self.Wb
        
        # use the super class convolve method    
        rvalue = super(Conv2DLayer, self).convolve(input, **kwargs)
        
        self.W = Wr
        
        return rvalue

# This function computes the gradient of the binary weights
def compute_grads(loss,network):
    # get all layers below the given one including the given one    
    layers = lasagne.layers.get_all_layers(network)
    grads = []
    
    for layer in layers:
        # get the list of theano expressions that parametrize the layer with the binary tag
        params = layer.get_params(binary=True)
        if params:
            # print(params[0].name)
            # compute the gradient of the loss with respect to the binary weight
            grads.append(theano.grad(loss, wrt=layer.Wb))
                
    return grads

# This functions clips the weights after the parameter update to [-1, 1]
def clipping_scaling(updates,network):
    
    layers = lasagne.layers.get_all_layers(network)
    # remembers the order entries were added
    updates = OrderedDict(updates)
    
    for layer in layers:
    
        params = layer.get_params(binary=True)
        for param in params:
            print("W_LR_scale = "+str(layer.W_LR_scale))
            print("H = "+str(layer.H))
            updates[param] = param + layer.W_LR_scale*(updates[param] - param)
            updates[param] = T.clip(updates[param], -layer.H,layer.H)     

    return updates
        
# Given a dataset and a model, this function trains the model on the dataset for several epochs
# (There is no default trainer function in Lasagne yet)
def train(train_fn,val_fn,
            model,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            X_train,y_train,
            X_val,y_val,
            X_test,y_test, 
            save_path=None,
            shuffle_parts=1):
    
    # A function which shuffles a dataset
    def shuffle(X,y):
        
        # print(len(X))
        
        chunk_size = len(X)/shuffle_parts
        shuffled_range = range(chunk_size)
        
        X_buffer = np.copy(X[0:chunk_size])
        y_buffer = np.copy(y[0:chunk_size])
        
        for k in range(shuffle_parts):
            
            np.random.shuffle(shuffled_range)

            for i in range(chunk_size):
                
                X_buffer[i] = X[k*chunk_size+shuffled_range[i]]
                y_buffer[i] = y[k*chunk_size+shuffled_range[i]]
            
            X[k*chunk_size:(k+1)*chunk_size] = X_buffer
            y[k*chunk_size:(k+1)*chunk_size] = y_buffer
        
        return X,y
        
        # shuffled_range = range(len(X))
        # np.random.shuffle(shuffled_range)
        
        # new_X = np.copy(X)
        # new_y = np.copy(y)
        
        # for i in range(len(X)):
            
            # new_X[i] = X[shuffled_range[i]]
            # new_y[i] = y[shuffled_range[i]]
            
        # return new_X,new_y
    
    # This function trains the model a full epoch (on the whole dataset)
    def train_epoch(X,y,LR):
        
        loss = 0
        batches = len(X)/batch_size
        
        for i in range(batches):
            # pass in the x_train, y_train, learning rate
            new_loss = train_fn(X[i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size],LR)
            loss += new_loss

        loss/=batches
        
        return loss
    
    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(X,y):
        
        err = 0
        loss = 0
        batches = len(X)/batch_size
        
        for i in range(batches):
            new_loss, new_err = val_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
            err += new_err
            loss += new_loss
       
        err = err / batches * 100
        loss /= batches

        return err, loss
    
    # shuffle the train set
    X_train,y_train = shuffle(X_train,y_train)
    best_val_err = 100
    best_epoch = 1
    LR = LR_start
    
    # We iterate over epochs:
    for epoch in range(num_epochs):
        
        start_time = time.time()
        
        train_loss = train_epoch(X_train,y_train,LR)

        train_err, train_loss_new = val_epoch(X_train, y_train)
        X_train,y_train = shuffle(X_train,y_train)
        
        val_err, val_loss = val_epoch(X_val,y_val)
        test_err, test_loss = val_epoch(X_test,y_test)

        train_errors_array = []
        test_errors_array = []
        # find the values of the class
        class_values_train = np.argmax(y_train, axis=1)
        class_values_test = np.argmax(y_test, axis=1)
        # determine the test/train error values for each of the separate classes
        for i in range(len(y_train[0])):
            # find the indices of this specific class
            class_values_train_indices = np.where(class_values_train==i)
            class_values_test_indices = np.where(class_values_test==i)

            # get the indices that correspond to this specific class.
            class_X_train = X_train[class_values_train_indices]
            class_y_train = y_train[class_values_train_indices]
            # calculate the train/test error for this specific class
            class_train_err, class_train_loss = val_epoch(class_X_train, class_y_train)
            train_errors_array.append(class_train_err)

            class_X_test = X_test[class_values_test_indices]
            class_y_test = y_test[class_values_test_indices]
            class_test_err, class_test_loss = val_epoch(class_X_test, class_y_test)
            test_errors_array.append(class_test_err)
        
        if save_path is not None:
            np.savez(save_path, *lasagne.layers.get_all_param_values(model))
            print("Saving binary net parameters at path " + save_path)

        # test if validation error went down
        if val_err <= best_val_err:
            
            # update the values, save model
            best_val_err = val_err
            best_epoch = epoch+1
            
            

        epoch_duration = time.time() - start_time
        
        # Then we print the results for this epoch:
        print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
        print("  LR:                            "+str(LR))
        print("  training error:                "+str(train_err)+"%")
        print("  training loss:                 "+str(train_loss))
        print("  validation loss:               "+str(val_loss))
        print("  validation error rate:         "+str(val_err)+"%")
        print("  best epoch:                    "+str(best_epoch))
        print("  best validation error rate:    "+str(best_val_err)+"%")
        print("  test loss:                     "+str(test_loss))
        print("  test error rate:               "+str(test_err)+"%") 
        for i in range(len(y_train[0])):
            print("        Class " + str(i) + " train error:     " + str(train_errors_array[i]) + "%")
            print("        Class " + str(i) + " test error:      " + str(test_errors_array[i]) + "%")        
        
        # decay the LR
        LR *= LR_decay
