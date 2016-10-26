import os
import sys
import timeit
import scipy.io
import theano
import numpy as np
import theano.tensor as T

from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams

from mlp import FullyConnectedLayer
from log_reg_sgd import LogisticRegression
from conv_pool import ConvPoolLayer

def ReLU(z):
	return T.maximum(0.0, z)

def load_data():
	
	#Loading training data  
	data = scipy.io.loadmat('train.mat')
	train_data = np.asarray(data['patches'])
	# Size is number of patches, number of input feature maps, patch height, patch width
	train_data = np.reshape(train_data,(train_data.shape[0],3,51,51))
	print('Size of training data is '+ str(train_data.shape))
	train_target = np.squeeze(np.asarray(data['target']))
	print('Size of training target is '+ str(train_target.shape))
	
	#Loading validation data    
	data = scipy.io.loadmat('valid.mat')
	valid_data = np.asarray(data['patches'])
	# Size is number of patches, number of input feature maps, patch height, patch width
	valid_data = np.reshape(valid_data,(valid_data.shape[0],3,51,51))
	print('Size of validation data is '+str(valid_data.shape))
	valid_target = np.squeeze(np.asarray(data['target']))
	print('Size of validation target is '+str(valid_target.shape))
	
	#Loading testing data
	data = scipy.io.loadmat('test.mat')
	test_data = np.asarray(data['patches'])
	# Size is number of patches, number of input feature maps, patch height, patch width
	test_data = np.reshape(test_data,(test_data.shape[0],3,51,51))
	print('Size of testing data is '+ str(test_data.shape))
	test_target = np.squeeze(np.asarray(data['target']))
	print('Size of testing target is '+ str(test_target.shape))

	def shared(data_x, data_y, borrow=True):
		shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
		shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=True)
		return shared_x, T.cast(shared_y, "int32")

	train_set_x, train_set_y = shared(train_data, train_target)
	valid_set_x, valid_set_y = shared(valid_data, valid_target)
	test_set_x, test_set_y = shared(test_data, test_target)

	return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

def DeepNetwork(batch_size=10, learning_rate=0.1, nkerns=[25, 50, 80], n_epochs=10):
	
	flag=0
	print('Loading data')
	
	dataset = load_data()
	
	train_set_x, train_set_y = dataset[0]
	valid_set_x, valid_set_y = dataset[1]
	test_set_x, test_set_y = dataset[2]
	
	# compute number of minibatches for training, validation and testing
	n_train_batches = train_set_x.get_value(borrow=True).shape[0]
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
	n_test_batches = test_set_x.get_value(borrow=True).shape[0]
	n_train_batches //= batch_size
	n_valid_batches //= batch_size
	n_test_batches //= batch_size
	
	
	index = T.lscalar()
	x = T.matrix('x')
	y = T.ivector('y')
	rng = np.random.RandomState(23455)
	
	print('Building layers')
	
	layer0_input = x.reshape((batch_size, 3, 51, 51))
	
	layer0 = ConvPoolLayer(
		rng, input=layer0_input, filter_shape=(nkerns[0], 3, 4, 4), image_shape=(batch_size, 3, 51, 51)
	)
	
	layer1 = ConvPoolLayer(
		rng, input=layer0.output, image_shape=(batch_size, nkerns[0], 24, 24),
		filter_shape=(nkerns[1], nkerns[0], 5, 5)
	)
	
	layer2 = ConvPoolLayer(
		rng, input=layer1.output, image_shape=(batch_size, nkerns[1], 10, 10),
		filter_shape=(nkerns[2], nkerns[1], 5, 5)
	)
	
	layer3_input = layer2.output.flatten(2)
	
	layer3 = FullyConnectedLayer(
		rng, input=layer3_input,
		n_in=nkerns[2]*3*3, n_out=1024
	)
	
	if flag==0:
		layer4_input = layer3.output
	else:
		layer4_input = layer3.output_dropout
	
	layer4 = FullyConnectedLayer(
		rng, input=layer3.output_dropout,
		n_in=1024, n_out=1024
	)
	
	if flag==0:
		layer5_input = layer4.output
	else:
		layer5_input = layer4.output_dropout
		
	layer5 = LogisticRegression(input=layer5_input, n_in=1024, n_out=2)
	
	cost = layer5.negative_log_likelihood(y)
	
	print('Building model')
	
	test_model = theano.function(
		[index],
	    layer5.errors(y),
	    givens={
	        x: test_set_x[index * batch_size: (index + 1) * batch_size],
	        y: test_set_y[index * batch_size: (index + 1) * batch_size]
	    }
	)
	
	validate_model = theano.function(
        [index],
        layer5.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
	params = layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

	grads = T.grad(cost, params)
    
	updates = [
		(param_i, param_i - learning_rate * grad_i)
		for param_i, grad_i in zip(params, grads)
	]

	train_model = theano.function(
		[index],
		cost,
		updates=updates,
		givens={
			x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)
	
	print('... training')
	# early-stopping parameters
	patience = 10000  # look as this many examples regardless
	patience_increase = 2  # wait this much longer when a new best is
			               # found
	improvement_threshold = 0.995  # a relative improvement of this much is
			                       # considered significant
	validation_frequency = min(n_train_batches, patience // 2)
			                      # go through this many
			                      # minibatche before checking the network
			                      # on the validation set; in this case we
			                      # check every epoch

	best_validation_loss = np.inf
	best_iter = 0
	test_score = 0.
	start_time = timeit.default_timer()

	epoch = 0
	done_looping = False

	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
		for minibatch_index in range(n_train_batches):

			iter = (epoch - 1) * n_train_batches + minibatch_index

			if iter % 100 == 0:
			    print('training @ iter = ', iter)
			cost_ij = train_model(minibatch_index)

			if (iter + 1) % validation_frequency == 0:
			    # compute zero-one loss on validation set
			    validation_losses = [validate_model(i) for i
			                         in range(n_valid_batches)]
			    this_validation_loss = np.mean(validation_losses)
			    print('epoch %i, minibatch %i/%i, validation error %f %%' %
			          (epoch, minibatch_index + 1, n_train_batches,
			           this_validation_loss * 100.))

			    # if we got the best validation score until now
			    if this_validation_loss < best_validation_loss:
			        #improve patience if loss improvement is good enough
			        if this_validation_loss < best_validation_loss *  \
			           improvement_threshold:
			            patience = max(patience, iter * patience_increase)

			        # save best validation score and iteration number
			        best_validation_loss = this_validation_loss
			        best_iter = iter

			        # test it on the test set
			        test_losses = [
			            test_model(i)
			            for i in range(n_test_batches)
			        ]
			        test_score = np.mean(test_losses)
			        print(('     epoch %i, minibatch %i/%i, test error of '
			               'best model %f %%') %
			              (epoch, minibatch_index + 1, n_train_batches,
			               test_score * 100.))

			if patience <= iter:
			    done_looping = True
			    break

	end_time = timeit.default_timer()
	print('Optimization complete.')
	print('Best validation score of %f %% obtained at iteration %i, '
		  'with test performance %f %%' %
		  (best_validation_loss * 100., best_iter + 1, test_score * 100.))

	print('The code for file ran for %.2fm' % ((end_time - start_time) / 60.))


def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
