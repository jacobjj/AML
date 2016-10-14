import os
import scipy.io
import theano
import numpy as np
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import softmax

rng = np.random.RandomState(23455)

def ReLU(z): return T.maximum(0.0, z)

def load_data():

	#Loading validation data    
	data = scipy.io.loadmat('valid_data.mat')
	valid_data = np.asarray(data['patches'])
	# Size is number of patches, number of input feature maps, patch height, patch width
	valid_data = np.reshape(valid_data,(valid_data.shape[0],3,51,51))
	print('Size of validation data is '+str(valid_data.shape))
	valid_target = np.asarray(data['target'])
	print('Size of validation target is '+str(valid_target.shape))

	#Loading training data  
	data = scipy.io.loadmat('train_data.mat')
	train_data = np.asarray(data['patches'])
	# Size is number of patches, number of input feature maps, patch height, patch width
	train_data = np.reshape(train_data,(train_data.shape[0],3,51,51))
	print('Size of training data is '+ str(train_data.shape))
	train_target = np.asarray(data['target'])
	print('Size of training target is '+ str(train_target.shape))

	#Loading testing data
	data = scipy.io.loadmat('test_data.mat')
	test_data = np.asarray(data['patches'])
	# Size is number of patches, number of input feature maps, patch height, patch width
	test_data = np.reshape(test_data,(test_data.shape[0],3,51,51))
	print('Size of testing data is '+ str(test_data.shape))
	test_target = np.asarray(data['target'])
	print('Size of testing target is '+ str(test_target.shape))
	
	def shared():
		shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return T.cast(shared_x, "int32"), T.cast(shared_y, "int32")

    return [shared(train_data), shared(valid_data), shared(test_data)]

class DeepNetwork(object):
	
	def __init__(self, mini_batch_size):
		self.mini_batch_size = mini_batch_size
		self.x = T.matrix('x')
		self.y = T.ivector('y')
		
		layer0_input = x.reshape((mini_batch_size, 3, 51, 51))
		
		layer0 = ConvPoolLayer(
			rng, inpt=layer0_input, image_shape=(mini_batch_size, 3, 51, 51),
			filter_shape(nkerns[0], 3, 4, 4)
		)
		
		layer1 = ConvPoolLayer(
			rng, inpt=layer0.output, image_shape(mini_batch_size, nkerns[0], 24, 24),
			filter_shape=(nkerns[1], nkerns[0], 5, 5)
		)
		
		layer2 = ConvPoolLayer(
			rng, inpt=layer1.output, image_shape(mini_batch_size, nkerns[1], 10, 10),
			filter_shape=(nkerns[2], nkerns[1], 5, 5)
		)
		
		layer3_input = layer2.output.flatten(2)
		
		layer3 = FullyConnectedLayer(
			rng, inpt=layer3_input, inpt_dropout=layer3_input,
			n_in=nkerns[2]*3*3, n_out=1024
		)
		
		layer4 = FullyConnectedLayer(
			rng, inpt=layer3.output, inpt_dropout=layer3.output,
			n_in=1024, n_out=1024
		)
		
		layer5 = LogisticRegression(input=layer4.output, n_in=1024, n_out=2)
		
		cost = layer5.negative_log_likelihood(y)

class ConvPoolLayer(object):

	def __init__(self, rng, inpt, filter_shape, image_shape, poolsize=(2, 2),
				activation_fn=ReLU):
		self.inpt = inpt
		self.filter_shape = filter_shape
		self.image_shape = image_shape
		self.poolsize = poolsize
		self.activation_fn = activation_fn
		
		n_in = numpy.prod(filter_shape[1:])
		n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
		self.W = theano.shared(
				np.asarray(rng.uniform(low=-np.sqrt(6/n_in+n_out), high=np.sqrt(6/n_in+n_out),
				size=filter_shape), dtype=theano.config.floatX), borrow=True)
		
		b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, borrow=True)
		
		conv_out = conv2d(
			inpt = inpt,
			filters=self.W
			filter_shape=self.filter_shape,
			input_shape=self.image_shape
		)
		
		pooled_out = pool.pool_2d(
			inpt=conv_out,
			ds=self.poolsize,
			ignore_border=True
		)
		
		self.output = self.activation_fn(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		
		self.params = [self.W, self.b]

class FullyConnectedLayer(object):
	
	def __init__(self, rng, inpt, inpt_dropout, n_in, n_out,
				activation_fn=ReLU, p_dropout=0.5):
		self.rng = rng
		self.inpt = inpt
		self.inpt_dropout = inpt_dropout
		self.n_in = n_in
		self.n_out = n_out
		self.activation_fn = activation_fn
		self.p_dropout=p_dropout
		
		self.W = theano.shared(np.asarray(rng.uniform(low=, high=, size=(n_in, n_out)),
				dtype=theano.config.floatX), name='W', borrow=True)
		
		self.b = theano.shared(np.asarray(rng.uniform(low=, high=, size=(n_out,)),
				dtype=theano.config.floatX), name='b', borrow=True)
		
		self.output = self.activation_fn((1-self.p_dropout)*T.dot(self.inpt, self.W) + self.b)
		self.y_out = T.argmax(self.output, axis=1)
		self.inpt_dropout = dropout_layer(inpt_dropout, self.p_dropout)
		self.output_dropout = self.activation_fn(T.dot(self.inpt_dropout, self.W) + self.b)

class SoftmaxLayer(object):
