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

def ReLU(z):
	return T.maximum(0.0, z)

class ConvPoolLayer(object):

	def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2),
				activation_fn=ReLU):
		self.input = input
		self.filter_shape = filter_shape
		self.image_shape = image_shape
		self.poolsize = poolsize
		self.activation_fn = activation_fn
	
		n_in = np.prod(filter_shape[1:])
		n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
		self.W = theano.shared(np.asarray(rng.uniform(low=-np.sqrt(6/n_in+n_out),
				high=np.sqrt(6/n_in+n_out), size=filter_shape),
				dtype=theano.config.floatX), borrow=True)
	
		b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, borrow=True)
	
		conv_out = conv2d(
			input = input,
			filters=self.W,
			filter_shape=self.filter_shape,
			input_shape=self.image_shape
		)
	
		pooled_out = pool.pool_2d(
			input=conv_out,
			ds=self.poolsize,
			ignore_border=True
		)
	
		self.output = self.activation_fn(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
	
		self.params = [self.W, self.b]
