# This file is part of the hIPPYflow package
#
# hIPPYflow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
#
# hIPPYflow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# If not, see <http://www.gnu.org/licenses/>.
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu

import tensorflow as tf
if int(tf.__version__[0]) > 1:
	import tensorflow.compat.v1 as tf
	tf.disable_v2_behavior()

import numpy as np


class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True)
    def call(self, x):
        return x + self.bias

def low_rank_layer(input_x,rank = 8,activation = 'softplus'):
	output_shape = input_x.shape
	assert len(output_shape) == 2
	output_dim = output_shape[-1]
	intermediate = tf.keras.layers.Dense(rank,activation = activation)(input_x)
	return tf.keras.layers.Dense(output_dim)(intermediate)



def projected_low_rank_residual_network(input_projector,output_projector,ranks = [4,4],\
							trainable = False,set_weights = True,random_weights = False):
	input_dim, reduced_input_dim = input_projector.shape
	output_dim, reduced_output_dim = output_projector.shape

	input_data = tf.keras.layers.Input(shape=(input_dim,))

	input_proj_layer = tf.keras.layers.Dense(reduced_input_dim,name = 'input_proj_layer',use_bias = False)(input_data)
	input_proj_layer = BiasLayer()(input_proj_layer)

	# z = low_rank_layer(input_proj_layer)

	z = input_proj_layer
	for rank in ranks:
		z = tf.keras.layers.Add()([low_rank_layer(z,rank = rank,activation = 'sigmoid'),z])



	z = tf.keras.layers.Dense(reduced_output_dim)(z)
	output_layer = tf.keras.layers.Dense(output_dim,name = 'output_layer')(z)

	regressor = tf.keras.models.Model(input_data,output_layer)

	########################################################################
	# Modify input layer by setting weights and setting trainable boolean
	regressor.get_layer('input_proj_layer').trainable =  trainable
	if set_weights:
		if random_weights:
			input_proj_weights = [np.random.randn(*input_projector.shape)]
		else:
			input_proj_weights = [input_projector]
		
		# input_proj_weights.append(np.zeros(input_projector.shape[-1]))
		regressor.get_layer('input_proj_layer').set_weights(input_proj_weights)

	########################################################################
	# Modify output layer by setting weights and setting trainable boolean
	regressor.get_layer('output_layer').trainable =  True
	if set_weights:
		if random_weights:
			output_proj_weights = [np.random.randn(*output_projector.T.shape)]
		else:
			output_proj_weights = [output_projector.T]

		output_proj_weights.append(np.zeros(output_projector.T.shape[-1]))
		regressor.get_layer('output_layer').set_weights(output_proj_weights)

	return regressor




def projected_dense(input_projector,output_projector,intermediate_layers = 1,\
		trainable = False):
	input_dim, reduced_input_dim = input_projector.shape
	output_dim, reduced_output_dim = output_projector.shape
	input_data = tf.keras.layers.Input(shape=(input_dim,))
	input_proj_layer = tf.keras.layers.Dense(reduced_input_dim,name = 'input_proj_layer',use_bias = False)(input_data)
	input_proj_layer = BiasLayer()(input_proj_layer)
	z =  tf.keras.layers.Dense(reduced_input_dim,activation = 'softplus')(input_proj_layer)
	for i in range(intermediate_layers):
		z = tf.keras.layers.Dense(reduced_output_dim,activation = 'softplus')(z)
	output_layer = tf.keras.layers.Dense(output_dim,name = 'output_layer')(z)

	regressor = tf.keras.models.Model(input_data,output_layer)

	regressor.get_layer('input_proj_layer').trainable =  trainable

	regressor.get_layer('output_layer').trainable =  True

	return regressor



def generic_dense(input_dim,output_dim):
	input_data = tf.keras.layers.Input(shape=(input_dim,))
	z = tf.keras.layers.Dense(output_dim, activation='softplus')(input_data)
	# z = tf.keras.layers.Dense(20, activation='softplus')(z)
	z = tf.keras.layers.Dense(output_dim, activation='softplus')(z)
	output = tf.keras.layers.Dense(output_dim)(z)
	regressor = tf.keras.models.Model(input_data, output)
	return regressor

def generic_projected_nn(input_dim,output_dim,project_dim = 98):
	input_data = tf.keras.layers.Input(shape=(input_dim,))
	z = tf.keras.layers.Dense(20, name = 'projection_layer')(input_data)
	z = tf.keras.layers.Dense(20, activation='softplus')(z)
	z = tf.keras.layers.Dense(20, activation='softplus')(z)
	output = tf.keras.layers.Dense(output_dim)(z)
	regressor = tf.keras.models.Model(input_data, output)
	regressor.get_layer('projection_layer').trainable = False
	return regressor

def generic_linear(input_dim,output_dim):
	input_data = tf.keras.layers.Input(shape=(input_dim,))
	output = tf.keras.layers.Dense(output_dim)(input_data)
	regressor = tf.keras.models.Model(input_data, output)
	return regressor


def low_rank_linear(input_dim,output_dim,rank = 16):
	input_data = tf.keras.layers.Input(shape=(input_dim,))
	intermediate = tf.keras.layers.Dense(rank,use_bias = False,name = 'intermediate')(input_data)
	output = tf.keras.layers.Dense(output_dim)(intermediate)
	regressor = tf.keras.models.Model(input_data, output)
	return regressor




