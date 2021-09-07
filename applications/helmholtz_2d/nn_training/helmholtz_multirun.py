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

import numpy as np
import os
import tensorflow as tf
import time
import pickle
# if int(tf.__version__[0]) > 1:
# 	import tensorflow.compat.v1 as tf
# 	tf.disable_v2_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["KMP_WARNINGS"] = "FALSE" 
import sys
sys.path.append( os.environ.get('HESSIANLEARN_PATH'))
from hessianlearn import *

from neuralNetworks import *

from helmholtz_utilities import *


# Parse run specifications
from argparse import ArgumentParser

parser = ArgumentParser(add_help=True)
parser.add_argument("-optimizer", dest='optimizer',required=False, default = 'incg', help="optimizer type",type=str)
parser.add_argument('-globalization',dest = 'globalization',\
					required= False,default = 'line_search',help='boolean for fixed step vs globalization',type = str)
parser.add_argument('-alpha',dest = 'alpha',required = False,default = 1e-3,help= 'learning rate alpha',type=float)
parser.add_argument('-hessian_low_rank',dest = 'hessian_low_rank',required= False,default = 20,help='low rank for sfn',type = int)
parser.add_argument('-record_spectrum',dest = 'record_spectrum',\
					required= False,default = 0,help='boolean for recording spectrum',type = int)

parser.add_argument('-batch_size',dest = 'batch_size',required= False,default = 128,help='batch size',type = int)
parser.add_argument('-hess_batch_size',dest = 'hess_batch_size',required= False,default = 16,help='hess batch size',type = int)
parser.add_argument('-max_sweeps',dest = 'max_sweeps',required= False,default = 100,help='max sweeps',type = float)

parser.add_argument('-fixed_input_rank',dest = 'fixed_input_rank',required= False,default = 8,help='fixed input rank',type = int)
parser.add_argument('-fixed_output_rank',dest = 'fixed_output_rank',required= False,default = 16,help='fixed input rank',type = int)

parser.add_argument("-architecture", dest='architecture',required=False, default = 'as_projected_dense', help="architecture type",type=str)

parser.add_argument('-test_data_size',dest = 'test_data_size',required= False,default = 512,help='test data size',type = int)
parser.add_argument('-train_data_size',dest = 'train_data_size',required= False,default = 512,help='train data size',type = int)

# Random seed for weight initialization
parser.add_argument('-weight_seed',dest = 'weight_seed',required= False,default = 0,help='tf seed',type = int)

parser.add_argument('-gamma',dest = 'gamma',required= False,default = 1.0,\
						help='Matern prior gamma, (delta I - gamma Laplacian)',type = float)
parser.add_argument('-delta',dest = 'delta',required= False,default = 5.0,\
						help='Matern prior delta, (delta I - gamma Laplacian)',type = float)

parser.add_argument('-nx',dest = 'nx',required= False,default = 128,help='Mesh discretization parameter',type = int)
args = parser.parse_args()


# Set random_seed 
tf.set_random_seed(args.weight_seed)


settings = {}
# Set run specifications
# Data specs

settings['test_data_size'] = args.test_data_size
settings['train_data_size'] = args.train_data_size
settings['batch_size'] = args.batch_size
settings['hess_batch_size'] = args.hess_batch_size

if args.train_data_size <= 256:
	settings['batch_size'] = int(args.batch_size/4)
	settings['hess_batch_size'] = int(args.batch_size/32)

################################################################################

ntargets = 100
gamma = args.gamma
delta = args.delta

freq = 600

for nx in [64,128]:
	for i in range(5):
		print(80*'#')
	print(('Running for nx = '+str(nx)).center(80))
	t0 = time.time()
	data_dir = '../data/single_freq_'+str(freq)+'_n_obs_'+str(ntargets)+'_g'+str(gamma)+'_d'+str(delta)+'_nx'+str(nx)+'/'
	assert os.path.isdir(data_dir), 'Directory does not exist: '+data_dir
	problem_name = 'helmholtz_'+str(freq)+'_nt_'+str(ntargets)+'_g_'+str(gamma)+'_d_'+str(delta)+'_nx_'+str(nx)

	if not os.path.isdir(problem_name+'_logging/'):
		os.makedirs(problem_name+'_logging/')
		
	def save_logger(logger,filename):
	    with open(problem_name+'_logging/'+ filename +'.pkl', 'wb+') as f:
	        pickle.dump(logger, f, pickle.HIGHEST_PROTOCOL)

	def unpickle(path):
	    logger = open(path, 'rb')
	    return pickle.load(logger)

	# Save 
	architecture = args.architecture
	assert architecture in ['generic_dense','generic_linear','kle_projected_dense',\
			'as_projected_dense','random_projected_dense','low_rank_linear']
	if 'projected' in architecture:
		architecture += '_'+str(args.fixed_input_rank)+'-'+str(args.fixed_output_rank)

	if os.path.exists(problem_name+'_logging/'+ architecture +'.pkl'):
		master_logger = unpickle(problem_name+'_logging/'+ architecture +'.pkl')
	else:
		print('Did not load the existing logger correctly')
		master_logger = {}

	for n_data in [32,64,128,256,512,768,1024,1280,1536]:
	# for n_data in [512]:

		if n_data <= 256:
			settings['batch_size'] = int(n_data/4)
			settings['hess_batch_size'] = int(n_data/16)
		else:
			settings['batch_size'] = args.batch_size
			settings['hess_batch_size'] = args.hess_batch_size


		n_data += settings['test_data_size']

		helmholtz_data = load_helmholtz_data(data_dir,rescale = False,n_data = n_data)

		m_data = helmholtz_data['m_data']
		q_data = helmholtz_data['q_data']


		input_dim = m_data.shape[-1]
		output_dim = q_data.shape[-1]

		################################################################################
		# Create the neural network in keras

		# Load the input and output projectors
		fixed_input_rank = args.fixed_input_rank
		fixed_output_rank = args.fixed_output_rank

		projectors = get_projectors(data_dir,fixed_input_rank = fixed_input_rank,fixed_output_rank = fixed_output_rank)

			################################################################################
		

		if 'generic_dense' in architecture:
			regressor = generic_dense(input_dim,output_dim)
			# regressor.summary()
			print('Using generic dense network'.center(80))

		elif architecture == 'generic_linear':
			regressor = generic_linear(input_dim,output_dim)
			# regressor.summary()
			print('Using generic linear network'.center(80))

		elif 'kle_projected_dense' in architecture:
			input_subspace = 'kle'
			output_subspace = 'pod'
			input_projector,output_projector = modify_projectors(projectors,input_subspace,output_subspace)
			trainable = False
			intermediate_layers = 2
			regressor = projected_dense(input_projector,output_projector,intermediate_layers = intermediate_layers,\
									trainable = trainable)
			# regressor.summary()
			print('Using KLE dense network'.center(80))

		elif 'as_projected_dense' in architecture:
			input_subspace = 'as'
			output_subspace = 'pod'
			input_projector,output_projector = modify_projectors(projectors,input_subspace,output_subspace)
			trainable = False
			intermediate_layers = 2
			regressor = projected_dense(input_projector,output_projector,intermediate_layers = intermediate_layers,\
									trainable = trainable)
			# regressor.summary()
			print('Using AS dense network'.center(80))

		elif 'random_projected_dense' in architecture:
			input_subspace = 'random'
			output_subspace = 'random'
			input_projector,output_projector = modify_projectors(projectors,input_subspace,output_subspace)
			trainable = False
			intermediate_layers = 2
			regressor = projected_dense(input_projector,output_projector,intermediate_layers = intermediate_layers,\
									trainable = trainable)
			# regressor.summary()
			print('Using random projected dense network'.center(80))


		elif 'low_rank_linear' in architecture:
			regressor = low_rank_linear(input_dim,output_dim,rank = fixed_input_rank)
			# regressor.summary()
			print('Using low rank linear network'.center(80))
		else:
			raise 

		if not architecture+'_'+str(n_data) in master_logger.keys():
			master_logger[architecture+'_'+str(n_data)] = {}

		if args.weight_seed not in master_logger[architecture+'_'+str(n_data)].keys():
			master_logger[architecture+'_'+str(n_data)][args.weight_seed] = {}

		for data_seed in range(1):
			print(80*'#')
			print(('Running for data seed = '+str(data_seed)).center(80))

			################################################################################
			# Instantiate the problem, data and regularization.
			q_mean = np.mean(q_data,axis = 0)
			problem = RegressionProblem(regressor,y_mean = q_mean,dtype=tf.float32)

			# Instante the data object
			data = Data({problem.x:m_data,problem.y_true:q_data},settings['batch_size'],\
				validation_data_size = settings['test_data_size'],hessian_batch_size = settings['hess_batch_size'])


			settings['tikhonov_gamma'] = 0.0

			regularization = L2Regularization(problem,gamma = settings['tikhonov_gamma'])

			################################################################################
			# Instantiate the model object
			HLModelSettings = HessianlearnModelSettings()

			HLModelSettings['optimizer'] = args.optimizer
			HLModelSettings['alpha'] = args.alpha
			if args.globalization == 'None':
				HLModelSettings['globalization'] = None
			else:
				HLModelSettings['globalization'] = args.globalization
			HLModelSettings['hessian_low_rank'] = args.hessian_low_rank
			HLModelSettings['max_backtrack'] = 16
			HLModelSettings['max_sweeps'] = args.max_sweeps

			HLModelSettings['problem_name'] = 'helmholtz_'+str(freq)+'_nt_'+str(ntargets)+'_g_'+str(gamma)+'_d_'+str(delta)+'_nx_'+str(nx)
			HLModelSettings['record_spectrum'] = bool(args.record_spectrum)
			HLModelSettings['rq_data_size'] = 100

			set_weights = True
			if 'projected_dense' in architecture and set_weights:
				HLModelSettings['layer_weights'] = {'input_proj_layer':[input_projector],\
								'output_layer':[output_projector.T,np.zeros(output_projector.T.shape[-1])]}

			HLModelSettings['printing_items'] = {'sweeps':'sweeps','Loss':'train_loss','acc train':'train_acc',\
												'||g||':'||g||','Loss test':'val_loss','acc test':'val_acc',\
												'maxacc':'max_val_acc','var red':'val_variance_reduction','alpha':'alpha'}


			HLModel = HessianlearnModel(problem,regularization,data,settings = HLModelSettings)

			HLModel.fit()
			master_logger[architecture+'_'+str(n_data)][args.weight_seed][data_seed] = HLModel._logger

			save_logger(master_logger,architecture)