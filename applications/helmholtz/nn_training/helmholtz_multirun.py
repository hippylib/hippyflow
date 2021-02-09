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

# Parse run specifications
from argparse import ArgumentParser

parser = ArgumentParser(add_help=True)
parser.add_argument("-optimizer", dest='optimizer',required=False, default = 'incg', help="optimizer type",type=str)
parser.add_argument('-fixed_step',dest = 'fixed_step',\
					required= False,default = 0,help='boolean for fixed step vs globalization',type = int)
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
# Instantiate data

def load_helmholtz_data(data_dir,rescale = False,n_data = np.inf):
	assert os.path.isdir(data_dir)
	data_files = os.listdir(data_dir)
	data_files = [data_dir + file for file in data_files]

	m_files = []
	q_files = []
	for file in data_files:
		if 'ms_on_rank_' in file:
			m_files.append(file)
		if 'qs_on_rank_' in file:
			q_files.append(file)

	ranks = [int(file.split(data_dir+'ms_on_rank_')[-1].split('.npy')[0]) for file in m_files]
	max_rank = max(ranks)

	# Serially concatenate data
	m_data = np.load(data_dir+'ms_on_rank_0.npy')
	q_data = np.load(data_dir+'qs_on_rank_0.npy')
	for i in range(1,max_rank+1):
			appendage_m = np.load(data_dir+'ms_on_rank_'+str(i)+'.npy')
			m_data = np.concatenate((m_data,appendage_m))
			appendage_q = np.load(data_dir+'qs_on_rank_'+str(i)+'.npy')
			q_data = np.concatenate((q_data,appendage_q))

	# print('m_data.shape = ',m_data.shape)
	# print('q_data.shape = ',q_data.shape)

	if n_data < np.inf:
		assert type(n_data) is int
		m_data = m_data[:n_data]
		q_data = q_data[:n_data]
	if rescale:
		from sklearn import preprocessing
		m_data = preprocessing.scale(m_data)
		q_data = preprocessing.scale(q_data)


	# print('m_data.shape = ',m_data.shape)
	# print('q_data.shape = ',q_data.shape)
	return [m_data,q_data]


def get_projectors(data_dir,as_input_tolerance=1e-4,as_output_tolerance=1e-4,\
					kle_tolerance = 1e-4,pod_tolerance = 1e-4,\
					 fixed_input_rank = 0, fixed_output_rank = 0, mixed_output = True, verbose = False):
	projector_dictionary = {}
	################################################################################
	# Derivative Informed Input Subspace
	AS_input_projector = np.load(data_dir+'AS_input_projector.npy')
	if verbose:
		print('AS input projector shape before truncation = ', AS_input_projector.shape)
	if fixed_input_rank > 0:
		AS_input_projector = AS_input_projector[:,:fixed_input_rank]
	else:
		AS_input_d = np.load(data_dir+'AS_d_GN.npy')
		AS_input_projector = AS_input_projector[:,np.where(AS_input_d>as_input_tolerance)[0]]
	if verbose:
		print('AS input projector shape after truncation = ', AS_input_projector.shape)
	projector_dictionary['AS_input'] = AS_input_projector
	################################################################################
	# Derivative Informed Output Subspace
	AS_output_projector = np.load(data_dir+'AS_output_projector.npy')
	
	if verbose:
		print('AS output projector shape before truncation = ', AS_output_projector.shape)
	if fixed_output_rank > 0:
		AS_output_projector = AS_output_projector[:,:fixed_output_rank]
	else:
		AS_output_d = np.load(data_dir+'AS_d_NG.npy')
		AS_output_projector = AS_output_projector[:,np.where(AS_output_d>as_output_tolerance)[0]]
	if verbose:
		print('AS output projector shape after truncation = ', AS_output_projector.shape)
	projector_dictionary['AS_output'] = AS_output_projector
	################################################################################
	# KLE Input Subspace
	KLE_projector = np.load(data_dir+'KLE_projector.npy')
	if verbose:
		print('KLE projector shape before truncation = ', KLE_projector.shape)
	if fixed_input_rank > 0:
		KLE_projector = KLE_projector[:,:fixed_input_rank]
	else:
		KLE_d = np.load(data_dir+'KLE_d.npy')
		KLE_projector = KLE_projector[:,np.where(KLE_d>kle_tolerance)[0]]
	if verbose:
		print('KLE projector shape after truncation = ', KLE_projector.shape)
	projector_dictionary['KLE'] = KLE_projector
	################################################################################
	# POD Output Subspace
	POD_projector = np.load(data_dir+'POD_projector.npy')
	if verbose:
		print('POD projector shape before truncation = ', POD_projector.shape)
	if fixed_output_rank > 0:
		POD_projector = POD_projector[:,:fixed_output_rank]
	else:
		POD_d = np.load(data_dir+'POD_d.npy')
		POD_projector = POD_projector[:,np.where(POD_d>pod_tolerance)[0]]
	if verbose:
		print('POD projector shape after truncation = ', POD_projector.shape)
	projector_dictionary['POD'] = POD_projector
	return projector_dictionary

def modify_projectors(projectors,input_subspace,output_subspace):
	# Modify the input projectors
	assert input_subspace in ['kle','as','random']

	if input_subspace in ['kle','as']:
		# Always orthogonalize AS and KLE for best results
		orthogonalize_input = True
		rescale_input = True
		if input_subspace == 'kle':
			input_projector = projectors['KLE']
		elif input_subspace == 'as':
			input_projector = projectors['AS_input']


		if orthogonalize_input:
			input_projector,_ = np.linalg.qr(input_projector)

		if rescale_input:
			# Scaling factor of 10 seemed to perform well for KLE and AS
			# and this was independent of the projector rank.
			scale_factor_input = 0.05*float(input_projector.shape[0])/(32*float(input_projector.shape[-1]))
			input_projector /= scale_factor_input*np.linalg.norm(input_projector)

	elif input_subspace == 'random':
		input_projector = np.random.randn(*projectors['KLE'].shape)
		input_projector,_ = np.linalg.qr(input_projector)
		scale_factor_input = 0.05*float(input_projector.shape[0])/(32*float(input_projector.shape[-1]))
		input_projector /= scale_factor_input*np.linalg.norm(input_projector)

	# Modify the output projectors
	# It seems that (re)-orthogonalizing the POD vectors
	# may not improve the neural network.
	assert output_subspace in ['pod','as','random']

	if output_subspace in ['pod','as']:
		orthogonalize_output = True
		rescale_output = True
		if output_subspace == 'pod':
			output_projector = projectors['POD']
		elif output_subspace == 'as':
			output_projector = projectors['AS_output']

		if orthogonalize_output:
			output_projector,_ = np.linalg.qr(output_projector)

		if rescale_output:
			scale_factor_output = 1.
			output_projector /= scale_factor_output*np.linalg.norm(output_projector)

	if output_subspace == 'random':
		output_projector = np.random.randn(*projectors['POD'].shape)
		output_projector /= np.linalg.norm(output_projector)

	return input_projector, output_projector

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
		m_data, q_data = load_helmholtz_data(data_dir,rescale = False,n_data = n_data)

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
			# Instante the data object
			data = Data([m_data,q_data],settings['batch_size'],\
				test_data_size = settings['test_data_size'],hessian_batch_size = settings['hess_batch_size'],seed = data_seed)

			################################################################################
			# Instantiate the problem, regularization.
			q_mean = np.mean(q_data,axis = 0)
			problem = RegressionProblem(regressor,y_mean = q_mean,dtype=tf.float32)

			settings['tikhonov_gamma'] = 0.0

			regularization = L2Regularization(problem,gamma = settings['tikhonov_gamma'])

			################################################################################
			# Instantiate the model object
			HLModelSettings = HessianlearnModelSettings()

			HLModelSettings['optimizer'] = args.optimizer
			HLModelSettings['alpha'] = args.alpha
			HLModelSettings['fixed_step'] = args.fixed_step
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

			HLModelSettings['printing_items'] = {'sweeps':'sweeps','Loss':'loss_train','acc train':'accuracy_train',\
															'||g||':'||g||','Loss test':'loss_test','acc test':'accuracy_test',\
															'maxacc test':'max_accuracy_test','var red':'variance_reduction','alpha':'alpha'}

			HLModel = HessianlearnModel(problem,regularization,data,settings = HLModelSettings)

			HLModel.fit()
			master_logger[architecture+'_'+str(n_data)][args.weight_seed][data_seed] = HLModel._logger

			save_logger(master_logger,architecture)