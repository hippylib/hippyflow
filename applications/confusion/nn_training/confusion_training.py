import numpy as np
import os
import tensorflow as tf
import time
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

tf.set_random_seed(1)

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

parser.add_argument('-fixed_input_rank',dest = 'fixed_input_rank',required= False,default = 16,help='fixed input rank',type = int)
parser.add_argument('-fixed_ouput_rank',dest = 'fixed_output_rank',required= False,default = 16,help='fixed input rank',type = int)

parser.add_argument("-architecture", dest='architecture',required=False, default = 'projected_dense', help="architecture type",type=str)

parser.add_argument('-test_data_size',dest = 'test_data_size',required= False,default = 512,help='test data size',type = int)
parser.add_argument('-train_data_size',dest = 'train_data_size',required= False,default = 512+256,help='train data size',type = int)

parser.add_argument('-gamma',dest = 'gamma',required= False,default = 0.5,\
						help='Matern prior gamma, (delta I - gamma Laplacian)',type = float)
parser.add_argument('-delta',dest = 'delta',required= False,default = 0.5,\
						help='Matern prior delta, (delta I - gamma Laplacian)',type = float)

parser.add_argument('-nx',dest = 'nx',required= False,default = 192,help='Mesh discretization parameter',type = int)
args = parser.parse_args()


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

def load_confusion_data(data_dir,rescale = False,n_data = np.inf):
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

	if n_data < np.inf:
		assert type(n_data) is int
		m_data = m_data[:n_data]
		q_data = q_data[:n_data]
	if rescale:
		from sklearn import preprocessing
		m_data = preprocessing.scale(m_data)
		q_data = preprocessing.scale(q_data)

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

# def check_subspace_angles(projector_dictionary):
# 	KL = projector_dictionary['']


ntargets = 100
gamma = args.gamma
delta = args.delta

nx = args.nx

n_data = settings['train_data_size'] + settings['test_data_size']

data_dir = '../data/n_obs_'+str(ntargets)+'_g'+str(gamma)+'_d'+str(delta)+'_nx'+str(nx)+'/'

assert os.path.isdir(data_dir), 'Directory does not exist'

m_data, q_data = load_confusion_data(data_dir,rescale = False,n_data = n_data)

input_dim = m_data.shape[-1]
output_dim = q_data.shape[-1]

# Instante the data object
data = Data([m_data,q_data],settings['batch_size'],test_data_size = settings['test_data_size'],hessian_batch_size = settings['hess_batch_size'])


################################################################################
# Create the neural network in keras

# Load the input and output projectors
fixed_input_rank = args.fixed_input_rank
fixed_output_rank = args.fixed_output_rank

projectors = get_projectors(data_dir,fixed_input_rank = fixed_input_rank,fixed_output_rank = fixed_output_rank)


# Modify the input projectors
input_subspace = 'kle'

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
		scale_factor_input = 10.
		input_projector /= scale_factor_input*np.linalg.norm(input_projector)

elif input_subspace == 'random':
	input_projector = np.random.randn(*projectors['KLE'].shape)
	input_projector /= np.linalg.norm(input_projector)

# Modify the output projectors
# It seems that (re)-orthogonalizing the POD vectors
# may not improve the neural network 
output_subspace = 'pod'

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

################################################################################

architecture = args.architecture

assert architecture in ['generic_dense','generic_projected','generic_linear','low_rank_linear','projected_dense','plrrn']


if architecture == 'generic_dense':
	regressor = generic_dense(input_dim,output_dim)

elif architecture == 'generic_linear':
	regressor = generic_linear(input_dim,output_dim)

elif architecture == 'projected_dense':
	trainable = False
	intermediate_layers = 2
	regressor = projected_dense(input_projector,output_projector,intermediate_layers = intermediate_layers,\
							trainable = trainable)

elif architecture == 'low_rank_linear':
	regressor = low_rank_linear(input_dim,output_dim,rank = input_projector.shape[-1])


regressor.summary()

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

HLModelSettings['problem_name'] = 'confusion_nt_'+str(ntargets)+'_g_'+str(gamma)+'_d_'+str(delta)+'_nx_'+str(nx)
HLModelSettings['record_spectrum'] = bool(args.record_spectrum)
HLModelSettings['rq_data_size'] = 100

set_weights = True
if architecture == 'projected_dense' and set_weights:
	HLModelSettings['layer_weights'] = {'input_proj_layer':[input_projector],\
					'output_layer':[output_projector.T,np.zeros(output_projector.T.shape[-1])]}

HLModelSettings['printing_items'] = {'sweeps':'sweeps','Loss':'loss_train','acc train':'accuracy_train',\
												'||g||':'||g||','Loss test':'loss_test','acc test':'accuracy_test',\
												'maxacc test':'max_accuracy_test','var red':'variance_reduction','alpha':'alpha'}

HLModel = HessianlearnModel(problem,regularization,data,settings = HLModelSettings)

HLModel.fit()


# if architecture in ['projected_dense']:
# 	final_input_proj = regressor.get_layer('input_proj_layer').get_weights()[0]
# 	print('ERROR input = ',np.linalg.norm(final_input_proj - input_projector))
# 	final_output_proj = regressor.get_layer('output_layer').get_weights()[0]
# 	print('ERROR output = ',np.linalg.norm(final_output_proj - output_projector.T))

