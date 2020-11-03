import dolfin as dl
dl.set_log_active(False)
import numpy as np
from mpi4py import MPI
import time
import pickle
import argparse
from pympler import asizeof


import sys, os
sys.path.append( os.environ.get('HIPPYLIB_PATH'))
from hippylib import *
sys.path.append( os.environ.get('HIPPYFLOW_PATH'))
from hippyflow import *


from maternPrior import BiLaplacian2D, Laplacian2D
from confusion_linear_observable import confusion_linear_observable


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-ninstance',dest = 'ninstance',required= False,default = 1,help='number of instances',type = int)
parser.add_argument('-nsubdomain',dest = 'nsubdomain',required= False,default = 1,help='number of partition',type = int)
parser.add_argument('-sample_per',dest = 'sample_per',required= False,default = 32,help='number of samples per instance',type = int)
parser.add_argument('-data_per_process',dest = 'data_per_process',required= False,default = 512,help='number of data generated per instance',type = int)
parser.add_argument('-as_rank',dest = 'as_rank',required= False,default = 128,help='rank for active subspace projectors',type = int)
parser.add_argument('-pod_rank',dest = 'pod_rank',required= False,default = 128,help='rank for POD projectors',type = int)
parser.add_argument('-sqrt_n_obs',dest = 'sqrt_n_obs',required= False,default = 10,help='targets for observable',type = int)
parser.add_argument('-nx',dest = 'nx',required= False,default = 32,help='targets for observable',type = int)
parser.add_argument('-ny',dest = 'ny',required= False,default = 32,help='targets for observable',type = int)
parser.add_argument('-gamma',dest = 'gamma',required=False,default = 1.0, help="gamma for matern prior",type=float)
parser.add_argument('-delta',dest = 'delta',required=False,default = 2.0, help="delta for matern prior",type=float)
parser.add_argument('-formulation',dest = 'formulation',required=False,default = 'cubic_nonlinearity', help="formulation name string",type=str)
parser.add_argument('-save_data',dest = 'save_data',\
					required= False,default = 1,help='boolean for saving of data',type = int)
parser.add_argument('-save_pod',dest = 'save_pod',\
					required= False,default = 1,help='boolean for saving of POD projectors',type = int)
parser.add_argument('-save_as',dest = 'save_as',\
					required= False,default = 1,help='boolean for saving of active subspace projectors',type = int)
parser.add_argument('-save_kle',dest = 'save_kle',\
					required= False,default = 1,help='boolean for saving of KLE projectors',type = int)
parser.add_argument('-save_two_states',dest = 'save_two_states',\
					required= False,default = 1,help='boolean for savign solution at mean and draw',type = int)

args = parser.parse_args()

# Check parallelism things
world = MPI.COMM_WORLD 
world_size = world.size
assert world_size == args.nsubdomain*args.ninstance 
my_rank = world.rank
if my_rank == 0:
	print(('World size is '+str(world_size)).center(80))

# Instantiate split communicators and collectives
mesh_constructor_comm, collective_comm = splitCommunicators(world,args.nsubdomain,args.ninstance)

my_collective = MultipleSamePartitioningPDEsCollective(collective_comm)

# Initialize directories for saving data
output_directory = 'data/'+args.formulation+'_n_obs_'+str(args.sqrt_n_obs**2)+'_g'+str(args.gamma)+'_d'+str(args.delta)+'_nx'+str(args.nx)+'/'
os.makedirs(output_directory,exist_ok = True)
save_states_dir = output_directory+'save_states/'

# Instantiate confusion linear observable
mesh = dl.UnitSquareMesh(mesh_constructor_comm, args.nx, args.ny)
observable_kwargs = {'sqrt_n_obs':args.sqrt_n_obs,'output_folder':save_states_dir}
observable = confusion_linear_observable(mesh,**observable_kwargs)
# Matern Covariance Prior is instantiated here so that each process can sample m.
Vh = observable.problem.Vh
prior = matern_prior_2d(Vh[PARAMETER],gamma = args.gamma, delta = args.delta)

# Active Subspace
if args.save_as:
	AS_parameters = ActiveSubspaceParameterList()
	AS_parameters['observable_constructor'] = confusion_linear_observable
	AS_parameters['observable_kwargs'] = observable_kwargs
	AS_parameters['output_directory'] = output_directory
	AS_parameters['plot_label_suffix'] = r' $\gamma = '+str(args.gamma)+',\enskip \delta = '+str(args.delta)+'$'
	AS_parameters['rank'] = args.as_rank
	AS = ActiveSubspaceProjector(observable,prior, mesh_constructor_comm = mesh_constructor_comm,collective = my_collective,parameters = AS_parameters)	
	AS.construct_input_subspace()
	AS.construct_output_subspace()

# Karhunen-Lo\`{e}ve Expansion
if args.save_kle:

	KLE_parameters = KLEParameterList()
	KLE_parameters['output_directory'] = output_directory
	KLE_parameters['plot_label_suffix'] = r' $\gamma = '+str(args.gamma)+',\enskip \delta = '+str(args.delta)+'$'
	KLE = KLEProjector(prior,\
		mesh_constructor_comm = mesh_constructor_comm,collective = my_collective,parameters = KLE_parameters)

	KLE.construct_input_subspace()


# Proper Orthogonal Decomposition
if args.save_data or args.save_pod:
	print('POD object being created')
	POD_parameters = PODParameterList()
	POD_parameters['data_per_process']	 = args.data_per_process
	POD_parameters['output_directory'] = output_directory
	POD_parameters['plot_label_suffix'] = r' $\gamma = '+str(args.gamma)+',\enskip \delta = '+str(args.delta)+'$'
	POD = PODProjector(observable,prior,\
		mesh_constructor_comm = mesh_constructor_comm,collective = my_collective,parameters = POD_parameters)

if args.save_data:
	print(80*'#')
	print('Made it to the POD data generation!')
	POD.generate_training_data(output_directory)

# Save the POD projector 
if args.save_pod:
	print(80*'#')
	print('Building POD projector')
	POD.parameters['rank'] = args.pod_rank
	POD.construct_subspace()

if args.save_two_states:
	print(80*'#')
	print('Saving two states')
	POD.two_state_solution()






