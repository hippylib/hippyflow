# Copyright (c) 2020, The University of Texas at Austin 
# & Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYflow package. For more information see
# https://github.com/hippylib/hippyflow/
#
# hIPPYflow is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.


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

from helmholtz_linear_observable import helmholtz_linear_observable


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-ninstance',dest = 'ninstance',required= False,default = 1,help='number of instances',type = int)
parser.add_argument('-nsubdomain',dest = 'nsubdomain',required= False,default = 1,help='number of partition',type = int)
parser.add_argument('-sample_per',dest = 'sample_per',required= False,default = 32,help='number of samples per instance',type = int)
parser.add_argument('-data_per_process',dest = 'data_per_process',required= False,default = 512,help='number of data generated per instance',type = int)
parser.add_argument('-as_rank',dest = 'as_rank',required= False,default = 128,help='rank for active subspace projectors',type = int)
parser.add_argument('-pod_rank',dest = 'pod_rank',required= False,default = 128,help='rank for POD projectors',type = int)
parser.add_argument('-sqrt_n_obs',dest = 'sqrt_n_obs',required= False,default = 10,help='sqrt of targets for observable',type = int)
parser.add_argument('-nx',dest = 'nx',required= False,default = 32,help='targets for observable',type = int)
parser.add_argument('-ny',dest = 'ny',required= False,default = 32,help='targets for observable',type = int)
parser.add_argument('-gamma',dest = 'gamma',required=False,default = 1.0, help="gamma for matern prior",type=float)
parser.add_argument('-delta',dest = 'delta',required=False,default = 5.0, help="delta for matern prior",type=float)
parser.add_argument('-formulation',dest = 'formulation',required=False,default = 'single_freq', help="formulation name string",type=str)
parser.add_argument('-use_laplace_prior',dest = 'use_laplace_prior',\
					required= False,default = 0,help='boolean for saving of data',type = int)

parser.add_argument('-frequency',dest = 'frequency',required=False,default = 300, help="formulation name string",type=int)

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
parser.add_argument('-save_errors',dest = 'save_errors',\
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
output_directory = 'data/'+args.formulation+'_'+str(args.frequency)+'_n_obs_'+str(args.sqrt_n_obs**2)+'_g'+str(args.gamma)+'_d'+str(args.delta)+'_nx'+str(args.nx)+'/'
if args.use_laplace_prior:
	print(80*'#')
	print('Using Laplace Prior (not a trace class operator)'.center(80))
	output_directory = 'laplace_' + output_directory

os.makedirs(output_directory,exist_ok = True)
save_states_dir = output_directory+'save_states/'

# Instantiate confusion linear observable

box = [0., 0., 3., 3.]
box_pml = [-1., -1., 4., 3.]

mesh = dl.RectangleMesh(mesh_constructor_comm,dl.Point(box_pml[0], box_pml[1]), dl.Point(box_pml[2], box_pml[3]), args.nx, args.ny)

observable_kwargs = {'box': box,'box_pml':box_pml,'sqrt_n_obs':args.sqrt_n_obs,'output_folder':save_states_dir, 'frequency':args.frequency}


observable = helmholtz_linear_observable(mesh,**observable_kwargs)


# Matern Covariance Prior is instantiated here so that each process can sample m.
if args.use_laplace_prior:
	prior = Laplacian2D(observable.problem.Vh[PARAMETER],gamma = args.gamma, delta = args.delta)

else:
	prior = BiLaplacian2D(observable.problem.Vh[PARAMETER],gamma = args.gamma, delta = args.delta)



metadata = {}

# Active Subspace
if args.save_as:
	AS_parameters = ActiveSubspaceParameterList()
	AS_parameters['observable_constructor'] = helmholtz_linear_observable
	AS_parameters['observable_kwargs'] = observable_kwargs
	AS_parameters['output_directory'] = output_directory
	AS_parameters['samples_per_process'] = 32
	AS_parameters['plot_label_suffix'] = r' $\gamma = '+str(args.gamma)+',\enskip \delta = '+str(args.delta)+'$'
	AS_parameters['rank'] = args.as_rank
	AS = ActiveSubspaceProjector(observable,prior, mesh_constructor_comm = mesh_constructor_comm,collective = my_collective,parameters = AS_parameters)	
	AS.construct_input_subspace()
	AS.construct_output_subspace()

	metadata['as_input_time'] = AS._input_subspace_construction_time
	metadata['as_output_time'] = AS._output_subspace_construction_time

# del(AS)

# Karhunen-Lo\`{e}ve Expansion
if args.save_kle:

	KLE_parameters = KLEParameterList()
	KLE_parameters['output_directory'] = output_directory
	KLE_parameters['plot_label_suffix'] = r' $\gamma = '+str(args.gamma)+',\enskip \delta = '+str(args.delta)+'$'
	KLE = KLEProjector(prior,\
		mesh_constructor_comm = mesh_constructor_comm,collective = my_collective,parameters = KLE_parameters)

	KLE.construct_input_subspace()

	metadata['kle_time'] = KLE._subspace_construction_time

# del(KLE)

# Proper Orthogonal Decomposition
if args.save_data or args.save_pod:
	print('POD object being created')
	POD_parameters = PODParameterList()
	POD_parameters['data_per_process']	 = args.data_per_process
	POD_parameters['output_directory'] = output_directory
	POD_parameters['plot_label_suffix'] = r' $\gamma = '+str(args.gamma)+',\enskip \delta = '+str(args.delta)+'$'
	POD = PODProjector(observable,prior,\
		mesh_constructor_comm = mesh_constructor_comm,collective = my_collective,parameters = POD_parameters)


# Save the POD projector 
if args.save_pod:
	print(80*'#')
	print('Building POD projector')
	POD.parameters['rank'] = args.pod_rank
	POD.construct_subspace()

	metadata['pod_time'] = POD._subspace_construction_time

# Test Errors
if args.save_errors:
	import pickle
	def save_logger(logger):
	    with open(output_directory+'error_data.pkl', 'wb+') as f:
	        pickle.dump(logger, f, pickle.HIGHEST_PROTOCOL)
	# Error Data
	error_data = {}
	input_ranks = [8,16,32,64,128]
	error_data['input_ranks'] = input_ranks
	rank_pairs = [(1,1),(2,2),(4,4)] + [(8*i,8*i) for i in range(1,17)]
	error_data['rank_pairs'] = rank_pairs

	if args.save_as:
		print(80*'#')
		print('Testing error for AS input bound'.center(80))
		error_data['AS_input_errors'],error_data['AS_output_errors'] = AS.test_errors(ranks = input_ranks)
		if args.save_pod:
			print(80*'#')
			print('Testing error for AS input-output error bound'.center(80))
			error_data['as_input_output'] = POD.input_output_error_test(AS.V_GN,Cinv = AS.prior.R, rank_pairs = rank_pairs)

	if args.save_kle:
		print(80*'#')
		print('Testing error for KLE input-error bound'.center(80))
		error_data['KLE_input_errors'] = KLE.test_errors(ranks = input_ranks)
		if args.save_pod:
			print(80*'#')
			print('Testing error for KLE input-output error bound'.center(80))
			error_data['kle_input_output'] = POD.input_output_error_test(KLE.V_KLE,Cinv = AS.prior.M, rank_pairs = rank_pairs)

	if True and args.save_pod:
		print(80*'#')
		print('Testing error for random input-output error bound'.center(80))
		Omega = KLE.random_input_projector()
		error_data['random_input_output'] = POD.input_output_error_test(Omega, rank_pairs = rank_pairs)

	save_logger(error_data)

if args.save_data:
	print(80*'#')
	print('Made it to the POD data generation!')
	POD.generate_training_data(output_directory)

	metadata['data_time'] = POD._data_generation_time


if args.save_two_states:
	print(80*'#')
	print('Saving two states')
	POD.two_state_solution()






