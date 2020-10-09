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


from matern_prior_2d import matern_prior_2d
from confusion_linear_observable import confusion_linear_observable


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-ninstance',dest = 'ninstance',required= False,default = 1,help='number of instances',type = int)
parser.add_argument('-nsubdomain',dest = 'nsubdomain',required= False,default = 1,help='number of partition',type = int)
parser.add_argument('-sample_per',dest = 'sample_per',required= False,default = 32,help='number of samples per instance',type = int)
parser.add_argument('-data_per_process',dest = 'data_per_process',required= False,default = 1024,help='number of data generated per instance',type = int)
parser.add_argument('-as_rank',dest = 'as_rank',required= False,default = 128,help='rank for active subspace projectors',type = int)
parser.add_argument('-pod_rank',dest = 'pod_rank',required= False,default = 128,help='rank for POD projectors',type = int)
parser.add_argument('-n_obs',dest = 'n_obs',required= False,default = 100,help='targets for observable',type = int)
parser.add_argument('-nx',dest = 'nx',required= False,default = 32,help='targets for observable',type = int)
parser.add_argument('-ny',dest = 'ny',required= False,default = 32,help='targets for observable',type = int)
parser.add_argument('-gamma',dest = 'gamma',required=False,default = 1.0, help="gamma for matern prior",type=float)
parser.add_argument('-delta',dest = 'delta',required=False,default = 2.0, help="delta for matern prior",type=float)


# parser.add_argument('-plot_spectra',dest = 'plot_spectra',\
# 					required= False,default = 0,help='boolean for plotting of spectra',type = int)
parser.add_argument('-save_data',dest = 'save_data',\
					required= False,default = 0,help='boolean for saving of data',type = int)
parser.add_argument('-save_pod',dest = 'save_pod',\
					required= False,default = 0,help='boolean for saving of POD projectors',type = int)
parser.add_argument('-save_as',dest = 'save_as',\
					required= False,default = 1,help='boolean for saving of active subspace projectors',type = int)
parser.add_argument('-save_two_solutions',dest = 'save_two_solutions',\
					required= False,default = 0,help='boolean for savign solution at mean and draw',type = int)

args = parser.parse_args()

world = MPI.COMM_WORLD 

world_size = world.size 
my_rank = world.rank

assert world_size == args.nsubdomain*args.ninstance

mesh_constructor_comm, collective_comm = splitCommunicators(world,args.nsubdomain,args.ninstance)

my_collective = MultipleSamePartitioningPDEsCollective(collective_comm)

if my_rank == 0:
	print('World size is ',world_size)

# Initialize directories for saving data
output_directory = 'data/n_obs_'+str(args.n_obs)+'_g'+str(args.gamma)+'_d'+str(args.delta)+'/'
os.makedirs(output_directory,exist_ok = True)
save_states_dir = output_directory+'save_states/'

# Instantiate confusion linear observable
mesh = dl.UnitSquareMesh(mesh_constructor_comm, args.nx, args.ny)

# Observable should take as argument a mesh and some kwargs

observable_kwargs = {'n_obs':args.n_obs,'output_folder':save_states_dir}
observable = confusion_linear_observable(mesh,**observable_kwargs)

# Matern Covariance Prior is instantiated here so that each process can sample m.
Vh = observable.problem.Vh

prior = matern_prior_2d(Vh[PARAMETER],gamma = args.gamma, delta = args.delta)

# Active Subspace
if args.save_as:
	print('AS object being created')
	AS_parameters = ActiveSubspaceParameterList()
	AS = ActiveSubspaceProjector(observable,prior,observable_kwargs = observable_kwargs\
		mesh_constructor_comm = mesh_constructor_comm,collective = my_collective,parameters = AS_parameters)

	exit()
	AS.parameters['rank'] = args.as_rank
	print(80*'#')
	print('Building AS input space')
	AS.construct_input_subspace()
	print(80*'#')
	print('Building AS output space')
	AS.construct_output_subspace()
	if int(my_rank) == 0:
		np.save(output_directory+'AS_input_projector',mv_to_dense(AS.V_GN))
		np.save(output_directory+'AS_d_GN',AS.d_GN)

		print('d_GN = ',AS.d_GN)
		out_name = output_directory+'AS_input_eigenvalues_'+str(args.as_rank)+'.pdf'
		_ = spectrum_plot(AS.d_GN,\
			axis_label = ['i',r'$\lambda_i$',r'Eigenvalues of $\mathbb{E}_{\nu}[{\nabla} q^T {\nabla} q]$'], out_name = out_name)
		with open(output_directory+'d_GN.txt','w') as my_file:
			my_file.write(str(AS.d_GN))

		np.save(output_directory+'AS_output_projector',mv_to_dense(AS.U_NG))
		np.save(output_directory+'AS_d_NG',AS.d_NG)

		print('d_NG = ',AS.d_NG)
		out_name = output_directory+'AS_output_eigenvalues_'+str(args.as_rank)+'.pdf'
		_ = spectrum_plot(AS.d_NG,\
			axis_label = ['i',r'$\lambda_i$',r'Eigenvalues of $\mathbb{E}_{\nu}[{\nabla} q {\nabla} q^T]$'], out_name = out_name)

		with open(output_directory+'d_NG.txt','w') as my_file:
			my_file.write(str(AS.d_NG))


if args.save_data or args.save_pod:
	print('POD object being created')
	POD_parameters = PODParameterList()
	POD_parameters['data_per_process']	 = args.data_per_process
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
	if int(my_rank) == 0:
		np.save(output_directory+'POD_projector',mv_to_dense(POD.U_MV))
		np.save(output_directory+'POD_d',POD.d)

		out_name = output_directory+'POD_eigenvalues_'+str(args.pod_rank)+'.pdf'
		_ = spectrum_plot(POD.d,\
			axis_label = ['i',r'$\lambda_i$',r'Eigenvalues of $\mathbb{E}_{\nu}[qq^T]$'], out_name = out_name)
		print('POD.d = ',POD.d)
		with open(output_directory+'pod_d.txt','w') as my_file:
			my_file.write(str(POD.d))

			






# # Solve the problem at the mean and save the mean field and velocity to file
# if args.save_two_solutions:
# 	# Solve for u at the mean
# 	m_mean = prior.mean
# 	print('||m_mean|| = ',m_mean.norm('l2'))
# 	m_mean_pvd = dl.File(save_states_dir+'m_mean.pvd')
# 	m_mean_pvd << vector2Function(m_mean,Vh[PARAMETER])

# 	u_at_mean = observable.problem.generate_state()
# 	observable.problem.solveFwd(u_at_mean,[u_at_mean,m_mean,None])

# 	print('||v_at_mean|| = ',u_at_mean.norm('l2'))
# 	v_at_mean_pvd = dl.File(save_states_dir+'v_at_mean.pvd')
# 	v_at_mean_pvd << vector2Function(u_at_mean,Vh[STATE])

# 	# Sample from prior:
# 	noise = dl.Vector()
# 	prior.init_vector(noise,"noise")
# 	parRandom.normal(1,noise)
# 	m_sample = observable.generate_vector(PARAMETER)
# 	prior.sample(noise,m_sample)

# 	print('||m_sample|| = ',m_sample.norm('l2'))
# 	m_sample_pvd = dl.File(save_states_dir+'m_sample.pvd')
# 	m_sample_pvd << vector2Function(m_sample,Vh[PARAMETER])

# 	u_at_sample = observable.problem.generate_state()
# 	observable.problem.solveFwd(u_at_sample,[u_at_sample,m_sample,None])

# 	print('||v_at_sample|| = ',u_at_sample.norm('l2'))
# 	v_at_sample_pvd = dl.File(save_states_dir+'v_at_sample.pvd')
# 	v_at_sample_pvd << vector2Function(u_at_sample,Vh[STATE])







