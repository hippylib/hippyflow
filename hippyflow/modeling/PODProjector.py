# Copyright (c) 2020-2022, The University of Texas at Austin 
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
import numpy as np
import time
from hippylib import *
from mpi4py import MPI 
import os

from ..collectives.collectiveOperator import CollectiveOperator
from ..collectives.comm_utils import checkMeshConsistentPartitioning
from ..utilities.mv_utilities import mv_to_dense
from ..utilities.plotting import *
from .priorPreconditionedProjector import PriorPreconditionedProjector

CONTROL = 3


def PODParameterList():
	"""
	This function returns a parameter list for the POD object
	"""
	parameters = {}
	parameters['sample_per_process'] = [100, 'Number of samples per process']
	parameters['rank'] 				 = [20, 'Rank of POD subspace']
	parameters['oversampling'] 		 = [10, 'Oversampling parameter for randomized algorithms']
	parameters['data_per_process']	 = [250,'Total number of testing and training data to be constructed']
	parameters['verbose']			 = [True,'Boolean for prints']

	parameters['output_directory']			= [None,'output directory for saving arrays and plots']
	parameters['plot_label_suffix']			= ['', 'suffix for plot label']

	return ParameterList(parameters)


class PODProjector:
	"""
	Projector class based on proper orthogonal decomposition
	"""
	def __init__(self,observable, prior, control_distribution = None, mesh_constructor_comm = None ,collective = None, parameters = PODParameterList()):
		"""
		Constructor
			- :code:`observable` - object that implements the observable mapping :math:`m -> q(m)`
			- :code:`prior` - object that implements the prior
			- :code:`mesh_constructor_comm` - MPI communicator that is used in mesh construction
			- :code:`collective` - MPI collective used in parallel collective operations
			- :code:`parameters` - parameter dictionary
		"""
		self.observable = observable
		self.prior = prior
		self.control_distribution = control_distribution

		if mesh_constructor_comm is not None:
			self.mesh_constructor_comm = mesh_constructor_comm
		else:
			self.mesh_constructor_comm = self.observable.mpi_comm()

		if collective is not None:
			self.collective = collective
		else:
			self.collective = NullCollective()


		self.parameters = parameters

		consistent_partitioning = checkMeshConsistentPartitioning(\
							self.observable.problem.Vh[0].mesh(), self.collective)
		print('Consistent partitioning:', consistent_partitioning)

		self.noise = dl.Vector(self.mesh_constructor_comm)
		self.prior.init_vector(self.noise,"noise")

		self.u = self.observable.generate_vector(STATE)
		self.m = self.observable.generate_vector(PARAMETER)
		if self.control_distribution is None:
			self.z = None
		else:
			self.z = self.observable.generate_vector(CONTROL)


		self.d = None
		self.U_MV = None

		self.u_at_mean = None

	def solve_at_mean(self):
		"""
		Solve the PDE at the mean
		"""
		m_mean = self.prior.mean
		self.u_at_mean = self.observable.problem.generate_state()
		if self.control_distribution is not None:
			if hasattr(self.control_distribution,'mean'):
				z_mean = self.control_distribution.mean
			else:
				z_mean = self.observable.generate_vector(CONTROL)
				self.control_distribution.sample(z_mean)
			self.observable.problem.solveFwd(self.u_at_mean,[self.u_at_mean,m_mean,None,z_mean])
		else:
			self.observable.problem.solveFwd(self.u_at_mean,[self.u_at_mean,m_mean,None])

	def generate_training_data(self,check_for_data = True,sequential = True,\
										compress_files = True):
		"""
		This method generates training data
			- :code:`output_directory` - a string specifying the path to the directory where data
			will be saved
			- :code:`check_for_data` - a boolean to decide whether to check to see if the training
			data already exists in directory specified by :code:`output_directory`.
		"""
		self.solve_at_mean()
		my_rank = int(self.collective.rank())
		output_directory = self.parameters['output_directory']

		os.makedirs(output_directory,exist_ok = True)

		observable_vector = dl.Vector(self.mesh_constructor_comm)
		self.observable.init_vector(observable_vector,dim = 0)
		last_datum_generated = 0
		parameter_dimension = self.m.get_local().shape[0]
		output_dimension = observable_vector.get_local().shape[0]
		print('dM = ',parameter_dimension)
		print('dQ = ',output_dimension)
		if self.control_distribution is not None:
			control_dimension = self.z.get_local().shape[0]
			print('dZ = ',control_dimension)
		
		if sequential:
			rank_specific_directory = output_directory+'data_on_rank_'+str(my_rank)+'/'
			os.makedirs(rank_specific_directory,exist_ok = True)
			if check_for_data:
				if self.control_distribution is not None:
					if os.path.isfile(rank_specific_directory+'m_sample_0.npy') and \
						os.path.isfile(rank_specific_directory+'q_sample_0.npy') and \
						os.path.isfile(rank_specific_directory+'z_sample_0.npy'):
						all_files = os.listdir(rank_specific_directory)
						ms_generated = []
						qs_generated = []
						zs_generated = []
						for file in all_files:
							if 'm_' in file:
								ms_generated.append(file)
							elif 'q_' in file:
								qs_generated.append(file)
							elif 'z_' in file:
								zs_generated.append(file)
						m_indices = [int(m_.split('m_sample_')[-1].split('.npy')[0]) for m_ in ms_generated]
						last_m = max(m_indices)
						q_indices = [int(q_.split('q_sample_')[-1].split('.npy')[0]) for q_ in qs_generated]
						last_q = max(q_indices)
						z_indices = [int(z_.split('z_sample_')[-1].split('.npy')[0]) for z_ in zs_generated]
						last_z = max(z_indices)
						last_datum_generated = min(last_m,last_q,last_z)
				else:
					if os.path.isfile(rank_specific_directory+'m_sample_0.npy') and \
						os.path.isfile(rank_specific_directory+'q_sample_0.npy'):
						all_files = os.listdir(rank_specific_directory)
						ms_generated = []
						qs_generated = []
						for file in all_files:
							if 'm_' in file:
								ms_generated.append(file)
							elif 'q_' in file:
								qs_generated.append(file)
						m_indices = [int(m_.split('m_sample_')[-1].split('.npy')[0]) for m_ in ms_generated]
						last_m = max(m_indices)
						q_indices = [int(q_.split('q_sample_')[-1].split('.npy')[0]) for q_ in qs_generated]
						last_q = max(q_indices)
						last_datum_generated = min(last_m,last_q)

			t0 = time.time()
			for i in range(last_datum_generated,self.parameters['data_per_process']):
				print('Generating data number '+str(i))
				solved = False
				while not solved:
					try:
						self.m.zero()
						self.noise.zero()
						parRandom.normal(1,self.noise)
						self.prior.sample(self.noise,self.m)
						if self.control_distribution is None:
							linearization_x = [self.u,self.m,None]
						else:
							self.z.zero()
							self.control_distribution.sample(self.z)
							linearization_x = [self.u,self.m,None,self.z]

						self.observable.solveFwd(self.u,linearization_x)
						this_m = self.m.get_local()
						this_q = self.observable.evalu(self.u).get_local()
						
						np.save(rank_specific_directory+'m_sample_'+str(i)+'.npy',this_m)
						np.save(rank_specific_directory+'q_sample_'+str(i)+'.npy',this_q)

						if self.control_distribution is not None:
							this_z = self.z.get_local()
							np.save(rank_specific_directory+'z_sample_'+str(i)+'.npy',this_z)
					except:
						print('Issue with the forward solution, moving on.')

					if self.parameters['verbose']:
						print('On datum generated every ',(time.time() -t0)/(i - last_datum_generated+1),' s, on average.')

			self._data_generation_time = time.time() - t0
			if compress_files:
				local_ms = np.zeros((self.parameters['data_per_process'],parameter_dimension))
				local_qs = np.zeros((self.parameters['data_per_process'],output_dimension))
				if self.control_distribution is not None:
					local_zs = np.zeros((self.parameters['data_per_process'],control_dimension))
				for i in range(0,self.parameters['data_per_process']):
					local_ms[i] = np.load(rank_specific_directory+'m_sample_'+str(i)+'.npy')
					local_qs[i] = np.load(rank_specific_directory+'q_sample_'+str(i)+'.npy')
					if self.control_distribution is not None:
						local_zs[i] = np.load(rank_specific_directory+'z_sample_'+str(i)+'.npy')
				if self.control_distribution is not None:
					np.savez_compressed(output_directory+'mqz_on_rank'+str(my_rank)+'.npz',m_data = local_ms,\
										q_data = local_qs, z_data = local_zs)
				else:
					np.savez_compressed(output_directory+'mq_on_rank'+str(my_rank)+'.npz',m_data = local_ms,q_data = local_qs)

		else:

			local_ms = np.zeros((0,parameter_dimension))
			local_qs = np.zeros((0,output_dimension))
			if self.control_distribution is not None:
				local_zs = np.zeros((0,control_distribution))

			if check_for_data:
				if os.path.isfile(output_directory+'ms_on_rank_'+str(my_rank)+'.npy') and \
					os.path.isfile(output_directory+'qs_on_rank_'+str(my_rank)+'.npy'):
					local_ms = np.load(output_directory+'ms_on_rank_'+str(my_rank)+'.npy')
					local_qs = np.load(output_directory+'qs_on_rank_'+str(my_rank)+'.npy')
					if control_distribution is not None:
						local_zs = np.load(output_directory+'zs_on_rank_'+str(my_rank)+'.npy')
						last_datum_generated = min(local_ms.shape[0],local_qs.shape[0],local_zs.shape[0])
					else:
						last_datum_generated = min(local_ms.shape[0],local_qs.shape[0])

			t0 = time.time()
			# I think this is all hard coded for a single serial mesh, check if 
			# the arrays need to be communicated to mesh rank 0 before being saved
			for i in range(last_datum_generated,self.parameters['data_per_process']):
				solved = False
				while not solved:
					try:
						print('Generating data number '+str(i))
						parRandom.normal(1,self.noise)
						self.prior.sample(self.noise,self.m)
						self.u.zero()
						self.u.axpy(1.,self.u_at_mean)


						if self.control_distribution is not None:
							self.z.zero()
							self.control_distribution.sample(self.z)
							linearization_x = [self.u,self.m,None,self.z]
						else:
							linearization_x = [self.u,self.m,None]


						self.observable.solveFwd(self.u,linearization_x)
						this_m = self.m.get_local()
						this_q = self.observable.evalu(self.u).get_local()
						local_qs = np.concatenate((local_qs,np.expand_dims(this_q,0)))
						local_ms = np.concatenate((local_ms,np.expand_dims(this_m,0)))
						np.save(output_directory+'ms_on_rank_'+str(my_rank)+'.npy',np.array(local_ms))
						np.save(output_directory+'qs_on_rank_'+str(my_rank)+'.npy',np.array(local_qs))
						if self.control_distribution is not None:
							this_z = self.z.get_local()
							local_zs = np.concatenate((local_zs,np.expand_dims(this_z,0)))
							np.save(output_directory+'zs_on_rank_'+str(my_rank)+'.npy',np.array(local_zs))

					except:
						print('Issue with nonlinear solve, moving on')
				if self.parameters['verbose']:
					print('On datum generated every ',(time.time() -t0)/(i - last_datum_generated+1),' s, on average.')
			self._data_generation_time = time.time() - t0

	def save_mass_and_stiffness_matrices(self):
		"""
		This method saves mass and stiffness matrices 
		"""
		# Save mass matrix
		output_directory = self.parameters['output_directory']
		os.makedirs(output_directory,exist_ok = True)
		import scipy.sparse as sp

		u_trial = dl.TrialFunction(self.observable.problem.Vh[STATE])
		u_test = dl.TestFunction(self.observable.problem.Vh[STATE])
		M = dl.PETScMatrix()
		dl.assemble(dl.inner(u_trial,u_test)*dl.dx, tensor=M)
		
		# from scipy.sparse import csc_matrix, csr_matrix, save_npz
		# from scipy.sparse import linalg as spla

		M_mat = dl.as_backend_type(M).mat()
		row,col,val = M_mat.getValuesCSR()
		M_csr = sp.csr_matrix((val,col,row)) 
		sp.save_npz(output_directory+'mass_csr',M_csr)

		# Save stiffness matrix
		K = dl.PETScMatrix()
		dl.assemble(dl.inner(dl.grad(u_trial),dl.grad(u_test))*dl.dx, tensor=K)
		K_mat = dl.as_backend_type(K).mat()
		row,col,val = K_mat.getValuesCSR()
		K_csr = sp.csr_matrix((val,col,row)) 
		sp.save_npz(output_directory+'stiffness_csr',K_csr)



	def construct_subspace(self):
		"""
		This method constructs the POD subspace
		"""
		t0 = time.time()
		self.solve_at_mean()
		observable_vector = dl.Vector(self.mesh_constructor_comm)
		self.observable.init_vector(observable_vector,dim = 0)
		LocalObservables = MultiVector(observable_vector,self.parameters['sample_per_process'])
		LocalObservables.zero()

		#Read data from file and build subspace option
		for i in range(LocalObservables.nvec()):
			# if self.parameters['verbose']:
				# print('Starting observable generation for draw ',i)
			observable_vector.zero()
			parRandom.normal(1,self.noise)
			self.prior.sample(self.noise,self.m)
			if self.control_distribution is not None:
				self.z.zero()
				self.control_distribution.sample(self.z)
				x = [self.u,self.m,None,self.z]
			else:
				x = [self.u,self.m,None]
			self.observable.solveFwd(self.u,x)
			observable_vector.axpy(1.,self.observable.evalu(self.u))
			LocalObservables[i].axpy(1.,observable_vector)

		init_vector_lambda = lambda x, dim: self.observable.init_vector(x,dim = 0)
		LocalPODOperator = LowRankOperator(np.ones(LocalObservables.nvec())/self.parameters['sample_per_process']\
																			,LocalObservables,init_vector_lambda)

		GlobalPODOperator = CollectiveOperator(LocalPODOperator, self.collective,mpi_op = 'avg')

		x_POD = dl.Vector(self.mesh_constructor_comm)
		LocalPODOperator.init_vector(x_POD,dim = 0)
		Omega_POD = MultiVector(x_POD,self.parameters['rank'] + self.parameters['oversampling'])

		if self.collective.rank() == 0:
			parRandom.normal(1.,Omega_POD)
		else:
			Omega_POD.zero()

		self.collective.bcast(Omega_POD,root = 0)

		self.d, self.U_MV = doublePass(GlobalPODOperator,Omega_POD,self.parameters['rank'],s = 1)

		self._subspace_construction_time = time.time() - t0
		if self.parameters['verbose'] and (self.mesh_constructor_comm.rank ==0):
			print('Construction of POD subspace took ', self._subspace_construction_time,'s')

		if True and MPI.COMM_WORLD.rank == 0:
			np.save(self.parameters['output_directory']+'POD_projector',mv_to_dense(self.U_MV))
			np.save(self.parameters['output_directory']+'POD_d',self.d)

			out_name = self.parameters['output_directory']+'POD_eigenvalues_'+str(self.parameters['rank'])+'.pdf'
			_ = spectrum_plot(self.d,\
				axis_label = ['i',r'$\lambda_i$',\
				r'Eigenvalues of $\mathbb{E}_{\nu}[qq^T]$'+self.parameters['plot_label_suffix']], out_name = out_name)


	def test_output_errors(self,ranks = [None],cut_off = 1e-10):
		"""
		This method performs a simple projection error test on the output
			- :code:`ranks` - a python list of integers specifying ranks for projection tests
			- :code:`cut_off` - where to truncate the ranks based on the spectral decay of the POD operator
		"""
		assert self.control_distribution is None, 'Not worked out yet for control problems'
		ranks.sort()
		if (self.d is None) or (self.U_MV is None):
			if self.mesh_constructor_comm.rank == 0:
				self.parameters['rank'] = ranks[-1]
				print('Subspace not constructed, constructing now')
			self.construct_subspace()
		elif len(self.d) < ranks[-1]:
			if self.mesh_constructor_comm.rank == 0:
				print('Constructing subspace because larger rank needed.')
				self.parameters['rank'] = ranks[-1]
			self.construct_subspace()
		# truncate eigenvalues for numerical stability
		numericalrank = np.where(self.d > cut_off)[-1][-1]
		ranks = ranks[:np.where(ranks < numericalrank)[0][-1]]

		observable_vector = dl.Vector(self.mesh_constructor_comm)
		self.observable.init_vector(observable_vector,dim = 0)
		LocalObservables = MultiVector(observable_vector,self.parameters['sample_per_process'])
		LocalObservables.zero()
		for i in range(LocalObservables.nvec()):
			t0 = time.time()
			parRandom.normal(1,self.noise)
			self.prior.sample(self.noise,self.m)
			if self.control_distribution is not None:
				self.z.zero()
				self.control_distribution.sample(self.z)
				x = [self.u,self.m,None,self.z]
			else:
				x = [self.u,self.m,None]
			self.observable.solveFwd(self.u,x)
			LocalObservables[i].axpy(1.,self.observable.evalu(self.u))
			# if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):
			# 	print('Generating local observable ',i,' for POD error test took',time.time() -t0, 's')

		LocalErrors = MultiVector(observable_vector,self.parameters['sample_per_process'])

		projection_vector = dl.Vector(self.mesh_constructor_comm)
		self.observable.init_vector(projection_vector,dim = 0)
		global_avg_rel_errors = []
		global_std_rel_errors = []
		for rank in ranks:
			LocalErrors.zero()
			if rank is None:
				U_MV = self.U_MV
				d = self.d
			else:
				U_MV = MultiVector(self.U_MV[0],rank)
				d = self.d[0:rank]
				for i in range(rank):
					U_MV[i].axpy(1.,self.U_MV[i])
					# U_MV[i].set_local(self.U_MV[i].get_local())
					# U_MV[i].apply('')

			init_vector_lambda = lambda x, dim: self.observable.init_vector(x,dim = 0)
			PODOperator = LowRankOperator(np.ones_like(d),U_MV,init_vector_lambda)

			rel_errors = np.zeros(LocalErrors.nvec())
			for i in range(LocalErrors.nvec()):
				t0 = time.time()
				LocalErrors[i].axpy(1.,LocalObservables[i])
				denominator = LocalErrors[i].norm('l2')
				projection_vector.zero()
				PODOperator.mult(LocalErrors[i],projection_vector)
				LocalErrors[i].axpy(-1.,projection_vector)
				numerator = LocalErrors[i].norm('l2')
				rel_errors[i] = numerator/denominator
				if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):
					print('Local error calculation ',i,' for POD error test took',time.time() -t0, 's')
					print('numerator for ',i, ' is ', numerator)
					print('denominator for ', i , ' is ', denominator)
					print('rel_errors[i] for ',i, ' is ',rel_errors[i])

			avg_rel_error = np.mean(rel_errors)
			std_rel_error_squared = np.std(rel_errors)**2
			global_avg_rel_errors.append(self.collective.allReduce(avg_rel_error,'avg'))
			global_std_rel_errors.append(np.sqrt(self.collective.allReduce(std_rel_error_squared,'avg')))
			if self.mesh_constructor_comm.rank == 0:
				print('Global average relative error = ',global_avg_rel_errors[-1])

		return global_avg_rel_errors, global_std_rel_errors


	def two_state_solution(self):
		"""
		Solve the problem at the mean and save the mean field and velocity to file
		"""
		save_states_dir = self.parameters['output_directory']+'two_states/'
		os.makedirs(save_states_dir,exist_ok = True)
		m_mean = self.prior.mean
		if self.parameters['verbose']:
			print('||m_mean|| = ',m_mean.norm('l2'))
		m_mean_pvd = dl.File(save_states_dir+'m_mean.pvd')
		m_mean_pvd << vector2Function(m_mean,self.observable.problem.Vh[PARAMETER])
		u_at_mean = self.observable.problem.generate_state()

		if self.control_distribution is not None:
			if hasattr(self.control_distribution,'mean'):
				z_mean = self.control_distribution.mean
			else:
				print('Substituting a sample of z for z_mean, since ')
				print('control_distribution did not have a mean attribute.')
				z_mean = self.observable.generate_vector(CONTROL)
				self.control_distribution.sample(z_mean)

			linearization_x = [u_at_mean,m_mean,None,z_mean]
		else:
			linearization_x = [u_at_mean,m_mean,None]

		self.observable.problem.solveFwd(u_at_mean,linearization_x)

		if self.parameters['verbose']:
			print('||v_at_mean|| = ',u_at_mean.norm('l2'))
		v_at_mean_pvd = dl.File(save_states_dir+'v_at_mean.pvd')
		v_at_mean_pvd << vector2Function(u_at_mean,self.observable.problem.Vh[STATE])

		# Sample from prior:
		parRandom.normal(1,self.noise)
		m_sample = self.observable.generate_vector(PARAMETER)
		self.prior.sample(self.noise,m_sample)

		if self.parameters['verbose']:
			print('||m_sample|| = ',m_sample.norm('l2'))
		m_sample_pvd = dl.File(save_states_dir+'m_sample.pvd')
		m_sample_pvd << vector2Function(m_sample,self.observable.problem.Vh[PARAMETER])

		u_at_sample = self.observable.problem.generate_state()
		if self.control_distribution is not None:
			z_sample = self.observable.generate_vector(CONTROL)
			self.control_distribution.sample(z_sample)
			linearization_x = [u_at_mean,m_mean,None,z_sample]
		else:
			linearization_x = [u_at_mean,m_mean,None]

		self.observable.problem.solveFwd(u_at_sample,linearization_x)

		if self.parameters['verbose']:
			print('||v_at_sample|| = ',u_at_sample.norm('l2'))
		v_at_sample_pvd = dl.File(save_states_dir+'v_at_sample.pvd')
		v_at_sample_pvd << vector2Function(u_at_sample,self.observable.problem.Vh[STATE])



	def input_output_error_test(self,V_MV,Cinv = None,rank_pairs = [None]):
		"""
		This method implements an input output projection error test
		The output projection basis is given by the POD eigenvectors
			- :code:`V_MV` - a multi vector object of the low rank basis used in the projector
			- :code:`Cinv` - the covariance inverse which may be used in a prior preconditioned projector
			- :code:`rank_pairs` - a python list of 2-tuples of ints specifying the input and output ranks 
				to be used in the projection error test.
		"""
		assert self.control_distribution is None, 'Not worked out yet for control problems'
		for (rank_in,rank_out) in rank_pairs:
			assert rank_in <=V_MV.nvec()
			assert rank_out <= self.U_MV.nvec()
		assert self.d is not None
		assert self.U_MV is not None
		# Check to see if the multivectors are large enough
		# truncate eigenvalues for numerical stability

		# Instantiate parameter vector for requisite data structures
		LocalParameters = MultiVector(self.m,self.parameters['sample_per_process'])
		LocalParameters.zero()

		input_projection_vector = self.observable.generate_vector(PARAMETER)

		# Instantiate an observable vector for requisite data structures
		observable_vector = dl.Vector(self.mesh_constructor_comm)
		self.observable.init_vector(observable_vector,dim = 0)

		LocalObservables = MultiVector(observable_vector,self.parameters['sample_per_process'])
		LocalObservables.zero()
		t0 = time.time()
		for i in range(LocalObservables.nvec()):
			parRandom.normal(1,self.noise)
			self.prior.sample(self.noise,LocalParameters[i])
			if self.control_distribution is not None:
				self.z.zero()
				self.control_distribution.sample(self.z)
				linearization_x = [self.u,LocalParameters[i],None,self.z]
			else:
				linearization_x = [self.u,LocalParameters[i],None]
			self.observable.solveFwd(self.u,linearization_x)
			LocalObservables[i].axpy(1.,self.observable.evalu(self.u))
		if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):
			print('Generating ',LocalObservables.nvec(),' local parameters and observables for POD error test took',time.time() -t0, 's')

		# LocalProjectedObservables = MultiVector(observable_vector,self.parameters['sample_per_process'])
		# LocalProjectedObservables.zero()

		LocalErrors = MultiVector(observable_vector,self.parameters['sample_per_process'])


		output_projection_vector = dl.Vector(self.mesh_constructor_comm)
		reduced_q_vector = dl.Vector(self.mesh_constructor_comm)
		self.observable.init_vector(output_projection_vector,dim = 0)
		self.observable.init_vector(reduced_q_vector,dim = 0)

		global_avg_rel_errors = []
		global_std_rel_errors = []
		for (rank_in,rank_out) in rank_pairs:
			# Define input projector operator for rank_in
			V_r = MultiVector(V_MV[0],rank_in)
			for i in range(rank_in):
				V_r[i].axpy(1.,V_MV[i])
			input_init_vector_lambda = lambda x, dim: self.observable.init_vector(x,dim = 1)
			if Cinv is not None:
				InputProjectorOperator = PriorPreconditionedProjector(V_r,Cinv, input_init_vector_lambda)
			else:
				InputProjectorOperator = LowRankOperator(np.ones(rank_in),V_r, input_init_vector_lambda)

			# Define output projector operator for rank_out
			U_MV = MultiVector(self.U_MV[0],rank_out)
			for i in range(rank_out):
				U_MV[i].axpy(1.,self.U_MV[i])

			init_vector_lambda = lambda x, dim: self.observable.init_vector(x,dim = 0)
			PODOperator = LowRankOperator(np.ones(rank_out),U_MV,init_vector_lambda)


			LocalErrors.zero()
			rel_errors = np.zeros(LocalErrors.nvec())
			t0 = time.time()
			for i in range(LocalErrors.nvec()):
				input_projection_vector.zero()
				output_projection_vector.zero()
				reduced_q_vector.zero()
				LocalErrors[i].axpy(1.,LocalObservables[i])
				denominator = LocalErrors[i].norm('l2')
				InputProjectorOperator.mult(LocalParameters[i],input_projection_vector)
				if self.control_distribution is not None:
					self.z.zero()
					self.control_distribution.sample(self.z)
					linearization_x = [self.u,input_projection_vector,None, self.z]
				else:
					linearization_x = [self.u,input_projection_vector,None]

				self.observable.solveFwd(self.u,linearization_x)
				reduced_q_vector.axpy(1.,self.observable.evalu(self.u))

				PODOperator.mult(reduced_q_vector,output_projection_vector)
				LocalErrors[i].axpy(-1.,output_projection_vector)
				numerator = LocalErrors[i].norm('l2')
				rel_errors[i] = numerator/denominator
			if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):
				print('The ',i,' local error calculations for POD error test took',time.time() -t0, 's')


			avg_rel_error = np.mean(rel_errors)
			std_rel_error_squared = np.std(rel_errors)**2
			global_avg_rel_errors.append(self.collective.allReduce(avg_rel_error,'avg'))
			global_std_rel_errors.append(np.sqrt(self.collective.allReduce(std_rel_error_squared,'avg')))
			if self.mesh_constructor_comm.rank == 0:
				print('Rank pair = ',(rank_in,rank_out),'Global average relative error = ',global_avg_rel_errors[-1])

		return global_avg_rel_errors, global_std_rel_errors











