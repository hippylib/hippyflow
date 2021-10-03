# Copyright (c) 2020-2021, The University of Texas at Austin 
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
	def __init__(self,observable, prior, mesh_constructor_comm = None ,collective = None, parameters = PODParameterList()):
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


		self.d = None
		self.U_MV = None

		self.u_at_mean = None

	def solve_at_mean(self):
		"""
		Solve the PDE at the mean
		"""
		m_mean = self.prior.mean
		self.u_at_mean = self.observable.problem.generate_state()
		self.observable.problem.solveFwd(self.u_at_mean,[self.u_at_mean,m_mean,None])


	def generate_training_data(self,output_directory = 'data/',check_for_data = True,sequential = True):
		"""
		This method generates training data
			- :code:`output_directory` - a string specifying the path to the directory where data
			will be saved
			- :code:`check_for_data` - a boolean to decide whether to check to see if the training
			data already exists in directory specified by :code:`output_directory`.
		"""
		self.solve_at_mean()
		my_rank = int(self.collective.rank())
		try:
			os.makedirs(output_directory)
		except:
			pass
		observable_vector = dl.Vector(self.mesh_constructor_comm)
		self.observable.init_vector(observable_vector,dim = 0)
		last_datum_generated = -1
		m_shape = self.m.get_local().shape[0]
		q_shape = observable_vector.get_local().shape[0]
		print('m_shape = ',m_shape)
		print('q_shape = ',q_shape)
		if sequential:
			rank_specific_directory = output_directory+'ms_on_rank_'+str(my_rank)+'/'
			os.makedirs(rank_specific_directory)
			if check_for_data:
				ms_generated = os.listdir(rank_specific_directory)
				qs_generated = os.listdir(rank_specific_directory)
				m_indices = [int(m_.split('m_sample_')[-1].split('.npy')[0]) for m_ in ms_generated]
				last_m = max(m_indices)
				q_indices = [int(m_.split('q_sample_')[-1].split('.npy')[0]) for q_ in qs_generated]
				last_q = max(q_indices)
				last_datum_generated = min(last_m,last_q)

			t0 = time.time()
			for i in range(last_datum_generated,self.parameters['data_per_process']):
				print('Generating data number '+str(i))
				parRandom.normal(1,self.noise)
				self.prior.sample(self.noise,self.m)
				
				self.u.zero()
				self.u.axpy(1.,self.u_at_mean)
				x = [self.u,self.m,None]
				self.observable.setLinearizationPoint(x)
				solution = self.observable.eval(self.m).get_local()
				# If there is an issue with the solve move on
				# local_qs = np.concatenate((local_qs,np.expand_dims(solution,0)))
				# local_ms = np.concatenate((local_ms,np.expand_dims(self.m.get_local(),0)))
				# np.save(output_directory+'ms_on_rank_'+str(my_rank)+'.npy',np.array(local_ms))
				# np.save(output_directory+'qs_on_rank_'+str(my_rank)+'.npy',np.array(local_qs))
				try:
					solution = self.observable.eval(self.m).get_local()
					this_m = self.m.get_local()
					this_q = solution
					np.save(rank_specific_directory+'m_sample_'+str(i)+'.npy',this_m)
					np.save(rank_specific_directory+'q_sample_'+str(i)+'.npy',this_q)
					# If there is an issue with the solve move on
				except:
					print('Issue with the nonlinear solve, moving on')
					pass
				
				if self.parameters['verbose']:
					print('On datum generated every ',(time.time() -t0)/(i - last_datum_generated+1),' s, on average.')
			self._data_generation_time = time.time() - t0

		else:

			local_ms = np.zeros((0,m_shape))
			local_qs = np.zeros((0,q_shape))
			if check_for_data:
				if os.path.isfile(output_directory+'ms_on_rank_'+str(my_rank)+'.npy') and \
					os.path.isfile(output_directory+'qs_on_rank_'+str(my_rank)+'.npy'):
					local_ms = np.load(output_directory+'ms_on_rank_'+str(my_rank)+'.npy')
					local_qs = np.load(output_directory+'qs_on_rank_'+str(my_rank)+'.npy')
					last_datum_generated = min(local_ms.shape[0],local_qs.shape[0])

			t0 = time.time()
			# I think this is all hard coded for a single serial mesh, check if 
			# the arrays need to be communicated to mesh rank 0 before being saved
			for i in range(last_datum_generated,self.parameters['data_per_process']):
				print('Generating data number '+str(i))
				parRandom.normal(1,self.noise)
				self.prior.sample(self.noise,self.m)
				
				self.u.zero()
				self.u.axpy(1.,self.u_at_mean)
				x = [self.u,self.m,None]
				self.observable.setLinearizationPoint(x)
				solution = self.observable.eval(self.m).get_local()
				# If there is an issue with the solve move on
				# local_qs = np.concatenate((local_qs,np.expand_dims(solution,0)))
				# local_ms = np.concatenate((local_ms,np.expand_dims(self.m.get_local(),0)))
				# np.save(output_directory+'ms_on_rank_'+str(my_rank)+'.npy',np.array(local_ms))
				# np.save(output_directory+'qs_on_rank_'+str(my_rank)+'.npy',np.array(local_qs))
				try:
					solution = self.observable.eval(self.m).get_local()
					# If there is an issue with the solve move on
					local_qs = np.concatenate((local_qs,np.expand_dims(solution,0)))
					local_ms = np.concatenate((local_ms,np.expand_dims(self.m.get_local(),0)))
					np.save(output_directory+'ms_on_rank_'+str(my_rank)+'.npy',np.array(local_ms))
					np.save(output_directory+'qs_on_rank_'+str(my_rank)+'.npy',np.array(local_qs))
				except:
					print('Issue with the nonlinear solve, moving on')
					pass
				
				if self.parameters['verbose']:
					print('On datum generated every ',(time.time() -t0)/(i - last_datum_generated+1),' s, on average.')
			self._data_generation_time = time.time() - t0


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
			x = [self.u,self.m,None]
			self.observable.setLinearizationPoint(x)
			observable_vector.axpy(1.,self.observable.eval(self.m))
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
			x = [self.u,self.m,None]
			self.observable.setLinearizationPoint(x)
			LocalObservables[i].axpy(1.,self.observable.eval(self.m))
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
		self.observable.problem.solveFwd(u_at_mean,[u_at_mean,m_mean,None])

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
		self.observable.problem.solveFwd(u_at_sample,[u_at_sample,m_sample,None])

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
			x = [self.u,LocalParameters[i],None]
			self.observable.setLinearizationPoint(x)
			LocalObservables[i].axpy(1.,self.observable.eval(LocalParameters[i]))
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
				x = [self.u,input_projection_vector,None]
				self.observable.setLinearizationPoint(x)
				reduced_q_vector.axpy(1.,self.observable.eval(input_projection_vector))

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











