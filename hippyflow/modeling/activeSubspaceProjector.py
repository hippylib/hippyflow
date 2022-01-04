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
import os
from hippylib import *
from mpi4py import MPI 
import time


from ..collectives.collectiveOperator import CollectiveOperator, MatrixMultCollectiveOperator
from ..collectives.collective import NullCollective
from ..collectives.comm_utils import checkMeshConsistentPartitioning
from .jacobian import *
from ..utilities.mv_utilities import mv_to_dense
from ..utilities.plotting import *

from .priorPreconditionedProjector import PriorPreconditionedProjector

def ActiveSubspaceParameterList():
	"""
	This function implements a parameter list for the ActiveSubspaceProjector
	"""
	parameters = {}
	parameters['samples_per_process'] 		= [64, 'Number of samples per process']
	parameters['error_test_samples'] 		= [50, 'Number of samples for error test']
	parameters['rank'] 				 		= [128, 'Rank of subspace']
	parameters['oversampling'] 		 		= [10, 'Oversampling parameter for randomized algorithms']
	parameters['double_loop_samples']		= [20, 'Number of samples used in double loop MC approximation']
	parameters['verbose']					= [True, 'Boolean for printing']


	parameters['initialize_samples'] 		= [False,'Boolean for the initialization of samples when\
														many samples are allocated on one process ']
	parameters['serialized_sampling']		= [False, 'Boolean for the serialization of sampling on a process\
													 to reduce memory for large problems']

	parameters['observable_constructor'] 	= [None,'observable constructor function, assumed to take a mesh, and kwargs']
	parameters['observable_kwargs'] 		= [{},'kwargs used when instantiating multiple local instances of observables']

	parameters['output_directory']			= [None,'output directory for saving arrays and plots']
	parameters['plot_label_suffix']			= ['', 'suffix for plot label']
	parameters['save_and_plot']				= [True,'Boolean for saving data and plots (only False for unit testing)']
	parameters['store_Omega']				= [False,'Boolean for storing Gaussian random matrix (only True for unit testing)']
	parameters['ms_given']					= [False,'Boolean for passing ms into serialized AS construction (only True for unit testing)']
	return ParameterList(parameters)


class SummedListOperator:
	'''
	Assumes that list of operators all have the same dimensionality
	'''
	def __init__(self,operators,communicator = None,average = True):
		assert type(operators) is list
		self.operators = operators
		self.average = average
		if communicator is None:
			self.temp = None
		else:
			self.temp = dl.Vector(communicator)

	def mult(self,x,y):
		if self.temp is None:
			temp = dl.Vector(y)
		else:
			temp = self.temp
		for op in self.operators:
			t0 = time.time()
			op.mult(x,y)
			temp.axpy(1.,y)
		y.zero()
		if self.average:
			y.axpy(1./float(len(self.operators)),temp)
		else:
			y.axpy(1.,temp)

class SeriallySampledJacobianOperator:
	'''
	Alterantive to SummedListOperator when memory is an issue for active subspace
	'''
	def __init__(self,observable,noise,prior,operation = 'JTJ',nsamples = None,ms = None,communicator=None,average=True):
		'''
		'''
		assert operation in ['JTJ','JJT']
		assert (nsamples is not None) or (ms is not None)
		self.observable = observable
		self.noise = noise
		self.prior = prior
		self.operation = operation
		self.nsamples = nsamples
		self.average = average
		self.ms = ms
		
		if communicator is None:
			self.temp = None
		else:
			self.temp = dl.Vector(communicator)

		self.u = self.observable.generate_vector(STATE)
		self.m = self.observable.generate_vector(PARAMETER)

	def init_vector(self,x):
		"""
		Reshape the Vector :code:`x` so that it is compatible with the Jacobian
		operator.

		Parameters:

		- :code:`x`: the vector to reshape.
		 """
		if self.operation == 'JJT':
			self.observable.init_vector(x,0)
		elif self.operation == 'JTJ':
			self.observable.init_vector(x,1)
		else: 
			raise

	def matMvMult(self,x,y):
		'''
		'''
		if self.temp is None:
			temp = dl.Vector(y[0])
		else:
			temp = self.temp
		assert x.nvec() == y.nvec(), "x and y have non-matching number of vectors"
		# Instance Jacobian operator for this input-output pair
		if self.operation == 'JTJ':
			operator_i = JTJ(ObservableJacobian(self.observable))
		elif self.operation == 'JJT':
			operator_i = JJT(ObservableJacobian(self.observable))

		if self.ms is None:
			for i in range(self.nsamples):
				print('Updating mat vec for sample = ',i)
				# Iterate if there are solver issues
				solved = False
				while not solved:
					try:
						# Sample from the prior
						self.m.zero()
						self.noise.zero()
						parRandom.normal(1,self.noise)
						self.prior.sample(self.noise,self.m)
						# Solve the PDE
						linearization_x = [self.u,self.m,None]
						print('Attempting to solve')
						t0 = time.time()
						self.observable.solveFwd(self.u,linearization_x)
						print('Solution succesful, and took ',time.time() - t0,'s') 
						# set linearization point
						self.observable.setLinearizationPoint(linearization_x)
						solved = True
					except:
						print('Issue with the solution, moving on')
						pass
				# Define action on matrix (as represented by MultiVector)
				for j in range(x.nvec()):
					temp.zero()
					t0 = time.time()
					operator_i.mult(x[j],temp)
					if self.average:
						y[j].axpy(1./self.nsamples, temp)
					else:
						y[j].axpy(1., temp)
		################################################################################
		# Otherwise m represents points already chosen,
		# where we assume the solution will hold, and thus we 
		# do not handle the sampling of m here
		# This section is mostly for the sake of unit testing
		else:
			# Each m should already not run into solver issues
			nsamples = len(self.ms)
			for m in self.ms:
				# Solve the PDE
				print('Attempting to solve')
				linearization_x = [self.u,m,None]
				self.observable.solveFwd(self.u,linearization_x)
				print('Solution succesful')
				# set linearization point
				self.observable.setLinearizationPoint(linearization_x)
				solved = True
				# Define action on matrix (as represented by MultiVector)
				for j in range(x.nvec()):
					temp.zero()
					operator_i.mult(x[j],temp)
					if self.average:
						y[j].axpy(1./nsamples, temp)
					else:
						y[j].axpy(1., temp)



class ActiveSubspaceProjector:
	"""
	This class implements projectors based on globally averages GN Hessian and inside out GN Hessian
	We have a forward mapping: :math:`m -> q(m)`
	And a forward map Jacobian:  :math:`\nabla q(m)`
	Jacobian SVD: :math:`J = U S V^*`
	Output active subspace: :math:`J^*J = US^2U^*`
	Input active subspace: :math:`JJ^* = VS^2V^*`
	"""
	def __init__(self,observable, prior, mesh_constructor_comm = None ,collective = NullCollective(),\
								  parameters = ActiveSubspaceParameterList()):
		"""
		Constructor
			- :code:`observable` - object that implements the observable mapping :math:`m -> q(m)`
			- :code:`prior` - object that implements the prior
			- :code:`mesh_constructor_comm` - MPI communicator that is used in mesh construction
			- :code:`collective` - MPI collective used in parallel collective operations
			- :code:`parameters` - parameter dictionary
		"""
		self.parameters = parameters
		if self.parameters['verbose']:
			print(80*'#')
			print('Active Subspace object being created'.center(80))
		self.observable = observable
		self.prior = prior


		if mesh_constructor_comm is not None:
			self.mesh_constructor_comm = mesh_constructor_comm
		else:
			self.mesh_constructor_comm = self.observable.mpi_comm()

		self.collective = collective

		consistent_partitioning = checkMeshConsistentPartitioning(\
							self.observable.problem.Vh[0].mesh(), self.collective)
		assert consistent_partitioning
		if self.parameters['verbose']:
			print(('Consistent partitioning: '+str(consistent_partitioning)).center(80))

		# Initialize many copies of observables here
		self.observables = [self.observable]

		# Can I infer input and output dimension here??
		# If so then I can check for data first in low rank Jacobian generation
		# and avoid the time consuming sample initialization.


		# Here we allocate many copies of the observable if serialized_sampling is not True in the
		# active subspace parameters
		if self.parameters['samples_per_process'] > 1 and not self.parameters['serialized_sampling']:
			assert self.parameters['observable_constructor'] is not None
			for i in range(self.parameters['samples_per_process']-1):
				new_observable = self.parameters['observable_constructor'](self.observable.problem.Vh[0].mesh(),**self.parameters['observable_kwargs'])
				self.observables.append(new_observable)

		self.noise = dl.Vector(self.mesh_constructor_comm)
		self.prior.init_vector(self.noise,"noise")

		if self.parameters['serialized_sampling']:
			self.u = None
			self.m = None
			self.J = None
		else:		
			self.us = None
			self.ms = None
			self.Js = None

		# Draw a new sample and set linearization point.
		if self.parameters['initialize_samples']:
			if not self.parameters['serialized_sampling']:
				self._initialize_batched_samples()

		
		self.d_GN = None
		self.V_GN = None
		self.d_GN_noprior = None
		self.V_GN_noprior = None
		self.prior_preconditioned = None

		self.d_NG = None
		self.U_NG = None

		# For unit testing different methods, want to save Omega
		self.Omega_GN = None
		self.Omega_NG = None

	def _initialize_batched_samples(self):
		"""
		This method initializes the samples from the prior used in sampling
		"""
		t0 = time.time()
		self.us = [self.observable.generate_vector(STATE) for i in range(self.parameters['samples_per_process'])]
		self.ms = [self.observable.generate_vector(PARAMETER) for i in range(self.parameters['samples_per_process'])]
		for u,m,observable in zip(self.us,self.ms,self.observables):
			solved = False
			while not solved:
				try:
					self.noise.zero()
					parRandom.normal(1,self.noise)
					# set linearization point
					self.prior.sample(self.noise,m)
					x = [u,m,None]
					observable.solveFwd(u,x)
					observable.setLinearizationPoint(x)
					solved = True
				except:
					m.zero()
					print('Issue with the solution, moving on')
					pass
		if self.parameters['verbose']:
			try:
				from pympler import asizeof
				print(('Size of one observable is '+str(asizeof.asizeof(self.observables[0])/1e6)+' MB').center(80))
				print(('Size of '+str(self.parameters['samples_per_process'])+' observable is '+str(asizeof.asizeof(self.observables)/1e6)+' MB').center(80))
			except:
				print('Install pympler and run again: pip install pympler'.center(80))
		self.Js = [ObservableJacobian(observable) for observable in self.observables]
		total_init_time = time.time() - t0
		for i in range(100):
			print(80*'#')
			print('Initializing all batched samples took ',total_init_time, 's ')


	def construct_input_subspace(self,prior_preconditioned = True):
		if self.parameters['serialized_sampling']:
			print('Construction via serialized AS construction'.center(80))
			self._construct_serialized_jacobian_subspace(prior_preconditioned = prior_preconditioned,operation = 'JTJ')
		else:
			print('Construction via batched AS construction'.center(80))
			self._construct_input_subspace_batched(prior_preconditioned = prior_preconditioned)




	def _construct_input_subspace_batched(self,prior_preconditioned = True):
		"""
		This method implements the input subspace constructor 
			-:code:`prior_preconditioned` - a Boolean to decide whether to include the prior covariance in the decomposition
				The default parameter is True which is customary in active subspace construction
		"""
		if self.Js is None:
			self._initialize_batched_samples()

		if self.parameters['verbose']:
			print(80*'#')
			print('Building derivative informed input subspace'.center(80))
		

		t0 = time.time()
		GN_Hessians = [JTJ(J) for J in self.Js]
		Local_Average_GN_Hessian = SummedListOperator(GN_Hessians,average = True)
		# This averaging assumes every process has an equal number of samples
		# Otherwise it will bias towards a process with the fewest samples
		Average_GN_Hessian = CollectiveOperator(Local_Average_GN_Hessian, self.collective, mpi_op = 'avg')

		x_GN = dl.Vector(self.mesh_constructor_comm)
		GN_Hessians[0].init_vector(x_GN)
		Omega = MultiVector(x_GN,self.parameters['rank'] + self.parameters['oversampling'])

		if self.collective.rank() == 0:
			parRandom.normal(1.,Omega)
			if self.parameters['store_Omega']:
				self.Omega_GN = Omega
		else:
			Omega.zero()
		self.collective.bcast(Omega,root = 0)

		t0 = time.time()

		if prior_preconditioned:
			if hasattr(self.prior, "R"):
				self.d_GN, self.V_GN = doublePassG(Average_GN_Hessian,\
					self.prior.R, self.prior.Rsolver, Omega,self.parameters['rank'],s=1)
			else:
				self.d_GN, self.V_GN = doublePassG(Average_GN_Hessian,\
					self.prior.Hlr, self.prior.Hlr, Omega,self.parameters['rank'],s=1)
		else:
			self.d_GN, self.V_GN = doublePass(Average_GN_Hessian,Omega,self.parameters['rank'],s=1)
		total_init_time = time.time() - t0
		for i in range(100):
			print(80*'#')
			print('Full active subspace took ',total_init_time, 's ')

		self.prior_preconditioned = prior_preconditioned
		self._input_subspace_construction_time = time.time() - t0
		if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):	
			print(('Input subspace construction took '+str(self._input_subspace_construction_time)[:5]+' s').center(80))
		if self.parameters['save_and_plot'] and MPI.COMM_WORLD.rank == 0:
			np.save(self.parameters['output_directory']+'AS_input_projector',mv_to_dense(self.V_GN))
			np.save(self.parameters['output_directory']+'AS_d_GN',self.d_GN)

			out_name = self.parameters['output_directory']+'AS_input_eigenvalues_'+str(self.parameters['rank'])+'.pdf'
			_ = spectrum_plot(self.d_GN,\
				axis_label = ['i',r'$\lambda_i$',\
				r'Eigenvalues of $\mathbb{E}_{\nu}[C{\nabla} q^T {\nabla} q]$'+self.parameters['plot_label_suffix']], out_name = out_name)

	def _construct_serialized_jacobian_subspace(self,prior_preconditioned = True, operation = 'JTJ'):
		"""
		This method implements the input subspace constructor 
			-:code:`prior_preconditioned` - a Boolean to decide whether to include the prior covariance in the decomposition
				The default parameter is True which is customary in active subspace construction
			-:code:`operation` - 
		"""
		t0 = time.time()
		# ms_given is a unit testing case
		if self.parameters['ms_given']:
			assert self.ms is not None
			Local_Average_Jacobian_Operator = SeriallySampledJacobianOperator(self.observable,self.noise,self.prior,operation = operation,\
																				ms = self.ms)
		else:
			Local_Average_Jacobian_Operator = SeriallySampledJacobianOperator(self.observable,self.noise,self.prior,operation = operation,\
																				nsamples = self.parameters['samples_per_process'])
		# This averaging assumes every process has an equal number of samples
		# Otherwise it will bias towards a process with the fewest samples
		Average_Jacobian_Operator = MatrixMultCollectiveOperator(Local_Average_Jacobian_Operator, self.collective, mpi_op = 'avg')
		# Instantiate Gaussian random matrix
		if self.observable.problem.C is None:
			m_mean = self.prior.mean
			u_at_mean = self.observable.problem.generate_state()
			self.observable.problem.solveFwd(u_at_mean,[u_at_mean,m_mean,None])
			self.observable.setLinearizationPoint([u_at_mean,m_mean,None])

		x_Omega_construction = dl.Vector(self.mesh_constructor_comm)

		Local_Average_Jacobian_Operator.init_vector(x_Omega_construction)
		Omega = MultiVector(x_Omega_construction,self.parameters['rank'] + self.parameters['oversampling'])
		Omega.zero()

		if self.collective.rank() == 0:

			if (operation == 'JTJ' and self.Omega_GN is None) or (operation == 'JJT' and self.Omega_NG is None):
				parRandom.normal(1.,Omega)
			else:
				if operation == 'JTJ':
					Omega = self.Omega_GN
				elif operation == 'JJT':
					Omega = self.Omega_NG
				
		self.collective.bcast(Omega,root = 0)

		if operation == 'JTJ':
			if prior_preconditioned:
				if hasattr(self.prior, "R"):
					self.d_GN, self.V_GN = doublePassG(Average_Jacobian_Operator,\
						self.prior.R, self.prior.Rsolver, Omega,self.parameters['rank'],s=1)
				else:
					self.d_GN, self.V_GN = doublePassG(Average_Jacobian_Operator,\
						self.prior.Hlr, self.prior.Hlr, Omega,self.parameters['rank'],s=1)
			else:
				self.d_GN, self.V_GN = doublePass(Average_Jacobian_Operator,Omega,self.parameters['rank'],s=1)
			self.prior_preconditioned = prior_preconditioned
			self._input_subspace_construction_time = time.time() - t0
			if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):	
				print(('Input subspace construction took '+str(self._input_subspace_construction_time)[:5]+' s').center(80))
		elif operation == 'JJT':
			self.d_NG, self.U_NG = doublePass(Average_Jacobian_Operator,Omega,self.parameters['rank'],s=1)
			self._output_subspace_construction_time = time.time() - t0
			if self.parameters['verbose'] and (self.mesh_constructor_comm.rank ==0):	
				print(('Output subspace construction took '+str(self._output_subspace_construction_time)[:5]+' s').center(80))
		
		if self.parameters['save_and_plot'] and MPI.COMM_WORLD.rank == 0:
			if operation == 'JTJ':
				np.save(self.parameters['output_directory']+'AS_input_projector',mv_to_dense(self.V_GN))
				np.save(self.parameters['output_directory']+'AS_d_GN',self.d_GN)
				out_name = self.parameters['output_directory']+'AS_input_eigenvalues_'+str(self.parameters['rank'])+'.pdf'
				_ = spectrum_plot(self.d_GN,\
					axis_label = ['i',r'$\lambda_i$',\
					r'Eigenvalues of $\mathbb{E}_{\nu}[C{\nabla} q^T {\nabla} q]$'+self.parameters['plot_label_suffix']], out_name = out_name)
			if operation == 'JJT':
				np.save(self.parameters['output_directory']+'AS_output_projector',mv_to_dense(self.U_NG))
				np.save(self.parameters['output_directory']+'AS_d_NG',self.d_NG)

				out_name = self.parameters['output_directory']+'AS_output_eigenvalues_'+str(self.parameters['rank'])+'.pdf'
				_ = spectrum_plot(self.d_NG,\
					axis_label = ['i',r'$\lambda_i$',\
					r'Eigenvalues of $\mathbb{E}_{\nu}[{\nabla} q {\nabla} q^T]$'+self.parameters['plot_label_suffix']], out_name = out_name)


	def construct_output_subspace(self):
		if self.parameters['serialized_sampling']:
			pass
		else:
			self._construct_output_subspace_batched()

	def _construct_output_subspace_batched(self):
		"""
		This method implements the output subspace constructor 
		"""
		if self.Js is None:
			self._initialize_batched_samples()

		if self.parameters['verbose']:
			print(80*'#')
			print('Building derivative informed output subspace'.center(80))
		t0 = time.time()
		NG_Hessians = [JJT(J) for J in self.Js]
		Local_Average_NG_Hessian = SummedListOperator(NG_Hessians,average = True)
		# This averaging assumes every process has an equal number of samples
		# Otherwise it will bias towards a process with the fewest samples
		Average_NG_Hessian = CollectiveOperator(Local_Average_NG_Hessian, self.collective, mpi_op = 'avg')

		x_NG = dl.Vector(self.mesh_constructor_comm)
		NG_Hessians[0].init_vector(x_NG)
		Omega = MultiVector(x_NG,self.parameters['rank'] + self.parameters['oversampling'])

		if self.collective.rank() == 0:
			parRandom.normal(1.,Omega)
			if self.parameters['store_Omega']:
				self.Omega_NG = Omega
		else:
			Omega.zero()

		self.collective.bcast(Omega,root = 0)
		self.d_NG, self.U_NG = doublePass(Average_NG_Hessian,Omega,self.parameters['rank'],s=1)
		self._output_subspace_construction_time = time.time() - t0
		if self.parameters['verbose'] and (self.mesh_constructor_comm.rank ==0):	
			print(('Output subspace construction took '+str(self._output_subspace_construction_time)[:5]+' s').center(80))
		if self.parameters['save_and_plot'] and MPI.COMM_WORLD.rank == 0:
			np.save(self.parameters['output_directory']+'AS_output_projector',mv_to_dense(self.U_NG))
			np.save(self.parameters['output_directory']+'AS_d_NG',self.d_NG)

			out_name = self.parameters['output_directory']+'AS_output_eigenvalues_'+str(self.parameters['rank'])+'.pdf'
			_ = spectrum_plot(self.d_NG,\
				axis_label = ['i',r'$\lambda_i$',\
				r'Eigenvalues of $\mathbb{E}_{\nu}[{\nabla} q {\nabla} q^T]$'+self.parameters['plot_label_suffix']], out_name = out_name)

	def construct_low_rank_Jacobians(self,output_directory = 'data/jacobian_data/',check_for_data = True):
		"""
		This method generates low rank Jacobians for training (and also saves input output data in tandem)
			- :code:`output_directory` - a string specifying the path to the directory where data
			will be saved
			- :code:`check_for_data` - a boolean to decide whether to check to see if the training
			data already exists in directory specified by :code:`output_directory`.
		Note that this method also saves the input output data and saves them to a directory that 
		is by default a jacobian_data/ directory
		This allows for separate sampling for l2 loss and h1 seminorm loss
		"""
		if self.Js is None:
			self.initialize_samples()

		my_rank = int(self.collective.rank())
		try:
			os.makedirs(output_directory)
		except:
			pass

		last_datum_generated = 0
		output_dimension,input_dimension = self.Js[0].shape

		rank = min(self.parameters['rank'],output_dimension,input_dimension)
		local_Us = np.zeros((0,output_dimension, rank))	
		local_sigmas = np.zeros((0,rank))
		local_Vs = np.zeros((0,input_dimension, rank))
		local_ms = np.zeros((0,input_dimension))
		local_qs = np.zeros((0,output_dimension))
		# Initialize arrays
		if check_for_data:
			# Save all five or restart sampling and saving
			if os.path.isfile(output_directory+'Us_on_rank_'+str(my_rank)+'.npy') and \
				os.path.isfile(output_directory+'sigmas_on_rank_'+str(my_rank)+'.npy') and \
				os.path.isfile(output_directory+'Vs_on_rank_'+str(my_rank)+'.npy') and \
				os.path.isfile(output_directory+'ms_on_rank_'+str(my_rank)+'.npy') and \
				os.path.isfile(output_directory+'qs_on_rank_'+str(my_rank)+'.npy'):

				local_Us = np.load(output_directory+'Us_on_rank_'+str(my_rank)+'.npy')
				local_sigmas = np.load(output_directory+'sigmas_on_rank_'+str(my_rank)+'.npy')
				local_Vs = np.load(output_directory+'Vs_on_rank_'+str(my_rank)+'.npy')
				local_ms = np.load(output_directory+'ms_on_rank_'+str(my_rank)+'.npy')
				local_qs = np.load(output_directory+'qs_on_rank_'+str(my_rank)+'.npy')

				last_datum_generated = min(local_Us.shape[0],local_sigmas.shape[0],local_Vs.shape[0],\
											local_ms.shape[0],local_qs.shape[0])

				if local_Us.shape[0] > last_datum_generated:
					local_Us = local_Us[:last_datum_generated,:,:]
				if local_sigmas.shape[0] > last_datum_generated:
					local_sigmas = local_sigmas[:last_datum_generated,:]
				if local_Vs.shape[0] > last_datum_generated:
					local_Vs = local_Vs[:last_datum_generated,:,:]
				if local_ms.shape[0] > last_datum_generated:
					local_ms = local_ms[:last_datum_generated,:]
				if local_qs.shape[0] > last_datum_generated:
					local_qs = local_qs[:last_datum_generated,:]


		# Generate the input output pairs that correspond to the 
		assert len(self.ms) == self.parameters['samples_per_process']
		assert len(self.us) == self.parameters['samples_per_process']
		# If the us are updated in place then create a method for the observable that just applies B to u
		t0_mq = time.time()
		# Then here we can just retrieve m and q = Bu and save as numpy arrays
		# with the same ordering as with the Jacobian data.
		for i in range(last_datum_generated,self.parameters['samples_per_process']):
			if self.parameters['verbose']:
				print('Saving input output data pair '+str(i))

			local_ms = np.concatenate((local_ms,np.expand_dims(self.ms[i].get_local(),0)))
			qi = self.observables[i].evalu(self.us[i]).get_local()
			local_qs = np.concatenate((local_qs,np.expand_dims(qi,0)))
			np.save(output_directory+'ms_on_rank_'+str(my_rank)+'.npy',np.array(local_ms))
			np.save(output_directory+'qs_on_rank_'+str(my_rank)+'.npy',np.array(local_qs))
			if self.parameters['verbose']:
				print('On datum saved every ',(time.time() -t0_mq)/(i - last_datum_generated+1),' s, on average.')


		# Initialize randomized Omega
		input_vector = dl.Vector(self.mesh_constructor_comm)
		self.Js[0].init_vector(input_vector,1)
		Omega = MultiVector(input_vector,rank + self.parameters['oversampling'])
		# Omega does not need to be communicated across processes in this case
		# like with the global reduction collectives
		parRandom.normal(1.,Omega)

		t0 = time.time()
		# I think this is all hard coded for a single serial mesh, check if 
		# the arrays need to be communicated to mesh rank 0 before being saved

		# Here the number of data generated are the length of the Jacobians
		# This should be true by default but if self.Js are manipulated that could
		# change
		assert len(self.Js) == self.parameters['samples_per_process']
		for i in range(last_datum_generated,self.parameters['samples_per_process']):
			if self.parameters['verbose']:
				print('Generating Jacobian data number '+str(i))
			# Reusing Omega for each randomized pass, this shouldn't be an issue,
			# but one could resample at each iteration

			U, sigma, V = accuracyEnhancedSVD(self.Js[i],Omega,rank,s=1)

			local_Us = np.concatenate((local_Us,np.expand_dims(mv_to_dense(U),0)))
			local_sigmas = np.concatenate((local_sigmas,np.expand_dims(sigma,0)))
			local_Vs = np.concatenate((local_Vs,np.expand_dims(mv_to_dense(V),0)))



			if self.mesh_constructor_comm.rank == 0:
				np.save(output_directory+'Us_on_rank_'+str(my_rank)+'.npy',np.array(local_Us))
				np.save(output_directory+'sigmas_on_rank_'+str(my_rank)+'.npy',np.array(local_sigmas))
				np.save(output_directory+'Vs_on_rank_'+str(my_rank)+'.npy',np.array(local_Vs))
			if self.parameters['verbose']:
				print('On Jacobian datum generated every ',(time.time() -t0)/(i - last_datum_generated+1),' s, on average.')


		if True:
			out_name = self.parameters['output_directory']+'jacobian_singular_values_'+str(rank)+'.pdf'
			plot_singular_values_with_std(np.mean(local_sigmas,axis=0),np.std(local_sigmas,axis=0),outname= out_name)

		self._jacobian_data_generation_time = time.time() - t0



	def test_errors(self,test_input = True, test_output = False, ranks = [None],cut_off = 1e-12):
		"""
		This method implements an error test
			- :code:`test_input` - a Boolean for whether input tests are executed
			- :code:`test_output` - a Boolean for whether output tests are executed
			- :code:`ranks` - a python list of integers specifying ranks for projection tests
			- :code:`cut_off` - Where to truncate the ranks based on eigenvalue decay
		"""
		global_avg_rel_errors_input, global_avg_rel_errors_output = None, None
		global_std_rel_errors_input, global_std_rel_errors_output = None, None
		# ranks assumed to be python list with sort in place member function
		ranks.sort()
		if test_input:
			# Simple projection test
			if self.d_GN is None:
				if self.mesh_constructor_comm.rank == 0:
					print('Constructing input subspace')
				self.construct_input_subspace()
			elif len(self.d_GN)<ranks[-1]:
				if self.mesh_constructor_comm.rank == 0:
					print('Constructing input subspace because larger rank needed.')
					self.parameters['rank'] = ranks[-1]
				self.construct_input_subspace()
			else: 
				if self.mesh_constructor_comm.rank == 0:
					print('Input subspace already computed proceeding with error tests')
			# truncate eigenvalues for numerical stability
			numericalrank = np.where(self.d_GN > cut_off)[-1][-1] + 1 # due to 0 indexing
			ranks = ranks[:np.where(ranks <= numericalrank)[0][-1]+1]# due to inclusion
			global_avg_rel_errors_input = np.ones_like(ranks,dtype = np.float64)
			global_std_rel_errors_input = np.zeros_like(ranks,dtype = np.float64)

			# Naive test on output space
			LocalParameters = MultiVector(self.ms[0],self.parameters['error_test_samples'])
			LocalParameters.zero()
			# Generate samples
			for i in range(self.parameters['error_test_samples']):
				t0 = time.time()
				parRandom.normal(1,self.noise)
				self.prior.sample(self.noise,LocalParameters[i])

			LocalErrors = MultiVector(self.ms[0],self.parameters['error_test_samples'])
			projection_vector = self.observable.generate_vector(PARAMETER) 

			for rank_index,rank in enumerate(ranks):
				LocalErrors.zero()
				if rank is None:
					V_GN = self.V_GN
					d_GN = self.d_GN
				else:
					V_GN = MultiVector(self.V_GN[0],rank)
					d_GN = self.d_GN[0:rank]
					for i in range(rank):
						V_GN[i].axpy(1.,self.V_GN[i])
				input_init_vector_lambda = lambda x, dim: self.observable.init_vector(x,dim = 1)
				if self.prior_preconditioned:
					InputProjectorOperator = PriorPreconditionedProjector(V_GN,self.prior.R, input_init_vector_lambda)
				else:
					InputProjectorOperator = LowRankOperator(np.ones_like(d_GN),V_GN, input_init_vector_lambda)
			
				rel_errors = np.zeros(LocalErrors.nvec())
				for i in range(LocalErrors.nvec()):
					LocalErrors[i].axpy(1.,LocalParameters[i])
					denominator = LocalErrors[i].norm('l2')
					projection_vector.zero()
					InputProjectorOperator.mult(LocalErrors[i],projection_vector)
					LocalErrors[i].axpy(-1.,projection_vector)
					numerator = LocalErrors[i].norm('l2')
					rel_errors[i] = numerator/denominator

				avg_rel_error = np.mean(rel_errors)
				std_rel_error_squared = np.std(rel_errors)**2
				global_avg_rel_errors_input[rank_index] = self.collective.allReduce(avg_rel_error,'avg')
				global_avg_rel_errors_input[rank_index] = np.sqrt(self.collective.allReduce(std_rel_error_squared,'avg'))
				if self.mesh_constructor_comm.rank == 0:
					print('Naive global average relative error input = ',global_avg_rel_errors_input[rank_index],' for rank ',rank)

			# Double Loop MC Error test does not work when prior preconditioning is used.
			# This will be fixed soon.
			if False:
				if self.d_GN is None:
					if self.mesh_constructor_comm.rank == 0:
						print('Constructing input subspace')
					self.construct_input_subspace()
				elif len(self.d_GN)<ranks[-1]:
					if self.mesh_constructor_comm.rank == 0:
						print('Constructing input subspace because larger rank needed.')
						self.parameters['rank'] = ranks[-1]
					self.construct_input_subspace()
				else: 
					if self.mesh_constructor_comm.rank == 0:
						print('Input subspace already computed proceeding with error tests')
				# truncate eigenvalues for numerical stability
				numericalrank = np.where(self.d_GN > cut_off)[-1][-1] + 1 # due to 0 indexing
				if numericalrank < ranks[-1]:
					ranks = ranks+[numericalrank]
				else:
					ranks = ranks[:np.where(ranks <= numericalrank)[0][-1]+1]
				double_loop_global_avg_rel_errors_input = np.ones_like(ranks,dtype = np.float64)
				double_loop_global_std_rel_errors_input = np.zeros_like(ranks,dtype = np.float64)

				# Double loop MC error approximation
				# Instantiate input and output data arrays
				observable_vector = dl.Vector(self.mesh_constructor_comm)
				self.observable.init_vector(observable_vector,dim = 0)
				LocalObservables = MultiVector(observable_vector,self.parameters['error_test_samples'])
				LocalObservables.zero()
				LocalParameters = MultiVector(self.ms[0],self.parameters['error_test_samples'])
				LocalParameters.zero()
				# Generate samples
				for i in range(self.parameters['error_test_samples']):
					t0 = time.time()
					parRandom.normal(1,self.noise)
					self.prior.sample(self.noise,LocalParameters[i])
					LocalObservables[i].axpy(1.,self.observable.eval(LocalParameters[i],setLinearizationPoint = True))
					if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):
						print('Generating local observable ',i,' for input error test took',time.time() -t0, 's')

				# Instantiate array for errors
				LocalErrors = MultiVector(observable_vector,self.parameters['error_test_samples'])

				m_r = dl.Vector(self.mesh_constructor_comm)
				self.observable.init_vector(m_r,dim = 1)
				y = dl.Vector(self.mesh_constructor_comm)
				self.observable.init_vector(y,dim = 1)
				y_r = dl.Vector(self.mesh_constructor_comm)
				self.observable.init_vector(y_r,dim = 1)

				for rank_index,rank in enumerate(ranks):
					if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):
						print('Beginning double loop MC for rank', rank)
					LocalErrors.zero()

					if rank is None:
						V_GN = self.V_GN
						d_GN = self.d_GN
					else:
						V_GN = MultiVector(self.V_GN[0],rank)
						V_GN.zero()
						d_GN = self.d_GN[0:rank]
						for i in range(rank):
							V_GN[i].axpy(1.,self.V_GN[i])
					input_init_vector_lambda = lambda x, dim: self.observable.init_vector(x,dim = 1)
					if self.prior_preconditioned:
						InputProjectorOperator = PriorPreconditionedProjector(V_GN,self.prior.R, input_init_vector_lambda)
					else:
						InputProjectorOperator = LowRankOperator(np.ones_like(d_GN),V_GN, input_init_vector_lambda)
					print('Constructed projection operator for rank ',rank)
					rel_errors = np.zeros(LocalErrors.nvec())

					for i in range(LocalErrors.nvec()):
						if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):
							print('Beginning outer loop i = ',i)
						LocalErrors[i].axpy(1.,LocalObservables[i])
						denominator = LocalErrors[i].norm('l2')
						m_r.zero()
						InputProjectorOperator.mult(LocalParameters[i],m_r)
						for j in range(self.parameters['double_loop_samples']):
							# Approximation of conditional expectation of qoi
							discarded_samples = 0
							t0 = time.time()
							y.zero()
							y_r.zero()
							parRandom.normal(1,self.noise)
							self.prior.sample(self.noise,y)
							InputProjectorOperator.mult(y,y_r)
							y.axpy(-1., y_r)
							try:
								LocalErrors[i].axpy(-1./float(self.parameters['double_loop_samples']),self.observable.eval(m_r + y,setLinearizationPoint = True))
							except:
								print('Issue solving PDE at truncated parameter (may be due to smoothness), discarding'.center(80))
								discarded_samples +=1

							if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):
								print(('Inner loop i,j = '+str((i,j))+' took '+str(time.time() - t0)+'s').center(80))
								print(('Discarded samples = '+str(discarded_samples)).center(80))
						# Rescale to account for discarded samples
						if discarded_samples > 0:
							rescale_factor = float(self.parameters['double_loop_samples'])/(float(self.parameters['double_loop_samples']) - float(discarded_samples))
							LocalErrors[i] *= rescale_factor

						numerator = LocalErrors[i].norm('l2')
						rel_errors[i] = numerator/denominator
						if numerator > denominator:
							for k in range(10):
								print(80*'#')
								print('Issue'.center(80))
								print('numerator = ',numerator)
								print('denominator = ',denominator)

					avg_rel_error = np.mean(rel_errors)
					std_rel_error_squared = np.std(rel_errors)**2
					double_loop_global_avg_rel_errors_input[rank_index] = self.collective.allReduce(avg_rel_error,'avg')
					double_loop_global_std_rel_errors_input[rank_index] = np.sqrt(self.collective.allReduce(std_rel_error_squared,'avg'))
					if self.mesh_constructor_comm.rank == 0:
						print('Double loop MC global average relative error input = ',double_loop_global_avg_rel_errors_input[rank_index],' for rank ',rank)
					self._double_loop_errors = double_loop_global_avg_rel_errors_input
					self._double_loop_stds = double_loop_global_std_rel_errors_input


		if test_output:
			if self.d_NG is None:
				if self.mesh_constructor_comm.rank == 0:
					print('Constructing output subspace')
				self.construct_output_subspace()
			elif len(self.d_NG)<ranks[-1]:
				if self.mesh_constructor_comm.rank == 0:
					print('Constructing output subspace because larger rank needed.')
					self.parameters['rank'] = ranks[-1]
				self.construct_output_subspace()
			else: 
				if self.mesh_constructor_comm.rank == 0:
					print('Output subspace already computed proceeding with error tests')
			# truncate eigenvalues for numerical stability
			numericalrank = np.where(self.d_NG > cut_off)[-1][-1] + 1 # due to 0 indexing
			ranks = ranks[:np.where(ranks <= numericalrank)[0][-1]+1]# due to inclusion
			global_avg_rel_errors_output = np.ones_like(ranks,dtype = np.float64)
			global_std_rel_errors_output = np.zeros_like(ranks,dtype = np.float64)
			# Naive test on output space
			observable_vector = dl.Vector(self.mesh_constructor_comm)
			self.observable.init_vector(observable_vector,dim = 0)
			LocalObservables = MultiVector(observable_vector,self.parameters['error_test_samples'])
			LocalObservables.zero()
			for i in range(LocalObservables.nvec()):
				t0 = time.time()
				parRandom.normal(1,self.noise)
				self.prior.sample(self.noise,self.ms[0])
				x = [self.us[0],self.ms[0],None]
				self.observable.setLinearizationPoint(x)
				LocalObservables[i].axpy(1.,self.observable.eval(self.ms[0]))
				if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):
					print('Generating local observable ',i, ' for output error test')

			LocalErrors = MultiVector(observable_vector,self.parameters['error_test_samples'])
			projection_vector = dl.Vector(self.mesh_constructor_comm)
			self.observable.init_vector(projection_vector,dim = 0)

			for rank_index,rank in enumerate(ranks):
				LocalErrors.zero()
				if rank is None:
					U_NG = self.U_NG
					d_NG = self.d_NG
				else:
					U_NG = MultiVector(self.U_NG[0],rank)
					d_NG = self.d_NG[0:rank]
					for i in range(rank):
						U_NG[i].axpy(1.,self.U_NG[i])
				output_init_vector_lambda = lambda x, dim: self.observable.init_vector(x,dim = 0)
				OutputProjectorOperator = LowRankOperator(np.ones_like(d_NG),U_NG,output_init_vector_lambda)
			
				rel_errors = np.zeros(LocalErrors.nvec())
				for i in range(LocalErrors.nvec()):
					LocalErrors[i].axpy(1.,LocalObservables[i])
					denominator = LocalErrors[i].norm('l2')
					projection_vector.zero()
					OutputProjectorOperator.mult(LocalErrors[i],projection_vector)
					LocalErrors[i].axpy(-1.,projection_vector)
					numerator = LocalErrors[i].norm('l2')
					rel_errors[i] = numerator/denominator

				avg_rel_error = np.mean(rel_errors)
				std_rel_error_squared = np.std(rel_errors)**2
				global_avg_rel_errors_output[rank_index] = self.collective.allReduce(avg_rel_error,'avg')
				global_std_rel_errors_output[rank_index] = np.sqrt(self.collective.allReduce(std_rel_error_squared,'avg'))
				if self.mesh_constructor_comm.rank == 0:
					print('Global average relative error output = ',global_avg_rel_errors_output[rank_index],' for rank ',rank)

		if False and (MPI.COMM_WORLD.size == 1) and (self.d_GN is not None) and (self.d_NG is not None):
			spectral_error = self.d_GN - self.d_NG

			print("[world rank {:d}] ".format(MPI.COMM_WORLD.rank)+"[mesh rank {:d}] ".format(self.mesh_constructor_comm.rank)+\
			"[sample rank {:d}] ".format(self.collective.rank())+ '|| d_GN_avg - d_NG_avg ||  = ' + str(np.linalg.norm(spectral_error)))

		return [global_avg_rel_errors_input, global_std_rel_errors_input], [global_avg_rel_errors_output, global_std_rel_errors_output]
			
