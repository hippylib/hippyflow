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
import os
import hippylib as hp
from mpi4py import MPI 
import time

from ..collectives.collectiveOperator import CollectiveOperator, MatrixMultCollectiveOperator
from ..collectives.collective import NullCollective
from ..collectives.comm_utils import checkMeshConsistentPartitioning
from .jacobian import *
from .controlJacobian import ObservableControlJacobian
from ..utilities.mv_utilities import mv_to_dense
from ..utilities.plotting import *

from .priorPreconditionedProjector import PriorPreconditionedProjector

CONTROL = 3

def ActiveSubspaceParameterList():
	"""
	This function implements a parameter list for the ActiveSubspaceProjector
	"""
	parameters = {}
	parameters['samples_per_process'] 		= [64, 'Number of samples per process']
	parameters['jacobian_data_per_process'] = [512, 'Number of samples per process']
	parameters['error_test_samples'] 		= [50, 'Number of samples for error test']
	parameters['rank'] 				 		= [128, 'Rank of subspace']
	parameters['jacobian_rank']				= [128, 'Rank of Jacobians generated']
	parameters['control_jacobian_rank'] 	= [None, 'Rank of control Jacobians generated']
	parameters['oversampling'] 		 		= [10, 'Oversampling parameter for randomized algorithms']
	parameters['double_loop_samples']		= [20, 'Number of samples used in double loop MC approximation']
	parameters['verbose']					= [True, 'Boolean for printing']

	parameters['input_decoder_name']			= ['_input_decoder', 'string for naming']
	parameters['output_decoder_name']			= ['_output_decoder', 'string for naming']



	parameters['initialize_samples'] 		= [False,'Boolean for the initialization of samples when\
														many samples are allocated on one process ']
	parameters['serialized_sampling']		= [True, 'Boolean for the serialization of sampling on a process\
													 to reduce memory for large problems']

	parameters['observable_constructor'] 	= [None,'observable constructor function, assumed to take a mesh, and kwargs']
	parameters['observable_kwargs'] 		= [{},'kwargs used when instantiating multiple local instances of observables']

	parameters['output_directory']			= [None,'output directory for saving arrays and plots']
	parameters['plot_label_suffix']			= ['', 'suffix for plot label']
	parameters['save_and_plot']				= [True,'Boolean for saving data and plots (only False for unit testing)']
	parameters['store_Omega']				= [False,'Boolean for storing Gaussian random matrix (only True for unit testing)']
	parameters['ms_given']					= [False,'Boolean for passing ms into serialized AS construction (only True for unit testing)']
	return hp.ParameterList(parameters)


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
	def __init__(self,observable,noise,prior,control_distribution = None,operation = 'JTJ',\
							nsamples = None,ms = None,zs = None,communicator=None,average=True):
		'''
		'''
		assert operation in ['JTJ','JJT']
		assert (nsamples is not None) or (ms is not None)
		self.observable = observable
		self.noise = noise
		self.prior = prior
		self.control_distribution = control_distribution
		self.operation = operation
		self.nsamples = nsamples
		self.average = average
		self.ms = ms
		if zs is not None:
			self.zs = zs
		else:
			if type(self.ms) is list:
				# This is a unit-test case
				self.zs = len(self.ms)*[None]
			else: 
				self.zs = zs

		
		if communicator is None:
			self.temp = None
		else:
			self.temp = dl.Vector(communicator)

		self.u = self.observable.generate_vector(hp.STATE)
		if hasattr(self.observable.problem,'parameter_projection'):
			if communicator is not None:
				self.m = dl.Vector(communicator)
			else:
				self.m = dl.Vector()
			self.prior.init_vector(self.m,0)
		else:
			self.m = self.observable.generate_vector(hp.PARAMETER)
		if self.control_distribution is None:
			self.z = None
		else:
			self.z = self.observable.generate_vector(CONTROL)

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
			if hasattr(self.observable.problem,'parameter_projection'):
				self.prior.init_vector(x,0)
			else:
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
				# Iterate if there are solver issues
				solved = False
				while not solved:
					iterate = 0
					try:
						# Sample from the prior
						self.m.zero()
						self.noise.zero()
						hp.parRandom.normal(1,self.noise)
						self.prior.sample(self.noise,self.m)
						if self.control_distribution is None:
							linearization_x = [self.u,self.m,None]
						else:
							self.z.zero()
							self.control_distribution.sample(self.z)
							linearization_x = [self.u,self.m,None,self.z]
						if hasattr(self.observable.problem,'parameter_projection'):
							# In this case the parameter needs to be projected to subdomain
							m_sample = self.observable.problem.parameter_projection(self.m)
							linearization_x[1] = m_sample

						# Solve the PDE
						print('Attempting to solve')
						t0 = time.time()
						self.observable.solveFwd(self.u,linearization_x)
						print('Solution succesful, and took ',time.time() - t0,'s') 
						# set linearization point
						self.observable.setLinearizationPoint(linearization_x)
						solved = True
					except:
						print('Issue with the solution, moving on')
						iterate += 1
					if iterate > 100:
						print('Some sort of issue, no infinite loop allowed (+:')
				# Define action on matrix (as represented by hp.MultiVector)
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
			for m,z in zip(self.ms,self.zs):
				# Solve the PDE
				print('Attempting to solve')
				linearization_x = [self.u,m,None]
				if z is not None:
					linearization_x.append(z)
				self.observable.solveFwd(self.u,linearization_x)
				print('Solution succesful')
				# set linearization point
				self.observable.setLinearizationPoint(linearization_x)
				solved = True
				# Define action on matrix (as represented by hp.MultiVector)
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
	def __init__(self,observable, prior, control_distribution = None, mesh_constructor_comm = None ,collective = NullCollective(),\
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
		self.control_distribution = control_distribution


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
			self.z = None
			self.Jz = None
			

		else:		
			self.us = None
			self.ms = None
			self.Js = None
			self.zs = None
			self.Jzs = None
			

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
		self.us = [self.observable.generate_vector(hp.STATE) for i in range(self.parameters['samples_per_process'])]
		self.ms = [self.observable.generate_vector(hp.PARAMETER) for i in range(self.parameters['samples_per_process'])]
		if self.control_distribution is not None:
			self.zs = [self.observable.generate_vector(CONTROL) for i in range(self.parameters['samples_per_process'])]
		else:
			self.zs = self.parameters['samples_per_process']*[None]
		for u,m,z,observable in zip(self.us,self.ms,self.zs,self.observables):
			solved = False
			while not solved:
				try:
					self.noise.zero()
					hp.parRandom.normal(1,self.noise)
					# set linearization point
					self.prior.sample(self.noise,m)
					if self.control_distribution is not None:
						self.control_distribution.sample(z)
						x = [u,m,None,z]
					else:
						x = [u,m,None]
					if hasattr(self.observable.problem,'parameter_projection'):
						m_sample = self.observable.problem.parameter_projection(m)
						x[1] = m_sample
					observable.solveFwd(u,x)
					observable.setLinearizationPoint(x)
					solved = True
				except:
					m.zero()
					z.zero()
					print('Issue with the solution, moving on')
					pass
		if self.parameters['verbose']:
			try:
				from pympler import asizeof
				print(('Size of one observable is '+str(asizeof.asizeof(self.observables[0])/1e6)+' MB').center(80))
				print(('Size of '+str(self.parameters['samples_per_process'])+' observable is '+str(asizeof.asizeof(self.observables)/1e6)+' MB').center(80))
			except:
				print('Install pympler and run again: pip install pympler'.center(80))
		# This should be the same independent of whether their is a control variable
		# dependence or not, since this Jacobian is just for the parameter.
		# When adding mixed derivatives the Jacobian operator additionally needs
		# to handle control adjoints.
		self.Js = [ObservableJacobian(observable) for observable in self.observables]
		total_init_time = time.time() - t0
		for i in range(100):
			print(80*'#')
			print('Initializing all batched samples took ',total_init_time, 's ')


	def construct_input_subspace(self,prior_preconditioned = True,name_suffix = None):
		if self.parameters['serialized_sampling']:
			print('Construction via serialized AS construction'.center(80))
			return self._construct_serialized_jacobian_subspace(prior_preconditioned = prior_preconditioned,operation = 'JTJ')
		else:
			print('Construction via batched AS construction'.center(80))
			return self._construct_input_subspace_batched(prior_preconditioned = prior_preconditioned,name_suffix = name_suffix)




	def _construct_input_subspace_batched(self,prior_preconditioned = True,name_suffix = None):
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
		Omega = hp.MultiVector(x_GN,self.parameters['rank'] + self.parameters['oversampling'])

		if self.collective.rank() == 0:
			hp.parRandom.normal(1.,Omega)
			if self.parameters['store_Omega']:
				self.Omega_GN = Omega
		else:
			Omega.zero()
		self.collective.bcast(Omega,root = 0)

		t0 = time.time()

		if prior_preconditioned:
			if hasattr(self.prior, "R"):
				self.d_GN, self.V_GN = hp.doublePassG(Average_GN_Hessian,\
					self.prior.R, self.prior.Rsolver, Omega,self.parameters['rank'],s=1)
				as_decoder = self.V_GN
				as_encoder = hp.MultiVector(as_decoder)
				hp.MatMvMult(self.prior.R,as_decoder,as_encoder)
			else:
				self.d_GN, self.V_GN = hp.doublePassG(Average_GN_Hessian,\
					self.prior.Hlr, self.prior.Hlr, Omega,self.parameters['rank'],s=1)
				as_decoder = self.V_GN
				as_encoder = hp.MultiVector(as_decoder)
				hp.MatMvMult(self.prior.Hlr,as_decoder,as_encoder)
		else:
			self.d_GN, self.V_GN = hp.doublePass(Average_GN_Hessian,Omega,self.parameters['rank'],s=1)
			as_decoder = self.V_GN
			as_encoder = hp.MultiVector(as_decoder)

		total_init_time = time.time() - t0
		for i in range(100):
			print(80*'#')
			print('Full active subspace took ',total_init_time, 's ')

		self.prior_preconditioned = prior_preconditioned
		self._input_subspace_construction_time = time.time() - t0
		if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):	
			print(('Input subspace construction took '+str(self._input_subspace_construction_time)[:5]+' s').center(80))
		if self.parameters['save_and_plot'] and MPI.COMM_WORLD.rank == 0:
			name = 'AS_'+str(int(self.parameters['samples_per_process']*self.collective.size()))
			if name_suffix is not None:
				assert type(name_suffix) is str
				name += name_suffix
			np.save(self.parameters['output_directory']+name+self.parameters['input_decoder_name'],mv_to_dense(self.V_GN))
			np.save(self.parameters['output_directory']+name+'_d_GN',self.d_GN)

			plot_out_name = self.parameters['output_directory']+name+'_input_eigenvalues_'+str(self.parameters['rank'])+'.pdf'
			_ = spectrum_plot(self.d_GN,\
				axis_label = ['i',r'$\lambda_i$',\
				r'Eigenvalues of $\mathbb{E}_{\nu}[C{\nabla} q^T {\nabla} q]$'+self.parameters['plot_label_suffix']], out_name = plot_out_name)

		return self.d_GN, as_decoder, as_encoder

	def _construct_serialized_jacobian_subspace(self,prior_preconditioned = True, operation = 'JTJ',name_suffix = None):
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
			if self.control_distribution is not None:
				assert self.zs is not None
				assert self.zs[0] is not None
			Local_Average_Jacobian_Operator = SeriallySampledJacobianOperator(self.observable,self.noise,self.prior,operation = operation,\
																				ms = self.ms,zs = self.zs)
		else:
			Local_Average_Jacobian_Operator = SeriallySampledJacobianOperator(self.observable,self.noise,self.prior,\
														 control_distribution = self.control_distribution,operation = operation,\
																				nsamples = self.parameters['samples_per_process'])
		# This averaging assumes every process has an equal number of samples
		# Otherwise it will bias towards a process with the fewest samples
		Average_Jacobian_Operator = MatrixMultCollectiveOperator(Local_Average_Jacobian_Operator, self.collective, mpi_op = 'avg')
		
		if self.observable.problem.C is None:
			# If this is the case, then the KKT blocks have not been built yet
			# we overcome this by solving somewhere and setting the linearization pt.
			m_mean = self.prior.mean
			if hasattr(self.observable.problem,'parameter_projection'):
				m_mean = self.observable.problem.parameter_projection(m_mean)

			u_at_mean = self.observable.problem.generate_state()
			if self.control_distribution is not None:
				if hasattr(self.control_distribution,'mean'):
					z_mean = self.control_distribution.mean
				else:
					# Sample it somewhere
					z_mean = self.observable.generate_vector(CONTROL)
					self.control_distribution.sample(z_mean)
				x_lin = [u_at_mean,m_mean,None,z_mean]
			else:
				x_lin = [u_at_mean,m_mean,None]
			
			self.observable.problem.solveFwd(u_at_mean,x_lin)
			self.observable.setLinearizationPoint(x_lin)

		# Instantiate Gaussian random matrix
		x_Omega_construction = dl.Vector(self.mesh_constructor_comm)
		Local_Average_Jacobian_Operator.init_vector(x_Omega_construction)
		Omega = hp.MultiVector(x_Omega_construction,self.parameters['rank'] + self.parameters['oversampling'])
		Omega.zero()

		if self.collective.rank() == 0:

			if (operation == 'JTJ' and self.Omega_GN is None) or (operation == 'JJT' and self.Omega_NG is None):
				hp.parRandom.normal(1.,Omega)
			else:
				if operation == 'JTJ':
					Omega = self.Omega_GN
				elif operation == 'JJT':
					Omega = self.Omega_NG
				
		self.collective.bcast(Omega,root = 0)

		if operation == 'JTJ':
			if prior_preconditioned:
				if hasattr(self.prior, "R"):
					self.d_GN, self.V_GN = hp.doublePassG(Average_Jacobian_Operator,\
						self.prior.R, self.prior.Rsolver, Omega,self.parameters['rank'],s=1)
					as_decoder = self.V_GN
					as_encoder = hp.MultiVector(as_decoder)
					hp.MatMvMult(self.prior.R,as_decoder,as_encoder)
				else:
					self.d_GN, self.V_GN = hp.doublePassG(Average_Jacobian_Operator,\
						self.prior.Hlr, self.prior.Hlr, Omega,self.parameters['rank'],s=1)
					as_decoder = self.V_GN
					as_encoder = hp.MultiVector(as_decoder)
					hp.MatMvMult(self.prior.Hlr,as_decoder,as_encoder)
			else:
				self.d_GN, self.V_GN = hp.doublePass(Average_Jacobian_Operator,Omega,self.parameters['rank'],s=1)
				as_decoder = self.V_GN
				as_encoder = hp.MultiVector(as_decoder)
			self.prior_preconditioned = prior_preconditioned
			self._input_subspace_construction_time = time.time() - t0
			if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):	
				print(('Input subspace construction took '+str(self._input_subspace_construction_time)[:5]+' s').center(80))

		elif operation == 'JJT':
			self.d_NG, self.U_NG = hp.doublePass(Average_Jacobian_Operator,Omega,self.parameters['rank'],s=1)
			output_decoder = self.U_NG
			output_encoder = hp.MultiVector(output_decoder)

			self._output_subspace_construction_time = time.time() - t0
			if self.parameters['verbose'] and (self.mesh_constructor_comm.rank ==0):	
				print(('Output subspace construction took '+str(self._output_subspace_construction_time)[:5]+' s').center(80))
			

		if self.parameters['save_and_plot'] and MPI.COMM_WORLD.rank == 0:
			name = 'AS_'+str(int(self.parameters['samples_per_process']*self.collective.size()))
			if name_suffix is not None:
				assert type(name_suffix) is str
				name += name_suffix
			if operation == 'JTJ':
				np.save(self.parameters['output_directory']+name+self.parameters['input_decoder_name'],mv_to_dense(self.V_GN))
				np.save(self.parameters['output_directory']+name+'_d_GN',self.d_GN)
				plot_out_name = self.parameters['output_directory']+name+'_input_eigenvalues_'+str(self.parameters['rank'])+'.pdf'
				try:
					_ = spectrum_plot(self.d_GN,\
						axis_label = ['i',r'$\lambda_i$',\
						r'Eigenvalues of $\mathbb{E}_{\nu}[C{\nabla} q^T {\nabla} q]$'+self.parameters['plot_label_suffix']], out_name = plot_out_name)
				except:
					print('Issue plotting, probably latex related')
			if operation == 'JJT':
				np.save(self.parameters['output_directory']+name+self.parameters['output_decoder_name'],mv_to_dense(self.U_NG))
				np.save(self.parameters['output_directory']+name+'_d_NG',self.d_NG)

				plot_out_name = self.parameters['output_directory']+name+'_output_eigenvalues_'+str(self.parameters['rank'])+'.pdf'
				try:
					_ = spectrum_plot(self.d_NG,\
						axis_label = ['i',r'$\lambda_i$',\
						r'Eigenvalues of $\mathbb{E}_{\nu}[{\nabla} q {\nabla} q^T]$'+self.parameters['plot_label_suffix']], out_name = plot_out_name)
				except:
					print('Issue plotting, probably, latex related')

		if operation == 'JTJ':
			return self.d_GN, as_decoder, as_encoder
		elif operation == 'JJT':
			return self.d_NG, output_decoder, output_encoder

	def construct_output_subspace(self,name_suffix = None):
		if self.parameters['serialized_sampling']:
			print('Construction via serialized construction'.center(80))
			return self._construct_serialized_jacobian_subspace(operation = 'JJT',name_suffix = name_suffix)
		else:
			return self._construct_output_subspace_batched(name_suffix = name_suffix)

	def _construct_output_subspace_batched(self,name_suffix = None):
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
		Omega = hp.MultiVector(x_NG,self.parameters['rank'] + self.parameters['oversampling'])

		if self.collective.rank() == 0:
			hp.parRandom.normal(1.,Omega)
			if self.parameters['store_Omega']:
				self.Omega_NG = Omega
		else:
			Omega.zero()

		self.collective.bcast(Omega,root = 0)
		self.d_NG, self.U_NG = hp.doublePass(Average_NG_Hessian,Omega,self.parameters['rank'],s=1)
		output_decoder = self.U_NG
		output_encoder = hp.MultiVector(output_decoder)
		self._output_subspace_construction_time = time.time() - t0
		if self.parameters['verbose'] and (self.mesh_constructor_comm.rank ==0):	
			print(('Output subspace construction took '+str(self._output_subspace_construction_time)[:5]+' s').center(80))
		if self.parameters['save_and_plot'] and MPI.COMM_WORLD.rank == 0:
			name = 'AS_'+str(int(self.parameters['samples_per_process']*self.collective.size()))
			if name_suffix is not None:
				assert type(name_suffix) is str
				name += name_suffix
			np.save(self.parameters['output_directory']+name+self.parameters['output_decoder_name'],mv_to_dense(self.U_NG))
			np.save(self.parameters['output_directory']+name+'_d_NG',self.d_NG)

			plot_out_name = self.parameters['output_directory']+name+'_output_eigenvalues_'+str(self.parameters['rank'])+'.pdf'
			_ = spectrum_plot(self.d_NG,\
				axis_label = ['i',r'$\lambda_i$',\
				r'Eigenvalues of $\mathbb{E}_{\nu}[{\nabla} q {\nabla} q^T]$'+self.parameters['plot_label_suffix']], out_name = plot_out_name)

		return self.d_NG, output_decoder, output_encoder


	def construct_low_rank_Jacobians(self,check_for_data = True,compress_files = True):
		if self.parameters['serialized_sampling']:
			self._construct_low_rank_Jacobians_serial(check_for_data = check_for_data,compress_files = compress_files)
		else:
			self._construct_low_rank_Jacobians_batched(check_for_data = check_for_data)

	def construct_low_rank_control_Jacobians(self,check_for_data = True,compress_files = True):
		if self.parameters['serialized_sampling']:
			self._construct_low_rank_Jacobians_serial(check_for_data = check_for_data,compress_files = compress_files,\
					parameter_jacobian = False,control_jacobian = True)
		else:
			raise
			self._construct_low_rank_Jacobians_batched(check_for_data = check_for_data, control_jacobian = True)

	def _construct_low_rank_Jacobians_serial(self,check_for_data = True,compress_files = True,\
						parameter_jacobian = True, control_jacobian = False):
		"""
		This method generates low rank Jacobians for training (and also saves input output data in tandem)
			- :code:`check_for_data` - a boolean to decide whether to check to see if the training
			data already exists in directory specified by :code:`output_directory`.
		Note that this method also saves the input output data and saves them to a directory that 
		is by default a jacobian_data/ directory
		This allows for separate sampling for l2 loss and h1 seminorm loss
		"""
		if control_jacobian:
			assert (self.control_distribution is not None)

		parameter_dimension = None

		proc_id = int(self.collective.rank())
		jacobian_process_specific_directory = self.parameters['output_directory']+'jacobian_data/proc_'+str(proc_id)+'/'
		os.makedirs(jacobian_process_specific_directory,exist_ok = True)
		process_specific_directory = self.parameters['output_directory']+'data_on_proc_'+str(proc_id)+'/'
		os.makedirs(process_specific_directory,exist_ok = True)
		if self.u is None:
			self.u = self.observable.generate_vector(hp.STATE)
		if self.m is None:
			self.m = self.observable.generate_vector(hp.PARAMETER)
		if self.control_distribution is not None and self.z is None:
			self.z = self.observable.generate_vector(CONTROL)

		if self.control_distribution is not None:
			assert self.observable.problem.Vh[hp.STATE].mesh().mpi_comm().size == 1, print('Only worked out for serial codes')
			control_dimension = self.z.get_local().shape[0]

		if parameter_jacobian:
			self.J = ObservableJacobian(self.observable)
			output_dimension,parameter_dimension = self.J.shape
			parameter_rank = min(self.parameters['jacobian_rank'],output_dimension,parameter_dimension)

		if parameter_dimension is None:
			parameter_dimension = self.m.size()

		if control_jacobian:
			self.Jz = ObservableControlJacobian(self.observable)
			output_dimension,control_dimension = self.Jz.shape
			control_rank = min(self.parameters['control_jacobian_rank'],output_dimension,control_dimension)


		last_datum_generated = 0
		
		if self.observable.problem.C is None:
			# If this is the case, then the KKT blocks have not been built yet
			# we overcome this by solving somewhere and setting the linearization pt.
			m_mean = self.prior.mean
			u_at_mean = self.observable.problem.generate_state()
			if self.control_distribution is not None:
				if hasattr(self.control_distribution,'mean'):
					z_mean = self.control_distribution.mean
				else:
					# Sample it somewhere
					z_mean = self.observable.generate_vector(CONTROL)
					self.control_distribution.sample(z_mean)
				x_lin = [u_at_mean,m_mean,None,z_mean]
			else:
				x_lin = [u_at_mean,m_mean,None]
			
			self.observable.problem.solveFwd(u_at_mean,x_lin)
			self.observable.setLinearizationPoint(x_lin)
			if self.control_distribution is not None:
				assert self.observable.problem.Cz is not None
		
		# Initialize randomized Omega
		if parameter_jacobian:
			parameter_vector = dl.Vector(self.mesh_constructor_comm)
			self.J.init_vector(parameter_vector,1)
			nvec_Omega_m = min(parameter_rank + self.parameters['oversampling'],output_dimension,parameter_dimension)
			Omega_m = hp.MultiVector(parameter_vector,nvec_Omega_m)
			# Omega does not need to be communicated across processes in this case
			# like with the global reduction collectives
			# Omega can be the same for all samples
			hp.parRandom.normal(1.,Omega_m)
		if control_jacobian:
			control_vector = dl.Vector(self.mesh_constructor_comm)
			self.Jz.init_vector(control_vector,1)
			nvec_Omega_z = min(control_rank + self.parameters['oversampling'],output_dimension,control_dimension)
			Omega_z = hp.MultiVector(control_vector,nvec_Omega_z)
			# Omega does not need to be communicated across processes in this case
			# like with the global reduction collectives
			# Omega can be the same for all samples
			hp.parRandom.normal(1.,Omega_z)

		if check_for_data:
			# Find largest mq pair generated

			# Find largest Jacobian generated
			print('Data check not yet implemented :('.center(80))
			pass
		t0 = time.time()
		for i in range(last_datum_generated,self.parameters['jacobian_data_per_process']):
			print(('Generating data number '+str(i)).center(80))
			hp.parRandom.normal(1,self.noise)
			self.m.zero() # This is probably redundant
			self.prior.sample(self.noise,self.m)
			if self.control_distribution is not None:
				self.control_distribution.sample(self.z)
				x = [self.u,self.m,None,self.z]
			else:
				x = [self.u,self.m,None]
			if hasattr(self.observable.problem,'parameter_projection'):
				# In this case the parameter needs to be projected to subdomain
				m_sample = self.observable.problem.parameter_projection(self.m)
				x[1] = m_sample
			self.observable.solveFwd(self.u,x)
			self.observable.setLinearizationPoint(x)
			if hasattr(self.observable.problem,'parameter_projection'):
				# In this case the parameter needs to be projected to subdomain
				this_m = m_sample.get_local()
			else:
				this_m = self.m.get_local()
			this_q = self.observable.evalu(self.u).get_local()
			np.save(process_specific_directory+'m_sample_'+str(i)+'.npy',this_m)
			np.save(process_specific_directory+'q_sample_'+str(i)+'.npy',this_q)
			if self.control_distribution is not None:
				this_z = self.z.get_local()
				np.save(process_specific_directory+'z_sample_'+str(i)+'.npy',this_z)

			if parameter_jacobian:
				Omega_m.zero() # probably unecessary
				hp.parRandom.normal(1.,Omega_m)
				U, sigma, V = hp.accuracyEnhancedSVD(self.J,Omega_m,parameter_rank,s=1)
				Unp = mv_to_dense(U)
				Vnp = mv_to_dense(V)

				np.save(jacobian_process_specific_directory+'U_sample_'+str(i)+'.npy',Unp)
				np.save(jacobian_process_specific_directory+'sigma_sample_'+str(i)+'.npy',sigma)
				np.save(jacobian_process_specific_directory+'V_sample_'+str(i)+'.npy',Vnp)

			if control_jacobian:
				Omega_z.zero() # probably unecessary
				hp.parRandom.normal(1.,Omega_z)

				Uz, sigmaz, Vz = hp.accuracyEnhancedSVD(self.Jz,Omega_z,control_rank,s=1)
				Uznp = mv_to_dense(Uz)
				Vznp = mv_to_dense(Vz)

				np.save(jacobian_process_specific_directory+'Uz_sample_'+str(i)+'.npy',Uznp)
				np.save(jacobian_process_specific_directory+'sigmaz_sample_'+str(i)+'.npy',sigmaz)
				np.save(jacobian_process_specific_directory+'Vz_sample_'+str(i)+'.npy',Vznp)

			if self.parameters['verbose']:
				if self.control_distribution is None:
					print('One (m,q(m),J(m)) generated every ',(time.time() -t0)/(i - last_datum_generated+1),' s, on average.')
				else:
					pref = 'One (m,z,q(m,z),'
					if parameter_jacobian:
						pref += 'J(m,z)'
					if control_jacobian:
						pref += 'J_z(m,z)'
					print(pref+') generated every ',(time.time() -t0)/(i - last_datum_generated+1),' s, on average.')

		if compress_files:
			print('Compressing mq data'.center(80))
			t_start_mq = time.time()
			local_ms = np.zeros((self.parameters['jacobian_data_per_process'],parameter_dimension))
			local_qs = np.zeros((self.parameters['jacobian_data_per_process'],output_dimension))
			if self.control_distribution is not None:
				local_zs = np.zeros((self.parameters['jacobian_data_per_process'],control_dimension))
			for i in range(0,self.parameters['jacobian_data_per_process']):
				local_ms[i] = np.load(process_specific_directory+'m_sample_'+str(i)+'.npy')
				local_qs[i] = np.load(process_specific_directory+'q_sample_'+str(i)+'.npy')
				if self.control_distribution is not None:
					local_zs[i] = np.load(process_specific_directory+'z_sample_'+str(i)+'.npy')
			if self.control_distribution is None:
				np.savez_compressed(self.parameters['output_directory']+'mq_on_proc'+str(proc_id)+'.npz',m_data = local_ms,q_data = local_qs)
				print(('mq compression took '+str(time.time()-t_start_mq)+' s '))
			else:
				np.savez_compressed(self.parameters['output_directory']+'mzq_on_proc'+str(proc_id)+'.npz',m_data = local_ms,z_data = local_zs,\
																															q_data = local_qs)
				print(('mzq compression took '+str(time.time()-t_start_mq)+' s '))

			if parameter_jacobian:
				t_start_J = time.time()
				print('Compressing Jacobian data'.center(80))
				local_Us = np.zeros((self.parameters['jacobian_data_per_process'],output_dimension,parameter_rank))
				local_sigmas = np.zeros((self.parameters['jacobian_data_per_process'],parameter_rank))
				local_Vs = np.zeros((self.parameters['jacobian_data_per_process'],parameter_dimension,parameter_rank))
				for i in range(0,self.parameters['jacobian_data_per_process']):
					local_Us[i] = np.load(jacobian_process_specific_directory+'U_sample_'+str(i)+'.npy')
					local_sigmas[i] = np.load(jacobian_process_specific_directory+'sigma_sample_'+str(i)+'.npy')
					local_Vs[i] = np.load(jacobian_process_specific_directory+'V_sample_'+str(i)+'.npy')
				np.savez_compressed(self.parameters['output_directory']+'J_on_proc'+str(proc_id)+'.npz',\
							U_data = local_Us,sigma_data = local_sigmas,V_data = local_Vs)
				print(('Jacobian compression took '+str(time.time()-t_start_J)+' s '))

				if True:
					out_name = self.parameters['output_directory']+'jacobian_singular_values_'+str(parameter_rank)+'.pdf'
					plot_singular_values_with_std(np.mean(local_sigmas,axis=0),np.std(local_sigmas,axis=0),outname= out_name)

			if control_jacobian:
				t_start_Jz = time.time()
				print('Compressing Jacobian data'.center(80))
				local_Uzs = np.zeros((self.parameters['jacobian_data_per_process'],output_dimension,control_rank))
				local_sigmazs = np.zeros((self.parameters['jacobian_data_per_process'],control_rank))
				local_Vzs = np.zeros((self.parameters['jacobian_data_per_process'],control_dimension,control_rank))
				for i in range(0,self.parameters['jacobian_data_per_process']):
					local_Uzs[i] = np.load(jacobian_process_specific_directory+'Uz_sample_'+str(i)+'.npy')
					local_sigmazs[i] = np.load(jacobian_process_specific_directory+'sigmaz_sample_'+str(i)+'.npy')
					local_Vzs[i] = np.load(jacobian_process_specific_directory+'Vz_sample_'+str(i)+'.npy')
				np.savez_compressed(self.parameters['output_directory']+'Jz_on_proc'+str(proc_id)+'.npz',\
							Uz_data = local_Uzs,sigmaz_data = local_sigmazs,Vz_data = local_Vzs)
				print(('Control Jacobian compression took '+str(time.time()-t_start_Jz)+' s '))

				if True:
					out_name = self.parameters['output_directory']+'control_jacobian_singular_values_'+str(control_rank)+'.pdf'
					plot_singular_values_with_std(np.mean(local_sigmazs,axis=0),np.std(local_sigmazs,axis=0),outname= out_name)

		self._jacobian_data_generation_time = time.time() - t0


	def _construct_low_rank_Jacobians_batched(self,check_for_data = True,\
						parameter_jacobian = True, control_jacobian = False):
		"""
		This method generates low rank Jacobians for training (and also saves input output data in tandem)
			- :code:`check_for_data` - a boolean to decide whether to check to see if the training
			data already exists in directory specified by :code:`output_directory`.
		Note that this method also saves the input output data and saves them to a directory that 
		is by default a jacobian_data/ directory
		This allows for separate sampling for l2 loss and h1 seminorm loss
		"""
		assert not control_jacobian

		if self.Js is None:
			self.initialize_samples()

		proc_id = int(self.collective.rank())
		output_directory = self.parameters['output_directory']+'jacobian_data/'
		os.makedirs(output_directory,exist_ok = True)

		last_datum_generated = 0
		output_dimension,parameter_dimension = self.Js[0].shape
		if self.control_distribution is not None:
			assert self.observable.problem.Vh[hp.STATE].mesh().mpi_comm().size == 1, print('Only worked out for serial codes')
			control_dimension = self.z.get_local().shape[0]

		rank = min(self.parameters['rank'],output_dimension,parameter_dimension)
		local_Us = np.zeros((0,output_dimension, rank))	
		local_sigmas = np.zeros((0,rank))
		local_Vs = np.zeros((0,parameter_dimension, rank))
		local_ms = np.zeros((0,parameter_dimension))
		local_zs = np.zeros((0,control_dimension))
		local_qs = np.zeros((0,output_dimension))
		# Initialize arrays
		if check_for_data:
			# Save all five or restart sampling and saving
			if os.path.isfile(output_directory+'Us_on_proc_'+str(proc_id)+'.npy') and \
				os.path.isfile(output_directory+'sigmas_on_proc_'+str(proc_id)+'.npy') and \
				os.path.isfile(output_directory+'Vs_on_proc_'+str(proc_id)+'.npy') and \
				os.path.isfile(output_directory+'ms_on_proc_'+str(proc_id)+'.npy') and \
				os.path.isfile(output_directory+'qs_on_proc_'+str(proc_id)+'.npy'):

				local_Us = np.load(output_directory+'Us_on_proc_'+str(proc_id)+'.npy')
				local_sigmas = np.load(output_directory+'sigmas_on_proc_'+str(proc_id)+'.npy')
				local_Vs = np.load(output_directory+'Vs_on_proc_'+str(proc_id)+'.npy')
				local_ms = np.load(output_directory+'ms_on_proc_'+str(proc_id)+'.npy')
				local_qs = np.load(output_directory+'qs_on_proc_'+str(proc_id)+'.npy')

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
			if (self.control_distribution is not None) and os.path.isfile(output_directory+'zs_on_proc_'+str(proc_id)+'.npy'):
				local_zs = np.load(output_directory+'zs_on_proc_'+str(proc_id)+'.npy')
				last_z_generated = local_zs.shape[0]
				if last_z_generated < last_datum_generated:
					last_datum_generated = last_z_generated
					# Update slices
					local_Us = local_Us[:last_datum_generated,:,:]
					local_sigmas = local_sigmas[:last_datum_generated,:]
					local_Vs = local_Vs[:last_datum_generated,:,:]
					local_ms = local_ms[:last_datum_generated,:]
					local_qs = local_qs[:last_datum_generated,:]

		# Generate the input output pairs that correspond to the 
		assert len(self.ms) == self.parameters['samples_per_process']
		assert len(self.zs) == self.parameters['samples_per_process']
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
			np.save(output_directory+'ms_on_proc_'+str(proc_id)+'.npy',np.array(local_ms))
			np.save(output_directory+'qs_on_proc_'+str(proc_id)+'.npy',np.array(local_qs))
			if self.control_distribution is not None:
				local_zs = np.concatentate((local_zs,np.expand_dims(self.zs[i].get_local(),0)))
				np.save(output_directory+'zs_on_proc_'+str(proc_id)+'.npy',np.array(local_zs))

			if self.parameters['verbose']:
				print('On datum saved every ',(time.time() -t0_mq)/(i - last_datum_generated+1),' s, on average.')


		# Initialize randomized Omega
		parameter_vector = dl.Vector(self.mesh_constructor_comm)
		self.Js[0].init_vector(parameter_vector,1)
		Omega = hp.MultiVector(parameter_vector,rank + self.parameters['oversampling'])
		# Omega does not need to be communicated across processes in this case
		# like with the global reduction collectives
		hp.parRandom.normal(1.,Omega)

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

			U, sigma, V = hp.accuracyEnhancedSVD(self.Js[i],Omega,rank,s=1)

			local_Us = np.concatenate((local_Us,np.expand_dims(mv_to_dense(U),0)))
			local_sigmas = np.concatenate((local_sigmas,np.expand_dims(sigma,0)))
			local_Vs = np.concatenate((local_Vs,np.expand_dims(mv_to_dense(V),0)))

			if self.mesh_constructor_comm.rank == 0:
				np.save(output_directory+'Us_on_proc_'+str(proc_id)+'.npy',np.array(local_Us))
				np.save(output_directory+'sigmas_on_proc_'+str(proc_id)+'.npy',np.array(local_sigmas))
				np.save(output_directory+'Vs_on_proc_'+str(proc_id)+'.npy',np.array(local_Vs))
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
			LocalParameters = hp.MultiVector(self.ms[0],self.parameters['error_test_samples'])
			LocalParameters.zero()
			# Generate samples
			for i in range(self.parameters['error_test_samples']):
				t0 = time.time()
				hp.parRandom.normal(1,self.noise)
				self.prior.sample(self.noise,LocalParameters[i])


			LocalErrors = hp.MultiVector(self.ms[0],self.parameters['error_test_samples'])
			projection_vector = self.observable.generate_vector(hp.PARAMETER) 

			for rank_index,rank in enumerate(ranks):
				LocalErrors.zero()
				if rank is None:
					V_GN = self.V_GN
					d_GN = self.d_GN
				else:
					V_GN = hp.MultiVector(self.V_GN[0],rank)
					d_GN = self.d_GN[0:rank]
					for i in range(rank):
						V_GN[i].axpy(1.,self.V_GN[i])
				input_init_vector_lambda = lambda x, dim: self.observable.init_vector(x,dim = 1)
				if self.prior_preconditioned:
					InputProjectorOperator = PriorPreconditionedProjector(V_GN,self.prior.R, input_init_vector_lambda)
				else:
					InputProjectorOperator = hp.LowRankOperator(np.ones_like(d_GN),V_GN, input_init_vector_lambda)
			
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
			# This will be fixed soon. Or someday. Someday soon! :+)
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
				LocalObservables = hp.MultiVector(observable_vector,self.parameters['error_test_samples'])
				LocalObservables.zero()
				LocalParameters = hp.MultiVector(self.ms[0],self.parameters['error_test_samples'])
				LocalParameters.zero()
				# Generate samples
				for i in range(self.parameters['error_test_samples']):
					t0 = time.time()
					hp.parRandom.normal(1,self.noise)
					self.prior.sample(self.noise,LocalParameters[i])
					LocalObservables[i].axpy(1.,self.observable.eval(LocalParameters[i],setLinearizationPoint = True))
					if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):
						print('Generating local observable ',i,' for input error test took',time.time() -t0, 's')

				# Instantiate array for errors
				LocalErrors = hp.MultiVector(observable_vector,self.parameters['error_test_samples'])

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
						V_GN = hp.MultiVector(self.V_GN[0],rank)
						V_GN.zero()
						d_GN = self.d_GN[0:rank]
						for i in range(rank):
							V_GN[i].axpy(1.,self.V_GN[i])
					input_init_vector_lambda = lambda x, dim: self.observable.init_vector(x,dim = 1)
					if self.prior_preconditioned:
						InputProjectorOperator = PriorPreconditionedProjector(V_GN,self.prior.R, input_init_vector_lambda)
					else:
						InputProjectorOperator = hp.LowRankOperator(np.ones_like(d_GN),V_GN, input_init_vector_lambda)
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
							hp.parRandom.normal(1,self.noise)
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
			LocalObservables = hp.MultiVector(observable_vector,self.parameters['error_test_samples'])
			LocalObservables.zero()
			for i in range(LocalObservables.nvec()):
				t0 = time.time()
				hp.parRandom.normal(1,self.noise)
				self.prior.sample(self.noise,self.ms[0])
				if self.control_distribution is not None:
					assert self.zs is not None
					if type(self.zs) is list:
						assert self.zs[0] is not None
						self.control_distribution.sample(self.zs[0])
						x = [self.us[0],self.ms[0],None,self.zs[0]]
					else:
						assert self.z is not None
						self.control_distribution.sample(self.z)
						x = [self.us[0],self.ms[0],None,self.z]
				else:
					x = [self.us[0],self.ms[0],None]
				self.observable.setLinearizationPoint(x)
				LocalObservables[i].axpy(1.,self.observable.eval(self.ms[0]))
				if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):
					print('Generating local observable ',i, ' for output error test')

			LocalErrors = hp.MultiVector(observable_vector,self.parameters['error_test_samples'])
			projection_vector = dl.Vector(self.mesh_constructor_comm)
			self.observable.init_vector(projection_vector,dim = 0)

			for rank_index,rank in enumerate(ranks):
				LocalErrors.zero()
				if rank is None:
					U_NG = self.U_NG
					d_NG = self.d_NG
				else:
					U_NG = hp.MultiVector(self.U_NG[0],rank)
					d_NG = self.d_NG[0:rank]
					for i in range(rank):
						U_NG[i].axpy(1.,self.U_NG[i])
				output_init_vector_lambda = lambda x, dim: self.observable.init_vector(x,dim = 0)
				OutputProjectorOperator = hp.LowRankOperator(np.ones_like(d_NG),U_NG,output_init_vector_lambda)
			
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
			
