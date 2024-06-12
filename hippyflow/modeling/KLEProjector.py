# Copyright (c) 2020-2023, The University of Texas at Austin 
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
from mpi4py import MPI 
import time

import hippylib as hp

from ..collectives.collective import NullCollective
from ..collectives.collectiveOperator import CollectiveOperator
from ..collectives.comm_utils import checkMeshConsistentPartitioning
from .jacobian import *
from ..utilities.mv_utilities import mv_to_dense
from ..utilities.plotting import *
from .priorPreconditionedProjector import PriorPreconditionedProjector

def KLEParameterList():
	"""
	This function implements the parameter list for the KLE object
	"""
	parameters = {}
	parameters['error_test_samples'] 		= [50, 'Number of samples for error test']
	parameters['rank'] 				 		= [128, 'Rank of subspace']
	parameters['oversampling'] 		 		= [10, 'Oversampling parameter for randomized algorithms']
	parameters['verbose']					= [True, 'Boolean for printing']
	parameters['output_directory']			= [None,'output directory for saving arrays and plots']
	parameters['plot_label_suffix']			= ['', 'suffix for plot label']
	parameters['save_and_plot']				= [True, 'save and plot or not']

	parameters['input_decoder_name']			= ['KLE_decoder', 'string for naming']

	return hp.ParameterList(parameters)

class MassPreconditionedCovarianceOperator:
	def __init__(self, C, M):
		"""
		Linear operator representing the mass matrix preconditioned
		covariance matrix :math:`M C M`
		"""
		self.C = C 
		self.M = M 
		self.mpi_comm = self.M.mpi_comm()


		self.Mx = dl.Vector(self.mpi_comm)
		self.CMx = dl.Vector(self.mpi_comm)
		self.M.init_vector(self.Mx, 0)
		self.M.init_vector(self.CMx, 0)

	def init_vector(self,x,dim):
		self.M.init_vector(x,dim)

	def mult(self, x, y):
		self.M.mult(x, self.Mx)
		self.C.mult(self.Mx, self.CMx)
		self.M.mult(self.CMx, y)




class KLEProjector:
	"""
	This class implements an input subspace projector based solely on the prior
	"""
	def __init__(self, prior, mesh_constructor_comm = None ,collective = None, parameters = KLEParameterList()):
		"""
		Constructor
			- :code:`observable` - object that implements the observable mapping :math:`m -> q(m)`
			- :code:`prior` - object that implements the prior
			- :code:`mesh_constructor_comm` - MPI communicator that is used in mesh construction
			- :code:`collective` - MPI collective used in parallel collective operations
			- :code:`parameters` - parameter dictionary
		"""
		self.prior = prior
		if mesh_constructor_comm is not None:
			self.mesh_constructor_comm = mesh_constructor_comm
		else:
			self.mesh_constructor_comm = self.prior.R.mpi_comm()

		if collective is not None:
			self.collective = collective
		else:
			self.collective = NullCollective()


		self.parameters = parameters

		self.noise = None

		self.C = hp.Solver2Operator(self.prior.Rsolver, mpi_comm=self.mesh_constructor_comm)


		consistent_partitioning = checkMeshConsistentPartitioning(\
							self.prior.Vh.mesh(), self.collective)
		print('Consistent partitioning:', consistent_partitioning)

		self.d_KLE = None
		self.V_KLE = None
		self.M_orthogonal = None

	def random_input_projector(self):
		"""
		This method computes and returns a random projection basis
		"""
		m_KLE = dl.Vector(self.mesh_constructor_comm)
		self.prior.M.init_vector(m_KLE,0)
		Omega = hp.MultiVector(m_KLE,self.parameters['rank'] + self.parameters['oversampling'])

		if self.collective.rank() == 0:
			hp.parRandom.normal(1.,Omega)
			Omega.orthogonalize()
		else:
			Omega.zero()
		self.collective.bcast(Omega,root = 0)
		return Omega

		self.kle_decoder = None
		self.kle_encoder = None
	



	def construct_input_subspace(self,orthogonality = 'mass'):
		"""
		This method computes the KLE subspace
			- :code:`M_orthogonal` - Boolean about whether the vectors are made to be mass matrix orthogonal
		"""
		t0 = time.time()
		assert hasattr(self.prior,'Rsolver') and hasattr(self.prior,'M') and hasattr(self.prior,'Msolver')



		KLE_Operator = MassPreconditionedCovarianceOperator(self.C,self.prior.M)

		# Totally unnecessary averaging that I am doing in order to keep the code consistent
		Average_KLE_Operator = CollectiveOperator(KLE_Operator, self.collective, mpi_op = 'avg')

		m_KLE = dl.Vector(self.mesh_constructor_comm)
		KLE_Operator.init_vector(m_KLE,0)
		Omega = hp.MultiVector(m_KLE,self.parameters['rank'] + self.parameters['oversampling'])

		if self.collective.rank() == 0:
			hp.parRandom.normal(1.,Omega)
		else:
			Omega.zero()

		self.collective.bcast(Omega,root = 0)

		if orthogonality.lower() == 'mass':
			self.d_KLE, self.V_KLE = hp.doublePassG(KLE_Operator,\
				self.prior.M, self.prior.Msolver, Omega,self.parameters['rank'],s=1)
			self.M_orthogonal = True
			kle_decoder = self.V_KLE
			kle_encoder = hp.MultiVector(kle_decoder)
			hp.MatMvMult(self.prior.M,kle_decoder,kle_encoder)

		elif orthogonality.lower() == 'prior':
			prior_orth_KLE_constructor = KLESubspaceConstructorSLEPc(self.prior)
			self.d_KLE, kle_decoder, kle_encoder = prior_orth_KLE_constructor.compute_kle_subspace(self.parameters['rank'])
			self.V_KLE = kle_decoder

		elif orthogonality.lower() == 'identity':
			RsolverOperator = hp.Solver2Operator(self.prior.Rsolver)
			self.d_KLE, self.V_KLE = hp.doublePass(RsolverOperator, Omega,self.parameters['rank'],s=1)
			self.M_orthogonal = False
			kle_decoder = self.V_KLE
			kle_encoder = hp.MultiVector(kle_decoder) #copy constructor

		else: 
			raise

		self._subspace_construction_time = time.time() - t0
		if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):	
			print('Construction of input subspace took ',self._subspace_construction_time,'s')
			# print('Input subspace eigenvalues = ',self.d_GN)

		if True and MPI.COMM_WORLD.rank == 0 and self.parameters['save_and_plot']:
			np.save(self.parameters['output_directory']+self.parameters['input_decoder_name'],mv_to_dense(self.V_KLE))
			np.save(self.parameters['output_directory']+'KLE_d',self.d_KLE)

			out_name = self.parameters['output_directory']+'KLE_eigenvalues_'+str(self.parameters['rank'])+'.pdf'
			_ = spectrum_plot(self.d_KLE,\
				axis_label = ['i',r'$\lambda_i$',\
				r'Eigenvalues of $C$'+self.parameters['plot_label_suffix']], out_name = out_name)

		return self.d_KLE, kle_decoder, kle_encoder


	def test_errors(self, ranks = [None],cut_off = 1e-12):
		"""
		This method implements projection error tests for the KLE basis
			-:code:`ranks` - a python list of ints specifying the ranks for the projection error tests
			-:code:`cut_off` - where to truncate the ranks based on the spectral decay of KLE
		"""
		if self.noise is None:
			self.noise = dl.Vector(self.mesh_constructor_comm)
			self.prior.init_vector(self.noise,"noise")

		# ranks assumed to be python list with sort in place member function
		ranks.sort()

		# Simple projection test
		if self.d_KLE is None:
			if self.mesh_constructor_comm.rank == 0:
				print('Constructing input subspace')
			self.construct_input_subspace()
		elif len(self.d_KLE)<ranks[-1]:
			if self.mesh_constructor_comm.rank == 0:
				print('Constructing input subspace because larger rank needed.')
				self.parameters['rank'] = ranks[-1]
			self.construct_input_subspace()
		else: 
			if self.mesh_constructor_comm.rank == 0:
				print('Input subspace already computed proceeding with error tests')
		# truncate eigenvalues for numerical stability
		numericalrank = np.where(self.d_KLE > cut_off)[-1][-1] + 1 # due to 0 indexing
		ranks = ranks[:np.where(ranks <= numericalrank)[0][-1]+1]# due to inclusion
		global_avg_rel_errors = np.ones_like(ranks,dtype = np.float64)
		global_std_rel_errors = np.zeros_like(ranks,dtype = np.float64)

		# Naive test on input space
		projection_vector = dl.Vector(self.mesh_constructor_comm)
		self.prior.init_vector(projection_vector,0)

		LocalParameters = hp.MultiVector(projection_vector,self.parameters['error_test_samples'])
		LocalParameters.zero()
		# Generate samples
		for i in range(self.parameters['error_test_samples']):
			t0 = time.time()
			hp.parRandom.normal(1,self.noise)
			self.prior.sample(self.noise,LocalParameters[i])

		LocalErrors = hp.MultiVector(projection_vector,self.parameters['error_test_samples'])
		

		for rank_index,rank in enumerate(ranks):
			LocalErrors.zero()
			if rank is None:
				V_KLE = self.V_KLE
				d_KLE = self.d_KLE
			else:
				V_KLE = MultiVector(self.V_KLE[0],rank)
				d_KLE = self.d_KLE[0:rank]
				for i in range(rank):
					V_KLE[i].axpy(1.,self.V_KLE[i])
			input_init_vector_lambda = lambda x, dim: self.prior.init_vector(x,dim = 1)
			if self.M_orthogonal:
				InputProjectorOperator = PriorPreconditionedProjector(V_KLE,self.prior.M, input_init_vector_lambda)
			else:
				InputProjectorOperator = hp.LowRankOperator(np.ones_like(d_KLE),V_KLE, input_init_vector_lambda)
		
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
			global_avg_rel_errors[rank_index] = self.collective.allReduce(avg_rel_error,'avg')
			global_std_rel_errors[rank_index] = np.sqrt(self.collective.allReduce(std_rel_error_squared,'avg'))
			if self.mesh_constructor_comm.rank == 0:
				print('Naive global average relative error input = ',global_avg_rel_errors[rank_index],' for rank ',rank)

		return global_avg_rel_errors, global_std_rel_errors


class KLESubspaceConstructorSLEPc:
	"""
	Class for a matern Gaussian constructed from an elliptic operator
	"""
	def __init__(self, hp_prior):
		assert hasattr(hp_prior, "A")
		assert hasattr(hp_prior, "M")
		assert hasattr(hp_prior, "R")
		self._hp_prior = hp_prior
		self.R = self._hp_prior.R
		self.Vh = self._hp_prior.Vh
		self.A = dl.as_backend_type(self._hp_prior.A)
		self.M = dl.as_backend_type(self._hp_prior.M)
		self.eigensolver = dl.SLEPcEigenSolver(self.A, self.M)
		self.eigensolver.parameters["solver"] = "krylov-schur"
		self.eigensolver.parameters["problem_type"] = "gen_hermitian"
		self.eigensolver.parameters["spectrum"] = "target magnitude"
		self.eigensolver.parameters["spectral_transform"] = "shift-and-invert"
		self.eigensolver.parameters["spectral_shift"] = 0.0

		self.m = dl.Function(self.Vh).vector()

	def mpi_comm(self):
		return self.R.mpi_comm()

	def compute_kle_subspace(self, rank):
		"""
		Compute the KLE basis using :code:`dl.SLEPcEigenSolver`
		:param rank: number of eigenpairs
		"""
		print("Solving eigenvalue problem")
		self.eigensolver.solve(rank)
		sqrt_precision_eigenvalues = np.zeros(rank)

		print("Initializing multivectors")
		kle_decoder = hp.MultiVector(self.m, rank)
		kle_encoder = hp.MultiVector(self.m, rank)

		kle_decoder.zero()
		kle_encoder.zero()


		for i in range(rank):
			sqrt_precision_eigenvalues[i], _, basis_i, _ = self.eigensolver.get_eigenpair(i)
			kle_decoder[i].axpy(1.0/sqrt_precision_eigenvalues[i], basis_i)

		covariance_eigenvalues = 1/sqrt_precision_eigenvalues**2

		hp.MatMvMult(self.R, kle_decoder, kle_encoder)
		return covariance_eigenvalues, kle_decoder, kle_encoder



