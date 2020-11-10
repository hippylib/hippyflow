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
import numpy as np
import os
from hippylib import *
from mpi4py import MPI 
import time

from ..collectives.collectiveOperator import CollectiveOperator
from ..collectives.comm_utils import checkMeshConsistentPartitioning
from .jacobian import *
from ..utilities.mv_utilities import mv_to_dense, dense_to_mv
from ..utilities.plotting import *
from .priorPreconditionedProjector import PriorPreconditionedProjector

def KLEParameterList():
	"""

	"""
	parameters = {}
	parameters['sample_per_process'] 	= [100, 'Number of samples per process']
	parameters['error_test_samples'] 		= [50, 'Number of samples for error test']
	parameters['rank'] 				 	= [128, 'Rank of subspace']
	parameters['oversampling'] 		 	= [10, 'Oversampling parameter for randomized algorithms']
	parameters['verbose']				= [True, 'Boolean for printing']

	parameters['output_directory']			= [None,'output directory for saving arrays and plots']
	parameters['plot_label_suffix']			= ['', 'suffix for plot label']

	return ParameterList(parameters)

class MRinvM:
	"""
	MRinvM implements the action of :math: `MR^{-1}M` for a BiLaplacianPrior
	"""

	def __init__(self,Rsolver,M):
		self.Rsolver = Rsolver
		self.M = M
		self.help = dl.Vector(self.M.mpi_comm())
		self.M.init_vector(self.help,1)

	def init_vector(self,x,dim):
		self.M.init_vector(x,dim)

	def mult(self,x,y):
		self.Rsolver.solve(self.help,self.M*x)
		self.M.mult(self.help,y)




class KLEProjector:
	"""
	This class implements an input subspace projector based solely on the prior
	
	"""
	def __init__(self, prior, mesh_constructor_comm = None ,collective = None, parameters = KLEParameterList()):

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


		consistent_partitioning = checkMeshConsistentPartitioning(\
							self.prior.Vh.mesh(), self.collective)
		print('Consistent partitioning:', consistent_partitioning)

		self.d_KLE = None
		self.V_KLE = None
		self.prior_preconditioned = None

	def random_input_projector(self):
		m_KLE = dl.Vector(self.mesh_constructor_comm)
		self.prior.M.init_vector(m_KLE,0)
		Omega = MultiVector(m_KLE,self.parameters['rank'] + self.parameters['oversampling'])

		if self.collective.rank() == 0:
			parRandom.normal(1.,Omega)
			Omega.orthogonalize()
		else:
			Omega.zero()
		self.collective.bcast(Omega,root = 0)
		return Omega



	def construct_input_subspace(self,prior_preconditioned = True):
		'''
		
		'''
		t0 = time.time()
		assert hasattr(self.prior,'Rsolver') and hasattr(self.prior,'M') and hasattr(self.prior,'Msolver')

		KLE_Operator = MRinvM(self.prior.Rsolver,self.prior.M)

		# Totally unnecessary averaging that I am doing in order to keep the code consistent
		Average_KLE_Operator = CollectiveOperator(KLE_Operator, self.collective, mpi_op = 'avg')

		m_KLE = dl.Vector(self.mesh_constructor_comm)
		KLE_Operator.init_vector(m_KLE,0)
		Omega = MultiVector(m_KLE,self.parameters['rank'] + self.parameters['oversampling'])

		if self.collective.rank() == 0:
			parRandom.normal(1.,Omega)
		else:
			Omega.zero()

		self.collective.bcast(Omega,root = 0)

		if prior_preconditioned:
			self.d_KLE, self.V_KLE = doublePassG(KLE_Operator,\
				self.prior.M, self.prior.Msolver, Omega,self.parameters['rank'],s=1)
			self.prior_preconditioned = True
		else:
			RsolverOperator = Solver2Operator(self.prior.Rsolver)
			self.d_KLE, self.V_KLE = doublePass(RsolverOperator, Omega,self.parameters['rank'],s=1)
			self.prior_preconditioned = False

		if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):	
			print('Construction of input subspace took ',time.time() - t0,'s')
			# print('Input subspace eigenvalues = ',self.d_GN)

		if True and MPI.COMM_WORLD.rank == 0:
			np.save(self.parameters['output_directory']+'KLE_projector',mv_to_dense(self.V_KLE))
			np.save(self.parameters['output_directory']+'KLE_d',self.d_KLE)

			out_name = self.parameters['output_directory']+'KLE_eigenvalues_'+str(self.parameters['rank'])+'.pdf'
			_ = spectrum_plot(self.d_KLE,\
				axis_label = ['i',r'$\lambda_i$',\
				r'Eigenvalues of $C$'+self.parameters['plot_label_suffix']], out_name = out_name)


	def test_errors(self, ranks = [None],cut_off = 1e-12):
		if self.noise is None:
			self.noise = dl.Vector(self.mesh_constructor_comm)
			self.prior.init_vector(self.noise,"noise")

		global_avg_rel_errors_input =  None
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
		global_avg_rel_errors_input = np.ones_like(ranks,dtype = np.float64)

		# Naive test on output space
		
		projection_vector = dl.Vector()
		self.prior.init_vector(projection_vector,0)

		LocalParameters = MultiVector(projection_vector,self.parameters['error_test_samples'])
		LocalParameters.zero()
		# Generate samples
		for i in range(self.parameters['error_test_samples']):
			t0 = time.time()
			parRandom.normal(1,self.noise)
			self.prior.sample(self.noise,LocalParameters[i])

		LocalErrors = MultiVector(projection_vector,self.parameters['error_test_samples'])
		

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
			if self.prior_preconditioned:
				InputProjectorOperator = PriorPreconditionedProjector(V_KLE,self.prior.M, input_init_vector_lambda)
			else:
				InputProjectorOperator = LowRankOperator(np.ones_like(d_KLE),V_KLE, input_init_vector_lambda)
		
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
			global_avg_rel_errors_input[rank_index] = self.collective.allReduce(avg_rel_error,'avg')
			if self.mesh_constructor_comm.rank == 0:
				print('Naive global average relative error input = ',global_avg_rel_errors_input[rank_index],' for rank ',rank)

		return global_avg_rel_errors_input
