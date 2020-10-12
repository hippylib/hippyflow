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

def KLEParameterList():
	"""

	"""
	parameters = {}
	parameters['sample_per_process'] 	= [100, 'Number of samples per process']
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

		consistent_partitioning = checkMeshConsistentPartitioning(\
							self.prior.Vh.mesh(), self.collective)
		print('Consistent partitioning:', consistent_partitioning)

		self.d_KLE = None
		self.V_KLE = None



	def construct_input_subspace(self):
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

		if True:
			self.d_KLE, self.V_KLE = doublePassG(KLE_Operator,\
		 	self.prior.M, self.prior.Msolver, Omega,self.parameters['rank'],s=1)
		if False:
			RsolverOperator = Solver2Operator(self.prior.Rsolver)
			self.d_KLE, self.V_KLE = doublePass(RsolverOperator, Omega,self.parameters['rank'],s=1)

		print(self.d_KLE)

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





