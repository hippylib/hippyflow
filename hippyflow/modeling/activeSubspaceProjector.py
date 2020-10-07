import dolfin as dl
import numpy as np
import os
from hippylib import *
from mpi4py import MPI 
import time

from ..collectives.collectiveOperator import CollectiveOperator
from .jacobian import *
from ..utilities.mv_utilities import mv_to_dense, dense_to_mv
from ..utilities.plotting import *

def ActiveSubspaceParameterList():
	"""

	"""
	parameters = {}
	parameters['sample_per_process'] 	= [100, 'Number of samples per process']
	parameters['rank'] 				 	= [20, 'Rank of subspace']
	parameters['oversampling'] 		 	= [10, 'Oversampling parameter for randomized algorithms']
	parameters['double_loop_samples']	= [10, 'Number of samples used in double loop MC approximation']
	parameters['verbose']				= [True, 'Boolean for printing']

	return ParameterList(parameters)

class ActiveSubspaceProjector:
	"""
	This class implements projectors based on globally averages GN Hessian and inside out GN Hessian
	We have a forward mapping: m --> q(m)
	And a forward map Jacobian: \nabla q(m)
	Jacobian SVD: J = U S V'
	Output active subspace: J'J = US^2U'
	Input active subspace: JJ' = VS^2V'
	"""
	def __init__(self,observable, prior, mesh_constructor_comm = None ,collective = None, parameters = ActiveSubspaceParameterList()):

		self.observable = observable
		self.prior = prior
		if mesh_constructor_comm is not None:
			self.mesh_constructor_comm = mesh_constructor_comm
		else:
			self.mesh_constructor_comm = dl.MPI.COMM_WORLD

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

		parRandom.normal(1,self.noise)

		self.u = self.observable.generate_vector(STATE)
		self.m = self.observable.generate_vector(PARAMETER)
		# Draw a new sample and set linearization point.
		self.new_sample()

		self.J = ObservableJacobian(self.observable)

		self.d_GN = None
		self.V_GN = None

		self.d_NG = None
		self.U_NG = None

	def new_sample(self):
		# set linearization point
		self.prior.sample(self.noise,self.m)
		x = [self.u,self.m,None]
		self.observable.solveFwd(self.u,x)
		self.observable.setLinearizationPoint(x)


	def construct_input_subspace(self):
		t0 = time.time()
		GN_Hessian = JTJ(self.J)
		Average_GN_Hessian = CollectiveOperator(GN_Hessian, self.collective, mpi_op = 'avg')

		x_GN = dl.Vector(self.mesh_constructor_comm)
		GN_Hessian.init_vector(x_GN)
		Omega = MultiVector(x_GN,self.parameters['rank'] + self.parameters['oversampling'])

		if self.collective.rank() == 0:
			parRandom.normal(1.,Omega)
		else:
			Omega.zero()

		self.collective.bcast(Omega,root = 0)

		self.d_GN, self.V_GN = doublePass(Average_GN_Hessian,Omega,self.parameters['rank'],s=1)
		if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):	
			print('Construction of input subspace took ',time.time() - t0,'s')
			print('Input subspace eigenvalues = ',self.d_GN)


	def construct_output_subspace(self):
		t0 = time.time()
		NG_Hessian = JJT(self.J)
		Average_NG_Hessian = CollectiveOperator(NG_Hessian, self.collective, mpi_op = 'avg')

		x_NG = dl.Vector(self.mesh_constructor_comm)
		NG_Hessian.init_vector(x_NG)
		Omega = MultiVector(x_NG,self.parameters['rank'] + self.parameters['oversampling'])

		if self.collective.rank() == 0:
			parRandom.normal(1.,Omega)
		else:
			Omega.zero()

		self.collective.bcast(Omega,root = 0)
		self.d_NG, self.U_NG = doublePass(Average_NG_Hessian,Omega,self.parameters['rank'],s=1)
		if self.parameters['verbose'] and (self.mesh_constructor_comm.rank ==0):	
			print('Construction of output subspace took ',time.time() - t0,'s')
			print('Output subspace eigenvalues = ',self.d_NG)

	def test_error_bounds(self,test_input = True, test_output = True, ranks = [None],cut_off = 1e-8):
		global_avg_rel_errors_input, global_avg_rel_errors_output = None, None
		# ranks assumed to be python list with sort in place member function
		ranks.sort()
		if test_input:
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

			ranks = ranks[:np.where(ranks <= numericalrank)[0][-1]+1]
			global_avg_rel_errors_input = np.ones_like(ranks,dtype = np.float64)

			# Double loop MC error approximation
			observable_vector = dl.Vector(self.mesh_constructor_comm)
			self.observable.init_vector(observable_vector,dim = 0)
			LocalObservables = MultiVector(observable_vector,self.parameters['sample_per_process'])
			LocalObservables.zero()
			LocalParameters = MultiVector(self.m,self.parameters['sample_per_process'])
			LocalParameters.zero()

			for i in range(LocalObservables.nvec()):
				t0 = time.time()
				parRandom.normal(1,self.noise)
				self.prior.sample(self.noise,LocalParameters[i])
				x = [self.u,LocalParameters[i],None]
				self.observable.setLinearizationPoint(x)
				LocalObservables[i].axpy(1.,self.observable.eval(LocalParameters[i]))
				if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):
					print('Generating local observable ',i,' for input error test took',time.time() -t0, 's')

			LocalErrors = MultiVector(observable_vector,self.parameters['sample_per_process'])
			Local_Reduced_Observable = MultiVector(observable_vector,self.parameters['sample_per_process'])

			x_r = dl.Vector(self.mesh_constructor_comm)
			self.observable.init_vector(x_r,dim = 1)
			y = dl.Vector(self.mesh_constructor_comm)
			self.observable.init_vector(y,dim = 1)
			y_r = dl.Vector(self.mesh_constructor_comm)
			self.observable.init_vector(y_r,dim = 1)

			for rank_index,rank in enumerate(ranks):
				if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):
					print('Beginning double loop MC for rank', rank)
				LocalErrors.zero()
				Local_Reduced_Observable.zero()
				if rank is None:
					V_GN = self.V_GN
					d_GN = self.d_GN
				else:
					V_GN = MultiVector(self.V_GN[0],rank)
					V_GN.zero()
					d_GN = self.d_GN[0:rank]
					for i in range(rank):
						V_GN[i].axpy(1.,self.V_GN[i])
						# V_GN[i].set_local(self.V_GN[i].get_local())
						# V_GN[i].apply('')
				input_init_vector_lambda = lambda x, dim: self.observable.init_vector(x,dim = 1)
				InputProjectorOperator = LowRankOperator(np.ones_like(d_GN),V_GN, input_init_vector_lambda)
				print('Constructed projection operator for rank ',rank)
				rel_errors = np.zeros(LocalErrors.nvec())

				for i in range(LocalErrors.nvec()):
					if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):
						print('Beginning outer loop i = ',i)
					LocalErrors[i].axpy(1.,LocalObservables[i])
					denominator = LocalErrors[i].norm('l2')
					x_r.zero()
					InputProjectorOperator.mult(LocalParameters[i],x_r)
					for j in range(self.parameters['double_loop_samples']):
						t0 = time.time()
						y.zero()
						y_r.zero()
						parRandom.normal(1,self.noise)
						self.prior.sample(self.noise,y)
						InputProjectorOperator.mult(y,y_r)
						y.axpy(-1., y_r)
						x = [self.u, x_r + y, None]
						self.observable.setLinearizationPoint(x)
						LocalErrors[i].axpy(-1./float(self.parameters['double_loop_samples']),self.observable.eval(x_r + y))
						if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):
							print('Inner loop i,j = ',(i,j), ' took ',time.time() - t0, 's')
					numerator = LocalErrors[i].norm('l2')
					rel_errors[i] = numerator/denominator

				avg_rel_error = np.mean(rel_errors)
				print('avg_rel_error = ',avg_rel_error)

				global_avg_rel_errors_input[rank_index] = self.collective.allReduce(avg_rel_error,'avg')
				if self.mesh_constructor_comm.rank == 0:
					print('Global average relative error input = ',global_avg_rel_errors_input[rank_index],' for rank ',rank)


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
			# Naive test on output space
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
				if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):
					print('Generating local observable ',i, ' for output error test')

			LocalErrors = MultiVector(observable_vector,self.parameters['sample_per_process'])
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
						# U_NG[i].set_local(self.U_NG[i].get_local())
						# U_NG[i].apply('')
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
				global_avg_rel_errors_output[rank_index] = self.collective.allReduce(avg_rel_error,'avg')
				if self.mesh_constructor_comm.rank == 0:
					print('Global average relative error output = ',global_avg_rel_errors_output[rank_index],' for rank ',rank)

		if False and (MPI.COMM_WORLD.size == 1) and (self.d_GN is not None) and (self.d_NG is not None):
			spectral_error = self.d_GN - self.d_NG

			print("[world rank {:d}] ".format(MPI.COMM_WORLD.rank)+"[mesh rank {:d}] ".format(self.mesh_constructor_comm.rank)+\
			"[sample rank {:d}] ".format(self.collective.rank())+ '|| d_GN_avg - d_NG_avg ||  = ' + str(np.linalg.norm(spectral_error)))

		return [global_avg_rel_errors_input,global_avg_rel_errors_output]
			

	# def save_asnp(filename = 'AS_basis'):
	# 	if int(dl.MPI.COMM_WORLD.rank) == 0:
	# 		print('Just on one process, we are saving')
	# 		np.save(mv_to_dense(self.U_MV),filename)
	# 		print('Save was successful')
