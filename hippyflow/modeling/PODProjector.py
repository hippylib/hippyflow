import dolfin as dl
import numpy as np
import time
from hippylib import *
import os

from ..collectives.collectiveOperator import CollectiveOperator
from ..collectives.comm_utils import checkMeshConsistentPartitioning
from ..utilities.mv_utilities import mv_to_dense, dense_to_mv

def PODParameterList():
	"""

	"""
	parameters = {}
	parameters['sample_per_process'] = [100, 'Number of samples per process']
	parameters['rank'] 				 = [20, 'Rank of POD subspace']
	parameters['oversampling'] 		 = [10, 'Oversampling parameter for randomized algorithms']
	parameters['data_per_process']	 = [250,'Total number of testing and training data to be constructed']
	parameters['verbose']			 = [False,'Boolean for prints']

	return ParameterList(parameters)


class PODProjector:
	"""
	Projector class based on proper orthogonal decomposition

	"""
	def __init__(self,observable, prior, mesh_constructor_comm = None ,collective = None, parameters = PODParameterList()):

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

		self.u = self.observable.generate_vector(STATE)
		self.m = self.observable.generate_vector(PARAMETER)


		self.d = None
		self.U_MV = None

		self.u_at_mean = None

	def solve_at_mean(self):
		m_mean = self.prior.mean
		self.u_at_mean = self.observable.problem.generate_state()
		self.observable.problem.solveFwd(self.u_at_mean,[self.u_at_mean,m_mean,None])


	def generate_training_data(self,output_directory = 'data/',check_for_data = True):
		self.solve_at_mean()
		my_rank = int(self.collective.rank())
		try:
			os.mkdir(output_directory)
		except:
			pass
		observable_vector = dl.Vector(self.mesh_constructor_comm)
		self.observable.init_vector(observable_vector,dim = 0)
		last_datum_generated = -1
		m_shape = self.m.get_local().shape[0]
		q_shape = observable_vector.get_local().shape[0]
		print('m_shape = ',m_shape)
		print('q_shape = ',q_shape)
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
		for i in range(last_datum_generated+1,self.parameters['data_per_process']):
			print('Generating data number '+str(i))
			parRandom.normal(1,self.noise)
			self.prior.sample(self.noise,self.m)
			local_ms = np.concatenate((local_ms,np.expand_dims(self.m.get_local(),0)))
			self.u.zero()
			self.u.axpy(1.,self.u_at_mean)
			x = [self.u,self.m,None]
			self.observable.setLinearizationPoint(x)
			local_qs = np.concatenate((local_qs,np.expand_dims(self.observable.eval(self.m).get_local(),0)))
			np.save(output_directory+'ms_on_rank_'+str(my_rank)+'.npy',np.array(local_ms))
			np.save(output_directory+'qs_on_rank_'+str(my_rank)+'.npy',np.array(local_qs))
			print('On datum generated every ',(time.time() -t0)/(i - last_datum_generated),' s, on average.')


	def construct_subspace(self):
		t0 = 0
		self.solve_at_mean()
		observable_vector = dl.Vector(self.mesh_constructor_comm)
		self.observable.init_vector(observable_vector,dim = 0)
		LocalObservables = MultiVector(observable_vector,self.parameters['sample_per_process'])
		LocalObservables.zero()
		#Read data from file and build subspace option
		for i in range(LocalObservables.nvec()):
			print('Starting observable generation for draw ',i)
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
		if self.parameters['verbose'] and (self.mesh_constructor_comm.rank ==0):
			print('Construction of POD subspace took ', time.time() - t0,'s')

		# print(self.d)


	def test_error_bounds(self,ranks = [None],cut_off = 1e-10):
		# ranks assumed to be python list with sort in place member function
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
			if self.parameters['verbose'] and (self.mesh_constructor_comm.rank == 0):
				print('Generating local observable ',i,' for POD error test took',time.time() -t0, 's')

		LocalErrors = MultiVector(observable_vector,self.parameters['sample_per_process'])

		projection_vector = dl.Vector(self.mesh_constructor_comm)
		self.observable.init_vector(projection_vector,dim = 0)
		global_avg_rel_errors = []
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
			global_avg_rel_errors.append(self.collective.allReduce(avg_rel_error,'avg'))
			if self.mesh_constructor_comm.rank == 0:
				print('Global average relative error = ',global_avg_rel_errors[-1])

		return global_avg_rel_errors

	# def save_asnp(filename = 'POD_basis'):
	# 	if int(dl.MPI.COMM_WORLD.rank) == 0:
	# 		print('Just on one process, we are saving')
	# 		np.save(mv_to_dense(self.U_MV),filename)
	# 		print('Save was successful')














