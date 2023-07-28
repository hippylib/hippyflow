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

import os
import dolfin as dl
import numpy as np

import hippylib as hp
import hippyflow as hf

import time


def data_generator_settings(settings = {}):
	"""
	"""
	settings['rM'] = None
	settings['rZ'] = None
	settings['oversample'] = 10

	settings['verbose'] = True

	return settings

class DataGenerator:

	def __init__(self,observable, prior, control_distribution = None,\
					settings = data_generator_settings(), parRandom = None,\
					mesh_constructor_comm = None):
		"""
		"""
		self.observable = observable
		self.prior = prior
		self.control_distribution = control_distribution

		# Constructor for mesh-partitioned vectors
		if mesh_constructor_comm is not None:
			self.mesh_constructor_comm = mesh_constructor_comm
		else:
			self.mesh_constructor_comm = self.observable.mpi_comm()

		# Random number generator
		if parRandom is None:
			self.parRandom = hp.parRandom
		else:
			self.parRandom = parRandom

		# Array allocation for sampling
		self.noise = dl.Vector(self.mesh_constructor_comm)
		self.prior.init_vector(self.noise,"noise")

		# Array allocation for the PDE solution
		self.u = None
		self.m = None
		self.z = None

		# Array allocation for the derivatives
		self.J = None
		self.Jz = None

		# Dimensions
		self.dQ = None # Output (QoI) dimension
		self.dM = None # Input (model parameter) dimension
		self.dZ = None # (Optional) input (control variable) dimension

		self.rQ = None # Output (QoI) rank
		self.rM = None # Input (model parameter) rank
		self.rZ = None # (Optional) input (control variable) rank

		# Settings 
		self.settings = settings

		


	def generate(self, n_samples, derivatives = (0,0),\
					output_basis = None,data_dir = 'data/test/', compress = True):
		"""
		"""
		os.makedirs(data_dir+'/mq_data/',exist_ok=True)
		if derivatives[0]:
			os.makedirs(data_dir+'/J_data/',exist_ok=True)
		if derivatives[1]:
			assert self.control_distribution is not None
			assert hasattr(self.observable.problem,'Cz')
			os.makedirs(data_dir+'/Jz_data/',exist_ok=True)


		sketching_arrays = self.initialize_sampling(derivatives = derivatives,output_basis = output_basis)
		Omega_m = sketching_arrays['Omega_m']
		Omega_z = sketching_arrays['Omega_z']
		Phi = sketching_arrays['Phi']
		JTPhi = sketching_arrays['JTPhi']



		if self.settings['verbose']:
			print(80*'#')

		for i in range(n_samples):
			t0_samplei = time.time()
			################################################################################
			# Sample forward map m -> q(m) or m,z -> q(m,z) and save
			self.parRandom.normal(1,self.noise)

			self.m.zero()
			self.prior.sample(self.noise,self.m)



			if self.control_distribution is not None:
				self.control_distribution.sample(self.z)
				x = [self.u,self.m,None,self.z]
			else:
				x = [self.u,self.m,None]


			self.observable.solveFwd(self.u,x)
			self.observable.setLinearizationPoint(x)
			this_m = self.m.get_local()
			this_q = self.observable.evalu(self.u).get_local()

			np.save(data_dir+'mq_data/m_sample_'+str(i)+'.npy',this_m)
			np.save(data_dir+'mq_data/q_sample_'+str(i)+'.npy',this_q)

			if self.control_distribution is not None:
				this_z = self.z.get_local()
				np.save(data_dir+'z_sample_'+str(i)+'.npy',this_z)

			fwd_sample_time = time.time() -t0_samplei



			################################################################################
			# Derivative computations and saving

			if derivatives[0] == 1:
				t0_jacobian = time.time()
				if output_basis is not None:
					assert Phi is not None
					assert JTPhi is not None
					JTPhi.zero()
					hp.MatMvTranspmult(self.J,Phi,JTPhi)
					JPhi_np = hf.mv_to_dense(JTPhi)
					np.save(data_dir+'J_data/JTPhi'+str(i)+'.npy',JPhi_np)
				else:
					# Compute it with randomized SVD
					rM = self.settings['rM']
					
					Omega_m.zero() # probably unecessary
					hp.parRandom.normal(1.,Omega_m)
					U, sigma, V = hp.accuracyEnhancedSVD(self.J,Omega_m,rM, s=1)
					Unp = hf.mv_to_dense(U)
					Vnp = hf.mv_to_dense(V)

					np.save(data_dir+'J_data/U_sample_'+str(i)+'.npy',Unp)
					np.save(data_dir+'J_data/sigma_sample_'+str(i)+'.npy',sigma)
					np.save(data_dir+'J_data/V_sample_'+str(i)+'.npy',Vnp)

				jacobian_time = time.time() - t0_jacobian

			if derivatives[1] == 1:
				t0_control_jacobian = time.time()
				if output_basis is not None:
					assert Phi is not None
					assert JzTPhi is not None
					JzTPhi.zero()
					hp.MatMvTranspmult(self.Jz,Phi,JzTPhi)
					JzPhi_np = hf.mv_to_dense(JzTPhi)
					np.save(data_dir+'Jz_data/JzTPhi'+str(i)+'.npy',JzPhi_np)
				else:
					Omega_z.zero() # probably unecessary
					hp.parRandom.normal(1.,Omega_z)

					Uz, sigmaz, Vz = hp.accuracyEnhancedSVD(self.Jz,Omega_z,rZ, s=1)
					Uznp = hf.mv_to_dense(Uz)
					Vznp = hf.mv_to_dense(Vz)

					np.save(jacobian_process_specific_directory+'Uz_sample_'+str(i)+'.npy',Uznp)
					np.save(jacobian_process_specific_directory+'sigmaz_sample_'+str(i)+'.npy',sigmaz)
					np.save(jacobian_process_specific_directory+'Vz_sample_'+str(i)+'.npy',Vznp)
				
				control_jacobian_time = time.time() - t0_control_jacobian

			################################################################################
			# Printing
			if self.settings['verbose']:
				message = 'Sample '+str(i)+' generation took '+('{:.2f}'.format(fwd_sample_time))+'s'
				print(message.center(80))
				if derivatives[0] == 1:
					messageJ = 'J sample '+str(i)+' generation took '+('{:.2f}'.format(jacobian_time))+'s'
					print(messageJ.center(80))
				if derivatives[1] == 1:
					messageJz = 'Jz sample '+str(i)+' generation took '+('{:.2f}'.format(control_jacobian_time))+'s'
					print(messageJz.center(80))

		################################################################################
		if compress:
			print('Commencing compression'.center(80))
			has_z_data = hasattr(observable.model.problem, 'Cz')
			compress_dataset(data_dir,derivatives = derivatives, clean_up = True,has_z_data = has_z_data)



	def initialize_sampling(self,derivatives,output_basis = None):
		"""
		"""
		Omega_m = None
		Omega_z = None
		Phi = None
		JTPhi = None
		JzTPhi = None

		setup_parameter_jacobian = bool(derivatives[0])
		setup_control_jacobian = bool(derivatives[1])

		if self.u is None:
			self.u = self.observable.generate_vector(hp.STATE)
		if self.m is None:
			self.m = self.observable.generate_vector(hp.PARAMETER)
		if self.control_distribution is not None and self.z is None:
			self.z = self.observable.generate_vector(CONTROL)

		if self.control_distribution is not None:
			assert self.observable.problem.Vh[hp.STATE].mesh().mpi_comm().size == 1, print('Only worked out for serial codes')
			self.control_dimension = self.z.get_local().shape[0]

		qsample = dl.Vector(self.mesh_constructor_comm)
		self.observable.B.init_vector(qsample,0)
		
		################################################################################
		# Setting up the derivative computations

		if derivatives[0]:
			self.J = hf.ObservableJacobian(self.observable)
			self.dQ, self.dM = self.J.shape

			if output_basis is not None:
				assert self.mesh_constructor_comm.size == 1, print('Only worked out for serial codes')
				Phi = hf.dense_to_mv_local(output_basis,qsample)
				JTPhi = hp.MultiVector(self.m,Phi.nvec())
			else:
				rM = self.settings['rM']
				oversample = self.settings['oversample']

				parameter_vector = dl.Vector(self.mesh_constructor_comm)
				self.J.init_vector(parameter_vector,1)
				nvec_Omega_m = min(rM +oversample,self.dQ,self.dM) # Fix me in the case that min is dQ :)
				Omega_m = hp.MultiVector(parameter_vector,nvec_Omega_m)
				hp.parRandom.normal(1.,Omega_m)

		

		if derivatives[1]:
			self.Jz = hf.ObservableControlJacobian(self.observable)
			self.dQ,self.dZ = self.Jz.shape

			if output_basis is not None:
				assert self.mesh_constructor_comm.size == 1, print('Only worked out for serial codes')
				if Phi is None:
					Phi = hf.dense_to_mv_local(output_basis,qsample)
				JzTPhi = hp.MultiVector(self.z,Phi.nvec())
			else:
				rZ = self.settings['rZ']
				oversample = self.settings['oversample']

				control_vector = dl.Vector(self.mesh_constructor_comm)
				self.Jz.init_vector(control_vector,1)
				nvec_Omega_z = min(rZ +oversample,self.dQ,self.dZ)		# Fix me in the case that min is dQ :)
				Omega_z = hp.MultiVector(control_vector,nvec_Omega_m)
				hp.parRandom.normal(1.,Omega_z)


		if self.dM is None:
			self.dM = self.m.size()

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




		return {'Omega_m':Omega_m, 'Omega_z': Omega_z, 'Phi': Phi, 'JTPhi':JTPhi, 'JzTPhi':JzTPhi}



def compress_dataset(file_path,derivatives = (0,0), clean_up = True,has_z_data = False):

	################################################################################
	# Pre-processing and array allocations

	# Booleans about what to save and assertions for safeguarding
	if derivatives[0]:
		compress_JTPhi = True
		compress_Jsvd = True

	if derivatives[1]:
		assert has_z_data
		compress_JzTPhi = True
		compress_Jzsvd = True

	if has_z_data:
		data_path = file_path+'/mqz_data/'
	else:
		data_path = file_path+'/mq_data/'
	
	all_files = os.listdir(data_path)
	
	ndata = 0

	for file in all_files:
		if 'm_sample' in file:
			index = int(file.split('m_sample_')[-1].split('.npy')[0])
			ndata = max(ndata,index)
			assert os.path.exists(data_path+'q_sample_'+str(index)+'.npy')
			if has_z_data:
				assert os.path.exists(data_path+'z_sample_'+str(index)+'.npy')
			if derivatives[0]:
				JTPhi_exists = os.path.exists(file_path+'/J_data/JTPhi'+str(index)+'.npy')
				Jsvd_exists = os.path.exists(file_path+'/J_data/U_sample_'+str(index)+'.npy') and \
								os.path.exists(file_path+'/J_data/sigma_sample_'+str(index)+'.npy') and \
								os.path.exists(file_path+'/J_data/V_sample_'+str(index)+'.npy')
				assert JTPhi_exists or Jsvd_exists
				compress_JTPhi = compress_JTPhi and JTPhi_exists
				compress_Jsvd = compress_Jsvd and Jsvd_exists

			if derivatives[1]:
				JzTPhi_exists = os.path.exists(file_path+'/Jz_data/JzTPhi'+str(index)+'.npy')
				Jzsvd_exists = os.path.exists(file_path+'/Jz_data/Uz_sample_'+str(index)+'.npy') and \
								os.path.exists(file_path+'/Jz_data/sigmaz_sample_'+str(index)+'.npy') and \
								os.path.exists(file_path+'/Jz_data/Vz_sample_'+str(index)+'.npy')
				assert JzTPhi_exists or Jzsvd_exists
				compress_JzTPhi = compress_JzTPhi and JzTPhi_exists
				compress_Jzsvd = compress_Jzsvd and Jzsvd_exists
	# Python indexing
	ndata+=1

	print('Total number of data = ',ndata)

	dM = np.load(data_path+'/m_sample_'+str(index)+'.npy').shape[0]
	dQ = np.load(data_path+'/q_sample_'+str(index)+'.npy').shape[0]
	m_data = np.zeros((ndata,dM))
	q_data = np.zeros((ndata,dQ))
	print('dM = ',dM)
	print('dQ = ',dQ)

	if has_z_data:
		dZ = np.load(data_path+'/z_sample_'+str(index)+'.npy').shape[0]
		z_data = np.zeros((ndata,dZ))
		print('dZ = ',dZ)

	print('Compressing mq data'.center(80))
	t0_compress = time.time()
	

	if derivatives[0]:
		if compress_JTPhi:
			rQ = np.load(file_path+'/J_data/JTPhi'+str(index)+'.npy').shape[1]
			JTPhi_data = np.zeros((ndata,dM,rQ))
		if compress_Jsvd:
			rank = np.load(file_path+'/J_data/sigma_sample_'+str(index)+'.npy').shape[1]
			U_data = np.zeros((ndata,dQ,rank))
			sigma_data = np.zeros((ndata,rank))
			V_data = np.zeros((ndata,dM,rank))
	if derivatives[1]:
		if compress_JzTPhi:
			JzTPhi_data = np.zeros((ndata,dQ,dZ))
		if compress_Jsvd:
			rank = np.load(file_path+'/Jz_data/sigmaz_sample_'+str(index)+'.npy').shape[1]
			Uz_data = np.zeros((ndata,dQ,rank))
			sigmaz_data = np.zeros((ndata,rank))
			Vz_data = np.zeros((ndata,dZ,rank))

	for index in range(0,ndata):
		m_data[index] = np.load(data_path+'/m_sample_'+str(index)+'.npy')
		q_data[index] = np.load(data_path+'/q_sample_'+str(index)+'.npy')


		if derivatives[0]:
			if compress_JTPhi:
				JTPhi_data[index] = np.load(file_path+'/J_data/JTPhi'+str(index)+'.npy')
			if compress_Jsvd:
				U_data[index] = np.load(file_path+'/J_data/U_sample_'+str(index)+'.npy')
				sigma_data[index] = np.load(file_path+'/J_data/sigma_sample_'+str(index)+'.npy')
				V_data[index] = np.load(file_path+'/J_data/V_sample_'+str(index)+'.npy')

		if derivatives[1]:
			if compress_JzTPhi:
				JzTPhi_data[index] = np.load(file_path+'/Jz_data/JzTPhi'+str(index)+'.npy')
			if compress_Jsvd:
				Uz_data[index] = np.load(file_path+'/Jz_data/Uz_sample_'+str(index)+'.npy')
				sigmaz_data[index] = np.load(file_path+'/Jz_data/sigmaz_sample_'+str(index)+'.npy')
				Vz_data[index] = np.load(file_path+'/Jz_data/Vz_sample_'+str(index)+'.npy')


	if has_z_data:
		np.savez_compressed(file_path+'mq_data.npz',m_data = m_data, q_data = q_data)
	else:
		np.savez_compressed(file_path+'mqz_data.npz',m_data = m_data, q_data = q_data)
	if derivatives[0]:
		if compress_JTPhi:
			np.savez_compressed(file_path+'JTPhi_data.npz',JTPhi_data = JTPhi_data)
		if compress_Jsvd:
			np.savez_compressed(file_path+'Jsvd_data.npz',U_data = U_data, sigma_data =sigma_data, V_data = V_data)

	if derivatives[1]:
		if compress_JzTPhi:
			np.savez_compressed(file_path+'JzTPhi_data.npz',JzTPhi_data = JzTPhi_data)
		if compress_Jzsvd:
			np.savez_compressed(file_path+'Jzsvd_data.npz',Uz_data = Uz_data, sigmaz_data =sigmaz_data, Vz_data = Vz_data)

	print('Whole process took ',time.time() - t0_compress,'s')

	if clean_up:
		os.system(' rm -r '+data_path)
		if derivatives[0]:
			os.system(' rm -r '+file_path+'/J_data/')
		if derivatives[1]:
			os.system(' rm -r '+file_path+'/Jz_data/')

		print('Cleanup successful'.center(80))

