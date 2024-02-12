# Copyright (c) 2020-2024, The University of Texas at Austin 
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

CONTROL = 3

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
					output_basis = None, input_basis = None,\
					data_dir = 'data/test/', compress = True, clean_up = True):
		"""
		"""
		if self.control_distribution is not None:
			os.makedirs(data_dir+'/mqz_data/',exist_ok=True)
		else:
			os.makedirs(data_dir+'/mq_data/',exist_ok=True)
		if derivatives[0]:
			os.makedirs(data_dir+'/J_data/',exist_ok=True)
		if derivatives[1]:
			assert self.control_distribution is not None
			assert hasattr(self.observable.problem,'Cz')
			os.makedirs(data_dir+'/Jz_data/',exist_ok=True)


		sketching_arrays = self.initialize_sampling(derivatives = derivatives,\
					output_basis = output_basis, input_basis = input_basis)
		Omega_m = sketching_arrays['Omega_m']
		Omega_z = sketching_arrays['Omega_z']
		Phi = sketching_arrays['Phi']
		JTPhi = sketching_arrays['JTPhi']
		Psi = sketching_arrays['Psi']
		JPsi = sketching_arrays['JPsi']

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

			if self.control_distribution is None:
				np.save(data_dir+'mq_data/m_sample_'+str(i)+'.npy',this_m)
				np.save(data_dir+'mq_data/q_sample_'+str(i)+'.npy',this_q)
			else:
				np.save(data_dir+'mqz_data/m_sample_'+str(i)+'.npy',this_m)
				np.save(data_dir+'mqz_data/q_sample_'+str(i)+'.npy',this_q)
				this_z = self.z.get_local()
				np.save(data_dir+'mqz_data/z_sample_'+str(i)+'.npy',this_z)

			fwd_sample_time = time.time() -t0_samplei

			################################################################################
			# Derivative computations and saving

			if derivatives[0]:
				t0_jacobian = time.time()
				if output_basis is not None:
					assert Phi is not None
					assert JTPhi is not None
					JTPhi.zero()
					hp.MatMvTranspmult(self.J,Phi,JTPhi)
					JTPhi_np = hf.mv_to_dense(JTPhi)
					np.save(data_dir+'J_data/JTPhi'+str(i)+'.npy',JTPhi_np)
				elif input_basis is not None:
					assert Psi is not None
					assert JPsi is not None
					JPsi.zero()
					hp.MatMvMult(self.J,Psi,JPsi)
					JPsi_np = hf.mv_to_dense(JPsi)
					np.save(data_dir+'J_data/JPsi'+str(i)+'.npy',JPsi_np)

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

			if derivatives[1]:
				t0_control_jacobian = time.time()
				if output_basis is not None:
					assert Phi is not None
					assert JzTPhi is not None
					JzTPhi.zero()
					hp.MatMvTranspmult(self.Jz,Phi,JzTPhi)
					JzTPhi_np = hf.mv_to_dense(JzTPhi)
					np.save(data_dir+'Jz_data/JzTPhi'+str(i)+'.npy',JzTPhi_np)
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
			has_z_data = hasattr(self.observable.problem, 'Cz')
			compress_dataset(data_dir,derivatives = derivatives, clean_up = clean_up,\
							has_z_data = has_z_data, input_basis = input_basis, output_basis = output_basis)


	def compute_jacobians_in_subspace(self, derivatives, output_basis, data_file_name, data_dir, compress=True,\
						 clean_up=True,compress_derivatives_only = True):
		sketching_arrays = self.initialize_sampling(derivatives = derivatives, output_basis = output_basis)
		Phi = sketching_arrays['Phi']
		if derivatives[0]:
			JTPhi = sketching_arrays['JTPhi']
		if derivatives[1]:
			JzTPhi = sketching_arrays['JzTPhi']

		data = np.load(data_dir+data_file_name)
		m_data = data['m_data']
		u_data = data['q_data']
		if self.control_distribution is not None:
			z_data = data['z_data']

		N_data = m_data.shape[0]
		if derivatives[0]:
			os.makedirs(data_dir+'/J_data/',exist_ok=True)
		if derivatives[1]:
			os.makedirs(data_dir+'/Jz_data/',exist_ok=True)

		for i in range(N_data):
			m = m_data[i]
			u = u_data[i]
			self.m.set_local(m)
			self.u.set_local(u)
			x = [self.u, self.m, None]
			if self.control_distribution is not None:
				z = z_data[i]
				self.z.set_local(z)
				x.append(self.z)
			self.observable.setLinearizationPoint(x)
			
			if derivatives[0]:
				JTPhi.zero()
				hp.MatMvTranspmult(self.J,Phi,JTPhi)
				JTPhi_np = hf.mv_to_dense(JTPhi)
				np.save(data_dir+f'J_data/JTPhi{i}.npy', JTPhi_np)

			if derivatives[1]:
				JzTPhi.zero()
				hp.MatMvTranspmult(self.Jz,Phi,JzTPhi)
				JzTPhi_np = hf.mv_to_dense(JzTPhi)
				np.save(data_dir+f'Jz_data/JzTPhi{i}.npy', JzTPhi_np)




		################################################################################
		if compress:
			print('Commencing compression'.center(80))
			has_z_data = hasattr(self.observable.problem, 'Cz')
			compress_dataset(data_dir,derivatives = derivatives, clean_up = clean_up, has_z_data = has_z_data,\
				 input_basis = None, output_basis = output_basis)


	def two_step_generate(self,n_samples, n_samples_pod=None, derivatives = (0,0),\
					pod_rank = None, data_dir = 'data/test/', compress = True, clean_up = True):
		# Assert that this is a full state PDE problem.
		assert type(self.observable.B) is hf.StateSpaceIdentityOperator

		if n_samples_pod is None:
			# Could be risky
			n_samples_pod = n_samples

		# Step 1. Generate m -> u(m) or (m,z) -> u(m,z)
		self.generate(n_samples, derivatives = (0,0),data_dir = data_dir, compress = True, clean_up = False)
		# Step 1.5 Compute POD
		if self.control_distribution is not None:
			data_file_name = 'mqz_data.npz'
			all_data = np.load(data_dir+'mqz_data.npz')
		else:
			data_file_name = 'mq_data.npz'
			all_data = np.load(data_dir+'mq_data.npz')
		u_data = all_data['q_data'][:n_samples_pod]
		POD = hf.PODProjectorFromData(self.observable.problem.Vh)
		d_POD, phi, Mphi, u_shift = POD.construct_subspace(u_data,pod_rank)
		if True:
			PsistarPsi = Mphi.T@phi
			orth_error = np.linalg.norm(PsistarPsi - np.eye(PsistarPsi.shape[0]))
			print('||Psi^*Psi - I|| = ',orth_error)
			assert orth_error < 1e-5

		# Step 2.
		self.compute_jacobians_in_subspace(derivatives = derivatives, output_basis = phi,\
						 data_file_name = data_file_name, data_dir = data_dir, compress_derivatives_only = True)



	def initialize_sampling(self,derivatives,output_basis = None,input_basis = None):
		"""
		"""
		Omega_m = None
		Omega_z = None
		Phi = None
		JTPhi = None
		JzTPhi = None
		# Psi is for the first parameter "m" and has no relation to z
		Psi = None
		JPsi = None
		

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
			elif input_basis is not None:
				assert self.mesh_constructor_comm.size == 1, print('Only worked out for serial codes')
				Psi = hf.dense_to_mv_local(input_basis,self.m)
				JPsi = hp.MultiVector(qsample,Psi.nvec())
			else:
				rM = self.settings['rM']
				oversample = self.settings['oversample']

				parameter_vector = self.observable.problem.generate_parameter()
				# self.J.init_vector(parameter_vector,1)
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

		return {'Omega_m':Omega_m, 'Omega_z': Omega_z,\
				 'Phi': Phi, 'JTPhi':JTPhi, 'JzTPhi':JzTPhi,\
				 'Psi': Psi, 'JPsi': JPsi}



def compress_dataset(file_path,derivatives = (0,0), clean_up = True,\
					has_z_data = False, input_basis = None, output_basis = None,\
					derivatives_only = False):

	################################################################################
	# Pre-processing and array allocations

	# Booleans about what to save and assertions for safeguarding
	if derivatives[0]:
		compress_JTPhi = True
		compress_JPsi = True
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
		if file.startswith('m_sample') and file.endswith('.npy'):
			index = int(file.split('m_sample_')[-1].split('.npy')[0])
			ndata = max(ndata,index)
			assert os.path.exists(data_path+'q_sample_'+str(index)+'.npy')
			if has_z_data:
				assert os.path.exists(data_path+'z_sample_'+str(index)+'.npy')
			if derivatives[0]:
				JTPhi_exists = os.path.exists(file_path+'/J_data/JTPhi'+str(index)+'.npy')
				JPsi_exists = os.path.exists(file_path+'/J_data/JPsi'+str(index)+'.npy')
				Jsvd_exists = os.path.exists(file_path+'/J_data/U_sample_'+str(index)+'.npy') and \
								os.path.exists(file_path+'/J_data/sigma_sample_'+str(index)+'.npy') and \
								os.path.exists(file_path+'/J_data/V_sample_'+str(index)+'.npy')
				assert JTPhi_exists or Jsvd_exists or JPsi_exists
				compress_JTPhi = compress_JTPhi and JTPhi_exists
				compress_JPsi = compress_JPsi and JPsi_exists
				compress_Jsvd = compress_Jsvd and Jsvd_exists

			if derivatives[1]:
				JzTPhi_exists = os.path.exists(file_path+'/Jz_data/JzTPhi'+str(index)+'.npy')
				Jzsvd_exists = os.path.exists(file_path+'/Jz_data/Uz_sample_'+str(index)+'.npy') and \
								os.path.exists(file_path+'/Jz_data/sigmaz_sample_'+str(index)+'.npy') and \
								os.path.exists(file_path+'/Jz_data/Vz_sample_'+str(index)+'.npy')
				assert JzTPhi_exists or Jzsvd_exists
				compress_JzTPhi = compress_JzTPhi and JzTPhi_exists
				compress_Jzsvd = compress_Jzsvd and Jzsvd_exists
	if ndata == 0:
		print('Some issue has arisen, no data found.')
		raise

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
		if compress_JPsi:
			rM = np.load(file_path+'/J_data/JPsi'+str(index)+'.npy').shape[1]
			JPsi_data = np.zeros((ndata,dQ,rM))

		if compress_Jsvd:
			rank = np.load(file_path+'/J_data/sigma_sample_'+str(index)+'.npy').shape[0]
			U_data = np.zeros((ndata,dQ,rank))
			sigma_data = np.zeros((ndata,rank))
			V_data = np.zeros((ndata,dM,rank))
	if derivatives[1]:
		if compress_JzTPhi:
			JzTPhi_data = np.zeros((ndata,dZ,rQ))
		if compress_Jsvd:
			rank = np.load(file_path+'/Jz_data/sigmaz_sample_'+str(index)+'.npy').shape[1]
			Uz_data = np.zeros((ndata,dQ,rank))
			sigmaz_data = np.zeros((ndata,rank))
			Vz_data = np.zeros((ndata,dZ,rank))

	for index in range(0,ndata):
		if not derivatives_only:
			m_data[index] = np.load(data_path+'/m_sample_'+str(index)+'.npy')
			q_data[index] = np.load(data_path+'/q_sample_'+str(index)+'.npy')
			if has_z_data:
				z_data[index] = np.load(data_path+'/z_sample_'+str(index)+'.npy')


		if derivatives[0]:
			if compress_JTPhi:
				JTPhi_data[index] = np.load(file_path+'/J_data/JTPhi'+str(index)+'.npy')
			if compress_JPsi:
				JPsi_data[index] = np.load(file_path+'/J_data/JPsi'+str(index)+'.npy')
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

	if not derivatives_only:
		if has_z_data:
			np.savez_compressed(file_path+'mqz_data.npz',m_data = m_data, q_data = q_data,z_data = z_data)
		else:
			np.savez_compressed(file_path+'mq_data.npz',m_data = m_data, q_data = q_data)
	if derivatives[0]:
		if compress_JTPhi:
			np.savez_compressed(file_path+'JTPhi_data.npz',JTPhi_data = JTPhi_data,Phi = output_basis)
		if compress_JPsi:
			np.savez_compressed(file_path+'JPsi_data.npz',JPsi_data = JPsi_data,Psi = input_basis)
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

