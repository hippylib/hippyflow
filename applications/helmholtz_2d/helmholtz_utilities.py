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

import numpy as np
import os

def load_helmholtz_data(data_dir,rescale = False,derivatives = False,n_data = np.inf):
	assert os.path.isdir(data_dir)
	data_files = os.listdir(data_dir)
	data_files = [data_dir + file for file in data_files]

	m_files = []
	q_files = []
	mq_files = []
	for file in data_files:
		if 'ms_on_rank_' in file:
			m_files.append(file)
		if 'qs_on_rank_' in file:
			q_files.append(file)
		if 'mq_on_rank_' in file:
			mq_files.append(file)

	if len(mq_files) == 0:
		ranks = [int(file.split(data_dir+'ms_on_rank_')[-1].split('.npy')[0]) for file in m_files]
	else:
		ranks = [int(file.split(data_dir+'mq_on_rank_')[-1].split('.npz')[0]) for file in m_files]
	max_rank = max(ranks)

	# Serially concatenate data
	if len(mq_files) == 0:
		m_data = np.load(data_dir+'ms_on_rank_0.npy')
		q_data = np.load(data_dir+'qs_on_rank_0.npy')
		for i in range(1,max_rank+1):
				appendage_m = np.load(data_dir+'ms_on_rank_'+str(i)+'.npy')
				m_data = np.concatenate((m_data,appendage_m))
				appendage_q = np.load(data_dir+'qs_on_rank_'+str(i)+'.npy')
				q_data = np.concatenate((q_data,appendage_q))
	else:
		npz_data = np.load(data_dir+'mq_on_rank_0.npz')
		m_data = npz_data['m_data']
		q_data = npz_data['q_data']
		for i in range(1,max_rank+1):
			npz_data = np.load(data_dir+'mq_on_rank_'+str(i)+'.npz')
			appendage_m = npz_data['m_data']
			appendage_q = npz_data['q_data']
			m_data = np.concatenate((m_data,appendage_m))
			q_data = np.concatenate((q_data,appendage_q))

	if n_data < np.inf:
		assert type(n_data) is int
		m_data = m_data[:n_data]
		q_data = q_data[:n_data]
	if rescale:
		from sklearn import preprocessing
		m_data = preprocessing.scale(m_data)
		q_data = preprocessing.scale(q_data)

	data_dict = {'m_data': m_data,'q_data':q_data}

	if derivatives:
		U_files = []
		sigma_files = []
		V_files = []
		for file in data_files:
			if 'Us_on_rank_' in file:
				U_files.append(file)
			if 'sigmas_on_rank_' in file:
				sigma_files.append(file)
			if 'Vs_on_rank_' in file:
				V_files.append(file)
		if not U_files or not sigma_files or not V_files:
			print('No derivative data'.center(80))
		else:
			ranks = [int(file.split(data_dir+'sigmas_on_rank_')[-1].split('.npy')[0]) for file in sigma_files]
			max_rank = max(ranks)

			# Serially concatenate derivative data
			U_data = np.load(data_dir+'Us_on_rank_0.npy')
			sigma_data = np.load(data_dir+'sigmas_on_rank_0.npy')
			V_data = np.load(data_dir+'Vs_on_rank_0.npy')
			for i in range(1,max_rank+1):
				appendage_U = np.load(data_dir+'Us_on_rank_'+str(i)+'.npy')
				U_data = np.concatenate((U_data,appendage_U))
				appendage_sigma = np.load(data_dir+'sigmas_on_rank_'+str(i)+'.npy')
				sigma_data = np.concatenate((sigma_data,appendage_sigma))
				appendage_V = np.load(data_dir+'Vs_on_rank_'+str(i)+'.npy')
				V_data = np.concatenate((V_data,appendage_V))

		if n_data < np.inf:
			assert type(n_data) is int
			U_data = U_data[:n_data]
			sigma_data = sigma_data[:n_data]
			V_data = V_data[:n_data]

		if rescale:
			raise NotImplementedError('This needs to be thought out with care')

			data_dict['U_data'] = U_data
			data_dict['sigma_data'] = sigma_data
			data_dict['V_data'] = V_data
		

	return data_dict


def get_projectors(data_dir,as_input_tolerance=1e-4,as_output_tolerance=1e-4,\
					kle_tolerance = 1e-4,pod_tolerance = 1e-4,\
					 fixed_input_rank = 0, fixed_output_rank = 0, mixed_output = True, verbose = False):
	projector_dictionary = {}
	################################################################################
	# Derivative Informed Input Subspace
	AS_input_projector = np.load(data_dir+'AS_input_projector.npy')
	if verbose:
		print('AS input projector shape before truncation = ', AS_input_projector.shape)
	if fixed_input_rank > 0:
		AS_input_projector = AS_input_projector[:,:fixed_input_rank]
	else:
		AS_input_d = np.load(data_dir+'AS_d_GN.npy')
		AS_input_projector = AS_input_projector[:,np.where(AS_input_d>as_input_tolerance)[0]]
	if verbose:
		print('AS input projector shape after truncation = ', AS_input_projector.shape)
	projector_dictionary['AS_input'] = AS_input_projector
	################################################################################
	# Derivative Informed Output Subspace
	AS_output_projector = np.load(data_dir+'AS_output_projector.npy')
	
	if verbose:
		print('AS output projector shape before truncation = ', AS_output_projector.shape)
	if fixed_output_rank > 0:
		AS_output_projector = AS_output_projector[:,:fixed_output_rank]
	else:
		AS_output_d = np.load(data_dir+'AS_d_NG.npy')
		AS_output_projector = AS_output_projector[:,np.where(AS_output_d>as_output_tolerance)[0]]
	if verbose:
		print('AS output projector shape after truncation = ', AS_output_projector.shape)
	projector_dictionary['AS_output'] = AS_output_projector
	################################################################################
	# KLE Input Subspace
	KLE_projector = np.load(data_dir+'KLE_projector.npy')
	if verbose:
		print('KLE projector shape before truncation = ', KLE_projector.shape)
	if fixed_input_rank > 0:
		KLE_projector = KLE_projector[:,:fixed_input_rank]
	else:
		KLE_d = np.load(data_dir+'KLE_d.npy')
		KLE_projector = KLE_projector[:,np.where(KLE_d>kle_tolerance)[0]]
	if verbose:
		print('KLE projector shape after truncation = ', KLE_projector.shape)
	projector_dictionary['KLE'] = KLE_projector
	################################################################################
	# POD Output Subspace
	POD_projector = np.load(data_dir+'POD_projector.npy')
	if verbose:
		print('POD projector shape before truncation = ', POD_projector.shape)
	if fixed_output_rank > 0:
		POD_projector = POD_projector[:,:fixed_output_rank]
	else:
		POD_d = np.load(data_dir+'POD_d.npy')
		POD_projector = POD_projector[:,np.where(POD_d>pod_tolerance)[0]]
	if verbose:
		print('POD projector shape after truncation = ', POD_projector.shape)
	projector_dictionary['POD'] = POD_projector
	return projector_dictionary

def modify_projectors(projectors,input_subspace,output_subspace):
	# Modify the input projectors
	assert input_subspace in ['kle','as','random']

	if input_subspace in ['kle','as']:
		# Always orthogonalize AS and KLE for best results
		orthogonalize_input = True
		rescale_input = True
		if input_subspace == 'kle':
			input_projector = projectors['KLE']
		elif input_subspace == 'as':
			input_projector = projectors['AS_input']


		if orthogonalize_input:
			input_projector,_ = np.linalg.qr(input_projector)

		if rescale_input:
			# Scaling factor of 10 seemed to perform well for KLE and AS
			# and this was independent of the projector rank.
			scale_factor_input = 0.05*float(input_projector.shape[0])/(32*float(input_projector.shape[-1]))
			input_projector /= scale_factor_input*np.linalg.norm(input_projector)

	elif input_subspace == 'random':
		input_projector = np.random.randn(*projectors['KLE'].shape)
		input_projector,_ = np.linalg.qr(input_projector)
		scale_factor_input = 0.05*float(input_projector.shape[0])/(32*float(input_projector.shape[-1]))
		input_projector /= scale_factor_input*np.linalg.norm(input_projector)

	# Modify the output projectors
	# It seems that (re)-orthogonalizing the POD vectors
	# may not improve the neural network.
	assert output_subspace in ['pod','as','random']

	if output_subspace in ['pod','as']:
		orthogonalize_output = True
		rescale_output = True
		if output_subspace == 'pod':
			output_projector = projectors['POD']
		elif output_subspace == 'as':
			output_projector = projectors['AS_output']

		if orthogonalize_output:
			output_projector,_ = np.linalg.qr(output_projector)

		if rescale_output:
			scale_factor_output = 1.
			output_projector /= scale_factor_output*np.linalg.norm(output_projector)

	if output_subspace == 'random':
		output_projector = np.random.randn(*projectors['POD'].shape)
		output_projector /= np.linalg.norm(output_projector)

	return input_projector, output_projector