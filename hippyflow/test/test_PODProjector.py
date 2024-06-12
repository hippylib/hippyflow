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

import unittest 
import dolfin as dl
import ufl
import numpy as np
import matplotlib.pyplot as plt 
from scipy.sparse import csr_matrix 


import sys, os
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp

sys.path.append(os.environ.get('HIPPYFLOW_PATH'))
import hippyflow as hf



class TestPODProjectorFromData(unittest.TestCase):
	def setUp(self):
		ndim = 2
		nx = 8
		ny = 8
		self.n_data = 100
		self.mesh = dl.UnitSquareMesh(nx, ny)
		self.Vh, self.observable, self.M = self._build_pde_problem(self.mesh)
		self.prior = hf.BiLaplacian2D(self.Vh[hp.PARAMETER],gamma = 0.1, delta = 1.0)
		self.u_data = self._sample_data(self.observable, self.prior, self.n_data)

		M_mat = dl.as_backend_type(self.M).mat()
		row, col, val = M_mat.getValuesCSR()
		self.M_csr = csr_matrix((val, col, row))

	def test_pod_with_shift(self):
		"""
		Test the POD constructor with a shift 

		 - Orthogonality check for decoder and encoder
		 - Check the shift is non zero 
		 - Check POD satisfies eigenvalue problem 
		"""
		u_rank = 15
		shift = True
		methods = ['hep', 'ghep', 'inverse_ghep']

		for method in methods:
			d, decoder, encoder, u_shift = self._construct_subspace(u_rank, shift, method)
			self._check_orthogonality(decoder, encoder)
			self._check_shift(u_shift, shift)
			self._check_eigenvalue_problem(self.u_data, d, decoder, encoder, u_shift)

	def test_pod_no_shift(self):
		"""
		Test the POD constructor with no shift 

		 - Orthogonality check for decoder and encoder
		 - Check the shift is zero
		 - Check POD satisfies eigenvalue problem 
		"""
		u_rank = 15
		shift = False
		methods = ['hep', 'ghep', 'inverse_ghep']

		for method in methods:
			d, decoder, encoder, u_shift = self._construct_subspace(u_rank, shift, method)
			self._check_orthogonality(decoder, encoder)
			self._check_shift(u_shift, shift)
			self._check_eigenvalue_problem(self.u_data, d, decoder, encoder, u_shift)


	def _build_pde_problem(self, mesh):
		"""
		Build the PDE problem

		:returns:	
		 - :code:`Vh` List of function spaces
		 - :code:`hf.Observable` The observable object
		 - :code:`M` the mass matrix 
		"""
		def u_boundary(x, on_boundary):
			return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

		self.rank = dl.MPI.rank(mesh.mpi_comm())
			
		Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
		Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
		Vh = [Vh2, Vh1, Vh2]
		# Initialize Expressions
		f = dl.Constant(0.0)
			
		u_bdr = dl.Expression("x[1]", degree=1)
		u_bdr0 = dl.Constant(0.0)
		bc = dl.DirichletBC(Vh[hp.STATE], u_bdr, u_boundary)
		bc0 = dl.DirichletBC(Vh[hp.STATE], u_bdr0, u_boundary)
		
		def pde_varf(u,m,p):
			return ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - f*p*ufl.dx

		self.pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

		u_trial = dl.TrialFunction(Vh[hp.STATE])
		u_test = dl.TestFunction(Vh[hp.STATE])

		M = dl.PETScMatrix()
		dl.assemble(u_trial*u_test*dl.dx, tensor=M)

		B = hf.StateSpaceIdentityOperator(M)
		obs = hf.LinearStateObservable(self.pde,B)
		return Vh, obs, M

	def _sample_data(self, observable, prior, n_data):
		"""
		Sample the state data by solving the PDE for different prior samples
		"""
		x = observable.generate_vector()
		noise = dl.Vector()
		prior.init_vector(noise, "noise")

		state_dim = x[hp.STATE].get_local().shape[0]
		u_data = np.zeros((n_data, state_dim))	

		for i in range(n_data):
			hp.parRandom.normal(1.0, noise)
			prior.sample(noise, x[hp.PARAMETER])
			observable.solveFwd(x[hp.STATE], x)
			u_data[i] = x[hp.STATE].get_local()

		return u_data 
	
	def _construct_subspace(self, u_rank, shift, pod_method):
		"""
		Run the subspace construction with the given rank, shift, and choice of POD method 
		"""
		pod_constructor = hf.PODProjectorFromData(self.Vh, self.M)

		d, decoder, encoder, u_shift = pod_constructor.construct_subspace(self.u_data, u_rank, 
																	shifted=shift, 
																	method=pod_method, 
																	verify=True)
		return d, decoder, encoder, u_shift 

	def _check_orthogonality(self, decoder, encoder):
		"""
		Check the orthogonality properties of the decoder and encoder 

		 - Decoder and encoder should be M-orthogonal
		 - Encoder should :code:`M @ decoder`
		"""
		fro_tol = 1e-8 

		# Orthogonality check 
		UMU = decoder.T @ encoder
		identity_of_size_U = np.eye(UMU.shape[0])
		U_orth_error = np.linalg.norm(identity_of_size_U - UMU, 'fro')/np.linalg.norm(identity_of_size_U, 'fro')
		print("U orthogonality error %g" %(U_orth_error))
		assert U_orth_error < fro_tol 

		# Check that MU is computed correctly 
		MU_test_np = self.M_csr @ decoder
		MU_error = np.linalg.norm(MU_test_np - encoder, 'fro')/np.linalg.norm(encoder, 'fro')
		print("MU error %g" %(MU_error))
		assert MU_error < fro_tol 

	def _check_shift(self, u_shift, is_shifted):
		"""
		Check that shift is being computed correctly 

		 - If :code:`is_shifted` is :code:`True`, :code:`u_shift` should be nonzero
		 - If :code:`is_shifted` is :code:`False`, :code:`u_shift` should be zeroes
		"""
		if is_shifted:
			assert not np.allclose(u_shift, 0)
		else:
			assert np.allclose(u_shift, 0)

	def _check_eigenvalue_problem(self, u_data, d, decoder, encoder, u_shift):
		"""
		Check that POD satisfies the eigenvalue problem 
		:math:`\mathbb{E}[(u - u_{s}) (u - u_{s})^T M] \phi = \lambda \phi`
		"""

		rel_tol = 1e-2
		# Check for zero shift 
		n_data = u_data.shape[0]
		shifted_data = u_data - u_shift 
		data_covariance = shifted_data.T @ shifted_data / n_data 
		CMU_np = data_covariance @ self.M_csr @ decoder

		u_fun = dl.Function(self.Vh[hp.STATE])
 
		print("POD with shift") 
		for i in range(d.shape[0]):
			diff = CMU_np[:,i] - d[i] * decoder[:,i]
			rel_error = np.linalg.norm(diff)/np.linalg.norm(decoder[:,i] * d[i]) 
			print("CM Eigenvector %d relative error, %g" %(i, rel_error))
			assert rel_error < rel_tol
 
		# 	plt.figure()
		# 	u_fun.vector().set_local(decoder[:,i])
		# 	hp.nb.plot(u_fun, mytitle="Output: phi %d" %(i))
 
		# 	plt.figure()
		# 	u_fun.vector().set_local(CMU_np[:,i])
		# 	hp.nb.plot(u_fun, mytitle="Output: C M phi %d" %(i))
		# plt.show()

if __name__ == "__main__":
	unittest.main()




