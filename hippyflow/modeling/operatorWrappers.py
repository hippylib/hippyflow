
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

class npToDolfinOperator:

	def __init__(self,npArray):
		"""
		"""
		assert len(npArray.shape) == 2
		self.matrix = npArray

		self.domain_help = None 
		self.range_help = None
		# assert that the dolfin mpi world size is one
		pass

	def init_vector(self,x,dim):
		"""
		"""
		if dim == 0:
			x.init(self.matrix.shape[0])
		elif dim == 1:
			x.init(self.matrix.shape[1])
		else:
			raise

	def mult(self,x,y):
		"""
		"""
		y.zero()
		y.set_local(self.matrix@x.get_local())

	def transpmult(self,x,y):
		"""
		"""
		y.zero()
		y.set_local(self.matrix.T@x.get_local())


class MeanJTJfromDataOperator:
	"""
	"""
	def __init__(self, J, prior, noise_cov_inv=None):
		"""
		"""

		# Assumes J.shape = (ndata,rank,dM)
		self._J = J
		self.ndata, self.r, self.dM = self.J.shape
		self._prior = prior

		if noise_cov_inv is not None:
			assert hasattr(noise_cov_inv, '__mult__')
		self._noise_cov_inv = noise_cov_inv

		if hasattr(self.prior, "R"):
			self.init_vector_lambda = lambda x,dim: prior.R.init_vector(x,dim)
		else:
			self.init_vector_lambda = lambda x,dim: prior.Hlr.init_vector(x,dim)

	@property
	def J(self):
		return self._J

	@property
	def prior(self):
		return self._prior
	
	@property
	def noise_cov_inv(self):
		return self._noise_cov_inv
	
	def init_vector(self,x,dim):
		"""
		Initialize :code:`x` to be compatible with the range (:code:`dim=0`) or domain (:code:`dim=1`) of :code:`A`.
		"""
		assert init_vector_lambda is not None
		self.init_vector_lambda(x,dim)

	def mult(self,x,y):
		"""
		Compute :math:`y = mean(JTJ)x `
		"""
		x_np = x.get_local()
		# print('x_np.shape = ',x_np.shape)
		X_np = np.tile(x_np,(self.ndata,1))
		# print('X_np.shape = ',X_np.shape)
		assert X_np.shape == (self.ndata,self.dM)
		JX_np = np.einsum('ijk,ik->ij',self.J,X_np)

		# compute with noise covariance, if present
		if self.noise_cov_inv is not None:
			JX_np = self.noise_cov_inv @ JX_np

		# print('PhiTJX_np.shape = ',JX_np.shape)
		JTJX_np = np.einsum('ijk,ij->ik',self.J,JX_np)
		# print('JTPhiPhiTJX_np.shape = ',JTJX_np.shape)
		y.set_local(np.mean(JTJX_np,axis = 0))

	def transpmult(self,x,y):
		"""
		Compute :math:`y = mean(JTJ)x `
		JTJ is naturally self-adjoint so tranpsmult is mult.
		"""
		return self.mult(x,y)



		