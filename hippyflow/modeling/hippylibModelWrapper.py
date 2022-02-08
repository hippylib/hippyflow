
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
from mpi4py import MPI 
import time
import warnings

import hippylib as hl

from .observable import hippylibModelLinearStateObservable
from .jacobian import ObservableJacobian
from ..utilities.mv_utilities import mv_to_dense, dense_to_mv_local

def hippylibModelWrapperParameterList():
	"""
	This function implements a parameter list for the hippylibModelWrapper
	"""

	parameters = {}
	return hl.ParameterList(parameters)



class hippylibModelWrapper:
	"""
	This class construcst a linear state observable from
	hIPPYlib.modeling.model.Model attributes

	"""
	def __init__(self,model,settings = hippylibModelWrapperParameterList()):
		"""
		"""
		warnings.warn('Experimental Class! Be Wary')
		self.model = model
		self.settings = settings
		assert hasattr(model,'misfit')

		self.observable = hippylibModelLinearStateObservable(model)

		self.J = ObservableJacobian(self.observable)




	def evalObs(self, m, u0 = None,setLinearizationPoint = False):
		"""
		Piggybacking on the observable evaluate the map from
		m to q(m)
		"""
		return self.observable.eval(m,u0=u0,setLinearizationPoint=setLinearizationPoint)


	def evalMisfit(self,m,u0 = None,setLinearizationPoint = False):
		"""
		"""
		assert hasattr(self.model.misfit,'d')
		Bu = self.observable.eval(m,u0=u0,setLinearizationPoint=setLinearizationPoint)
		Bu.axpy(-1.,self.model.misfit.d)
		return (1./self.model.misfit.noise_variance)*Bu

	def setLinearizationPoint(self, x):
		"""
		Specify the point :code:`x = [u,m,p]` at which the Hessian operator (or the Gauss-Newton approximation)
		needs to be evaluated.
		Parameters:
			- :code:`x = [u,m,p]`: the point at which the Hessian or its Gauss-Newton approximation needs to be evaluated.
		.. note:: This routine should either:
			- simply store a copy of x and evaluate action of blocks of the Hessian on the fly
			- or partially precompute the block of the hessian (if feasible)
		"""
		x[hl.ADJOINT] = self.model.generate_vector(hl.ADJOINT)
		self.problem.setLinearizationPoint(x, True)


	def evalVariationalGradient(self,x,u0 = None,setLinearizationPoint = False,misfit_only = True):
		"""
		Assumes  x = [u(m), m, p(m,u(m))] correspond to state solution at m,
		the parameter m, and the adjoint solution at m, u(m).
		If u, or p are None then this method will solve and optional initial guess
		for a nonlinear PDE solver iteration can be passed in
		"""
		u,m,p = x
		assert m is not None

		if u is None:
			if u0 is None:
				u0 = self.model.generate_vector(hl.STATE)
			self.model.solveFwd(u0,[u0,m,None])
			u = u0
			if p is None:
				p = self.model.generate_vector(hl.ADJOINT)
			self.model.solveAdj(p,[u,m,p])

		elif p is None:
			p = self.model.generate_vector(hl.ADJOINT)
			self.model.solveAdj(p,[u,m,p])

		if setLinearizationPoint:
			self.setLinearizationPoint([u,m,None],True)
		x = [u,m,p]
		mg = self.model.generate_vector(hl.PARAMETER)

		tmp = self.model.generate_vector(hl.PARAMETER)
		self.model.problem.evalGradientParameter(x, mg)
		self.model.misfit.grad(hl.PARAMETER,x,tmp)
		mg.axpy(1., tmp)
		if not misfit_only:
			self.model.prior.grad(x[hl.PARAMETER], tmp)
			mg.axpy(1., tmp)

		return mg

	def evalGradient(self,x,u0 = None,setLinearizationPoint = False,misfit_only = True,\
						invert_regularization = False):
		"""
		if invert regularization is false then a mass matrix will be inverted instead
		"""

		mg = self.evalVariationalGradient(x,u0=u0,setLinearizationPoint=setLinearizationPoint,misfit_only=misfit_only)
		mhat = self.model.generate_vector(hl.PARAMETER)
		if invert_regularization:
			self.invertRegularization(mhat,mg)
		else:
			self.invertMassMatrix(mhat,mg)

	def invertMassMatrix(self,out,rhs):
		"""
		"""
		self.model.prior.Msolver.solve(out,rhs)

	def invertRegularization(self,out,rhs):
		"""
		"""
		self.model.prior.Rsolver.solve(out,rhs)


	def evalLowRankJacobian(self,x,rank,u0 = None, linearizationPointSet = False,\
							randomized = True,over_sample = 5,power_iteration = 1):
		"""
		"""
		u,m,_ = x
		assert m is not None
		assert type(rank) is int
		if linearizationPointSet:
			# Assumes that solution at this point is passed in
			assert x[hl.STATE] is not None
		else:
			if u is None:
				if u0 is None:
					u0 = self.model.generate_vector(hl.STATE)
				self.model.solveFwd(u0,[u0,m,None])
				u = u0
			x = [u,m,None]
			self.observable.setLinearizationPoint(x)
		m_constructor = self.model.generate_vector(hl.PARAMETER)
		# Sample random matrix
		Omega = hl.MultiVector(m_constructor, rank + over_sample)
		hl.parRandom.normal(1.,Omega)
		U,sigma,V = hl.accuracyEnhancedSVD(self.J,Omega,rank,s=power_iteration)

		return U,sigma,V












