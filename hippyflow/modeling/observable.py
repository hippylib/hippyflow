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

import hippylib as hp 
import dolfin as dl
import numpy as np

CONTROL = 3


class Observable:
	"""
	This class implements a generic observable.

	In the following we will denote with
		- :code:`u` the state variable
		- :code:`m` the (model) parameter variable
		- :code:`p` the adjoint variable
		
	"""

	def eval(self):
		pass

	def setLinearizationPoint(self):
		pass

class DomainRestrictedOperator:
    """
    This class defines a linear operator that zeros out fields in the state vector.
    """
    def __init__(self, indicator_vec, B):
        """
        Constructor:
            :code:`indicator_vec`: vector that allows you to select what part of the state you want to zero out\
            when working with a mixed problem (or a problem whose state has multiple fields). 
            :code:`B` is a PETSc matrix that projects the state onto the location of observations.
        """
        self.indicator_vec = indicator_vec
        self.B = B
    
    def mpi_comm(self):
        return self.B.mpi_comm()
    
    def init_vector(self, v, dim):
        return self.B.init_vector(v, dim)

    def mult(self, u, y):
        self.B.mult(u*self.indicator_vec, y)

    def transpmult(self, x, p):
        self.B.transpmult(x, p)
        p *= self.indicator_vec
      

class LinearStateObservable:
	"""
		
	"""
	
	def __init__(self, problem, B):
		"""
		Create a model given:
			- problem: the description of the forward/adjoint problem and all the sensitivities
			- B: the pointwise observation operator
			- prior: the prior 
		"""
		self.problem = problem
		self.is_control_problem = hasattr(self.problem,'Cz')
		self.B = B
		# self.Bu = dl.Vector(self.B.mpi_comm())
		# self.B.init_vector(self.Bu,0)
		
		self.n_fwd_solve = 0
		self.n_adj_solve = 0
		self.n_inc_solve = 0
				
	def mpi_comm(self):
		return self.B.mpi_comm()


	def generate_vector(self, component = "ALL"):
		"""
		By default, return the list :code:`[u,m,p]` where:
		
			- :code:`u` is any object that describes the state variable
			- :code:`m` is a :code:`dolfin.Vector` object that describes the parameter variable. \
			(Needs to support linear algebra operations)
			- :code:`p` is any object that describes the adjoint variable
		
		If :code:`component = STATE` return only :code:`u`
			
		If :code:`component = PARAMETER` return only :code:`m`
			
		If :code:`component = ADJOINT` return only :code:`p`
		""" 
		if component == "ALL":
			if hasattr(self.problem,'Cz'):
				# Assumes that problem is a control problem
				x = [self.problem.generate_state(),
					 self.problem.generate_parameter(),
					 self.problem.generate_state(),
					 self.problem.generate_control()]
			else:
				x = [self.problem.generate_state(),
					 self.problem.generate_parameter(),
					 self.problem.generate_state()]
		elif component == hp.STATE:
			x = self.problem.generate_state()
		elif component == hp.PARAMETER:
			x = self.problem.generate_parameter()
		elif component == hp.ADJOINT:
			x = self.problem.generate_state()
		elif component == CONTROL:
			assert self.is_control_problem, 'Assuming it is a control problem'
			# 3 denotes a control variable needs to be generated
			x = self.problem.generate_control()

		return x

	def init_vector(self, x, dim):
		"""
		Reshape the Vector :code:`x` so that it is compatible with the reduced Hessian
		operator.

		Parameters:

		- :code:`x`: the vector to reshape.
		- :code:`dim`: if 0 then :code:`x` will be made compatible with the range of the Jacobian, if 1 then :code:`x` will be made compatible with the domain of the Jacobian.
			   
		 """
		if dim == 0:
			self.B.init_vector(x,0)
		elif dim == 1:
			# self.model.init_parameter(x)
			self.problem.C.init_vector(x,1)
		elif dim == 3:
			assert self.is_control_problem, 'Assuming it is a control problem'
			self.problem.Cz.init_vector(x,1)
		else: 
			raise

	def init_parameter(self, m):
		"""
		Reshape :code:`m` so that it is compatible with the parameter variable
		"""
		# Aha! I found the issue I think!!!!!!!!!
		# This is wrong since the STATE and PARAMETER dimension are not necessarily the same.

		self.problem.init_parameter(m)

			
	def eval(self, m, u0 = None, z = None, setLinearizationPoint = False):
		"""
		Given the input parameter :code:`m` solve for the state field $u(m)$, and evaluate 
		the linear state observable $Bu(m)$
		
		Return the linear state observable $Bu(m)
		
		"""
		if u0 is None:
			u0 = self.problem.generate_state()
		if self.is_control_problem:
			assert z is not None
			x = [u0,m,None,z]
		else:
			x = [u0, m, None]
		self.problem.solveFwd(u0,x)
		if setLinearizationPoint:
			self.setLinearizationPoint(x)
		out = dl.Vector(self.mpi_comm())
		self.B.init_vector(out,0)
		self.B.mult(u0,out)

		return out

	def evalu(self,u):
		"""
		Given a state field that is already solved for :code:`u`, evaluate the linear 
		state observable $Bu(m)$
		
		Return the linear state observable $Bu(m)
		
		"""
		out = dl.Vector(self.mpi_comm())
		self.B.init_vector(out,0)
		self.B.mult(u,out)
		return out
	
	def solveFwd(self, out, x):
		"""
		Solve the (possibly non-linear) forward problem.
		
		Parameters:
			- :code:`out`: is the solution of the forward problem (i.e. the state) (Output parameters)
			- :code:`x = [u,m,p]` provides
				1) the parameter variable :code:`m` for the solution of the forward problem
				2) the initial guess :code:`u` if the forward problem is non-linear
		
				.. note:: :code:`p` is not accessed.
		"""
		self.n_fwd_solve = self.n_fwd_solve + 1
		self.problem.solveFwd(out, x)
		
	
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
		x[hp.ADJOINT] = self.problem.generate_state()
		self.problem.setLinearizationPoint(x, True)

		
	def solveFwdIncremental(self, sol, rhs):
		"""
		Solve the linearized (incremental) forward problem for a given right-hand side
		Parameters:
			- :code:`sol` the solution of the linearized forward problem (Output)
			- :code:`rhs` the right hand side of the linear system
		"""
		self.n_inc_solve = self.n_inc_solve + 1
		self.problem.solveIncremental(sol,rhs, False)
		
	def solveAdjIncremental(self, sol, rhs):
		"""
		Solve the incremental adjoint problem for a given right-hand side
		Parameters:
			- :code:`sol` the solution of the incremental adjoint problem (Output)
			- :code:`rhs` the right hand side of the linear system
		"""
		self.n_inc_solve = self.n_inc_solve + 1
		self.problem.solveIncremental(sol,rhs, True)


	def applyB(self,x,out):
		self.B.mult(x,out)
		

	def applyBt(self,x,out):
		
		self.B.transpmult(x,out)
	
	def applyC(self, dm, out):
		"""

		Apply the :math:`C` block of the Hessian to a (incremental) parameter variable, i.e.
		:code:`out` = :math:`C dm`
		
		Parameters:
			- :code:`dm` the (incremental) parameter variable
			- :code:`out` the action of the :math:`C` block on :code:`dm`
			
		.. note:: This routine assumes that :code:`out` has the correct shape.
		"""
		self.problem.apply_ij(hp.ADJOINT,hp.PARAMETER, dm, out)
	
	def applyCt(self, dp, out):
		"""
		Apply the transpose of the :math:`C` block of the Hessian to a (incremental) adjoint variable.
		:code:`out` = :math:`C^t dp`
		Parameters:
			- :code:`dp` the (incremental) adjoint variable
			- :code:`out` the action of the :math:`C^T` block on :code:`dp`
			
		..note:: This routine assumes that :code:`out` has the correct shape.
		"""
		self.problem.apply_ij(hp.PARAMETER,hp.ADJOINT, dp, out)

	def applyCz(self, dz, out):
		"""

		Apply the :math:`C` block of the (control problem) Hessian to a (incremental) control variable, i.e.
		:code:`out` = :math:`C_z dz`
		
		Parameters:
			- :code:`dz` the (incremental) control variable
			- :code:`out` the action of the :math:`C_z` block on :code:`dz`
			
		.. note:: This routine assumes that :code:`out` has the correct shape.
		"""
		self.problem.apply_ij(hp.ADJOINT,CONTROL, dz, out)
	
	def applyCzt(self, dp, out):
		"""
		Apply the transpose of the :math:`C_z` block of the (control) Hessian to a (incremental) adjoint variable.
		:code:`out` = :math:`C_z^t dp`
		Parameters:
			- :code:`dp` the (incremental) adjoint variable
			- :code:`out` the action of the :math:`C_z^T` block on :code:`dp`
			
		..note:: This routine assumes that :code:`out` has the correct shape.
		"""
		self.problem.apply_ij(CONTROL,hp.ADJOINT, dp, out)

def hippylibModelLinearStateObservable(model):
	"""
	This function construcst a linear state observable from
	hIPPYlib.modeling.model.Model attributes
	Parameters:
		- :code:`model` represents the hippylib mode
	"""
	assert hasattr(model,'misfit')
	return LinearStateObservable(model.problem,model.misfit.B)


