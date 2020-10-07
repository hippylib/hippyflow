from hippylib import *
import dolfin as dl
import numpy as np


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
		self.B = B
		self.Bu = dl.Vector(self.B.mpi_comm())
		self.B.init_vector(self.Bu,0)
		
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
			x = [self.problem.generate_state(),
				 self.problem.generate_parameter(),
				 self.problem.generate_state()]
		elif component == STATE:
			x = self.problem.generate_state()
		elif component == PARAMETER:
			x = self.problem.generate_parameter()
		elif component == ADJOINT:
			x = self.problem.generate_state()

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
		else: 
			raise

	def init_parameter(self, m):
		"""
		Reshape :code:`m` so that it is compatible with the parameter variable
		"""
		# Aha! I found the issue I think!!!!!!!!!
		# This is wrong since the STATE and PARAMETER dimension are not necessarily the same.

		self.problem.init_parameter(m)

			
	def eval(self, m, u0 = None,setLinearizationPoint = False):
		"""
		Given the input parameter :code:`m` solve for the state field $u(m)$, and evaluate 
		the linear state observable $Bu(m)$
		
		Return the linear state observable $Bu(m)
		
		"""
		if u0 is None:
			u0 = self.problem.generate_state()
		x = [u0, m, None]
		self.problem.solveFwd(u0,x)
		if setLinearizationPoint:
			self.setLinearizationPoint(x)
		out = dl.Vector()
		self.B.init_vector(out,0)
		self.B.mult(u0,out)

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
		x[ADJOINT] = self.problem.generate_state()
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
		self.problem.apply_ij(ADJOINT,PARAMETER, dm, out)
	
	def applyCt(self, dp, out):
		"""
		Apply the transpose of the :math:`C` block of the Hessian to a (incremental) adjoint variable.
		:code:`out` = :math:`C^t dp`
		Parameters:
			- :code:`dp` the (incremental) adjoint variable
			- :code:`out` the action of the :math:`C^T` block on :code:`dp`
			
		..note:: This routine assumes that :code:`out` has the correct shape.
		"""
		self.problem.apply_ij(PARAMETER,ADJOINT, dp, out)
