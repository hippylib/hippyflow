import hippylib as hp
import dolfin as dl 
import numpy as np 

from .observable import LinearStateObservable

class StateSpaceIdentityOperator:
    """
    This class defines an identity operator on the state space
    """
    def __init__(self, M):
        """
        Constructor:
            :code: `M`: mass matrix of the state function space 
        """
        self.M = M
    
    def mpi_comm(self):
        return self.M.mpi_comm()

    def init_vector(self, v, dim):
        return self.M.init_vector(v, dim)

    def mult(self, u, y):
        y.zero()
        y.axpy(1.0, u)

    def transpmult(self, x, p):
        p.zero()
        p.axpy(1.0, x)

    def adjmult(self, x, p):
        self.M.transpmult(x, p)

class StateSpaceObservable(LinearStateObservable):
    """

    """
	def __init__(self, problem, B):
		"""
		Create a model given:
			- problem: the description of the forward/adjoint problem and all the sensitivities
			- B: the state space observation operator with method `transpmult` and `adjmult` 
			- prior: the prior 
		"""
        super().__init__(problem, B)
    
    def applyBt(self, x, out, operation="adjoint"):
        if operation == "adjoint":
            self.B.adjmult(x, out)
        else:
            self.B.transpmult(x, out)
