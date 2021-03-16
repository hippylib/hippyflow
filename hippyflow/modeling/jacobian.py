# Copyright (c) 2020, The University of Texas at Austin 
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


from hippylib import *
import dolfin as dl
import numpy as np
       

class Jacobian:
    """
    This class implements matrix free application of the Jacobian operator.
    The constructor takes the following parameters:

    - :code:`model`:               the object which contains the description of the problem.
    
    Type :code:`help(modelTemplate)` for more information on which methods model should implement.
    """
    def init_vector(self, x, dim):
        """
        Reshape the Vector :code:`x` so that it is compatible with the reduced Hessian
        operator.

        Parameters:

        - :code:`x`: the vector to reshape.
        - :code:`dim`: if 0 then :code:`x` will be made compatible with the range of the Jacobian, if 1 then :code:`x` will be made compatible with the domain of the Jacobian.
               
         """
        raise NotImplementedError

    
    def mpi_comm(self):
        """
        Return the mesh constructor mpi communicator 
        """
        raise NotImplementedError

        
    def mult(self,x,y):
        """
        Apply the Jacobian :code:`x`. Return the result in :code:`y`.
        """
        raise NotImplementedError

    def transpmult(self,x,y):
        """
        Apply the Jacobian :code:`x`. Return the result in :code:`y`.
        """
        raise NotImplementedError

class ObservableJacobian:
    """
    This class implements matrix free application of the Jacobian operator.
    The constructor takes the following parameters:

    - :code:`model`:               the object which contains the description of the problem.
    
    Type :code:`help(modelTemplate)` for more information on which methods model should implement.
    """
    def __init__(self, observable):
        """
        Construct the Observable Jacobian operator
        """
        self.observable = observable

        self.ncalls = 0
        
        self.rhs_fwd = observable.generate_vector(STATE)
        self.rhs_adj = observable.generate_vector(ADJOINT)
        self.rhs_adj2 = observable.generate_vector(ADJOINT)
        self.uhat    = observable.generate_vector(STATE)
        self.phat    = observable.generate_vector(ADJOINT)
        self.yhelp = observable.generate_vector(PARAMETER)



        self.Bu = dl.Vector(self.mpi_comm())
        self.observable.B.init_vector(self.Bu,0)
        self.Ctphat = observable.generate_vector(PARAMETER)
        self.shape = (self.Bu.get_local().shape[0],self.yhelp.get_local().shape[0])

    def mpi_comm(self):
        return self.observable.B.mpi_comm()
    
    def init_vector(self, x, dim):
        """
        Reshape the Vector :code:`x` so that it is compatible with the reduced Hessian
        operator.

        Parameters:

        - :code:`x`: the vector to reshape.
        - :code:`dim`: if 0 then :code:`x` will be made compatible with the range of the Jacobian, if 1 then :code:`x` will be made compatible with the domain of the Jacobian.
               
         """
        if dim == 0:
            self.observable.init_vector(x,0)
        elif dim == 1:
            self.observable.init_vector(x,1)
            #If the prior term shows up then the input dimension changes due to quadrature workaround
            # self.model.prior.sqrtM.init_vector(x,1)
        else: 
            raise

        
    def mult(self,x,y):
        """
        Apply the Jacobian :code:`x`. Return the result in :code:`y`.
        Implemented for dl.Vector
        """
        self.observable.applyC(x, self.rhs_fwd)
        self.observable.solveFwdIncremental(self.uhat, self.rhs_fwd)
        assert hasattr(self.observable,'applyB'), 'LinearObservable must have attribute applyB'
        self.observable.applyB(self.uhat,y)

        self.ncalls += 1

    def transpmult(self,x,y):
        """
        Apply the Jacobian transpose :code:`x`. Return the result in :code:`y`.
        Implemented for dl.Vector
        """
        assert hasattr(self.observable,'applyBt'), 'LinearObservable must have attribute applyBt'
        self.observable.applyBt(x,self.rhs_adj)
        self.observable.solveAdjIncremental(self.phat, self.rhs_adj)
        self.observable.applyCt(self.phat, y)
        self.ncalls += 1


class JTJ:
    """
    This class implements the operator :math:`J^TJ` given a Jacobian :math:`J`
    """
    def __init__(self,J):
        """
        Constructor
            - :code:`J` - Jacobian object, assumed to be of of type :code:`hippyflow.modeling.Jacobian`
        """
        self.J = J
        self.vector_help = dl.Vector(self.J.mpi_comm())
        self.J.init_vector(self.vector_help,0)

    def mult(self,x,y):
        """
        Compute :math:`y = J^TJ x `
        """
        self.J.mult(x,self.vector_help)
        self.J.transpmult(self.vector_help,y)

    def init_vector(self,x,dim=None):
        """
        Initialize :code:`x` to be compatible with the range (:code:`dim=0`) or domain (:code:`dim=1`) of :code:`JTJ`.
        """
        self.J.init_vector(x,1)


class JJT:
    """
    This class implements the operator :math:`JJ^T` given a Jacobian :math:`J`
    """
    def __init__(self,J):
        """
        Constructor
            - :code:`J` - Jacobian object, assumed to be of of type :code:`hippyflow.modeling.Jacobian`
        """
        self.J = J
        self.vector_help = dl.Vector(self.J.mpi_comm())
        self.J.init_vector(self.vector_help,1)

    def mult(self,x,y):
        """
        Compute :math:`y = JJ^T x `
        """
        self.J.transpmult(x,self.vector_help)
        self.J.mult(self.vector_help,y)

    def init_vector(self,x,dim = None):
        """
        Initialize :code:`x` to be compatible with the range (:code:`dim=0`) or domain (:code:`dim=1`) of :code:`JJ^`.
        """
        self.J.init_vector(x,0)








