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
import hippylib as hp
from .jacobian import Jacobian

CONTROL = 3

class ObservableControlJacobian(Jacobian):
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

        assert hasattr(self.observable,'applyCz')
        assert hasattr(self.observable,'applyCzt')

        self.ncalls = 0
        
        self.rhs_fwd = observable.generate_vector(hp.STATE)
        self.rhs_adj = observable.generate_vector(hp.ADJOINT)
        self.rhs_adj2 = observable.generate_vector(hp.ADJOINT)
        self.uhat    = observable.generate_vector(hp.STATE)
        self.phat    = observable.generate_vector(hp.ADJOINT)
        self.yhelp = observable.generate_vector(CONTROL)

        self.Bu = dl.Vector(self.mpi_comm())
        self.observable.B.init_vector(self.Bu,0)
        self.Ctphat = observable.generate_vector(CONTROL)
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
            # CONTROL = 3
            self.observable.init_vector(x,3)


        
    def mult(self,x,y):
        """
        Apply the Jacobian :code:`x`. Return the result in :code:`y`.
        Implemented for dl.Vector
        """
        self.observable.applyCz(x, self.rhs_fwd)
        self.observable.solveFwdIncremental(self.uhat, self.rhs_fwd)
        assert hasattr(self.observable,'applyB'), 'LinearObservable must have attribute applyB'
        self.observable.applyB(self.uhat,y)
        y *= -1.0
        self.ncalls += 1

    def transpmult(self,x,y):
        """
        Apply the Jacobian transpose :code:`x`. Return the result in :code:`y`.
        Implemented for dl.Vector
        """
        assert hasattr(self.observable,'applyBt'), 'LinearObservable must have attribute applyBt'
        self.observable.applyBt(x,self.rhs_adj)
        self.observable.solveAdjIncremental(self.phat, self.rhs_adj)
        self.observable.applyCzt(self.phat, y)
        y *= -1.0
        self.ncalls += 1
