# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib-ice library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib-ice is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.
import sys, os
sys.path.append( os.environ.get('HIPPYLIB_PATH'))

import dolfin as dl
import hippylib as hp

sys.path.append(os.environ.get('HIPPYFLOW_PATH'))
from hippyflow import *

class NonlinearStokesForm:
    def __init__(self, n, A, normal, ds_base, f, lam=0.):
        
        # Rheology
        self.n      = n
        self.A      = A
        
        # Basal Boundary
        self.normal = normal
        self.ds_base = ds_base
        
        # Forcing term
        self.f      = f
        
        # Smooth strain
        self.eps    = dl.Constant(1e-6)
        
        # penalty parameter for Dirichlet condition
        self.lam = dl.Constant(0.5*lam)
        
    def _epsilon(self, velocity):
        return dl.sym( dl.grad(velocity) )
    
    def _tang(self, velocity):
        return (velocity - dl.outer(self.normal, self.normal)*velocity)
        
    def energy_fun(self, u, m):
        
        velocity, p = dl.split(u)
        normEu12   = 0.5*dl.inner(self._epsilon(velocity),self._epsilon(velocity)) + self.eps
        
        return self.A**(-1./self.n)*((2.*self.n)/(1.+self.n))*(normEu12**((1. + self.n)/(2.*self.n)))*dl.dx \
               - dl.inner(self.f,velocity)*dl.dx \
               + dl.Constant(.5)*dl.inner(dl.exp(m)*self._tang(velocity), self._tang(velocity)) * self.ds_base \
               + self.lam * dl.inner(velocity, self.normal)**2 *self.ds_base

    
    def constraint(self, u):
        vel, pressure = dl.split(u)
        return dl.inner(-dl.div(vel), pressure)*dl.dx
    
    def varf_handler(self,u,m,p):
        return dl.derivative( self.energy_fun(u, m) + self.constraint(u), u, p) + self.constraint(u)

class EnergyFunctionalPDEVariationalProblem(hp.PDEVariationalProblem):
    def __init__(self, Vh, energy, constraint_vec, bc, bc0):
        """
        Initialize the EnergyFunctionalPDEVariationalProblem.
        """
        hp.PDEVariationalProblem.__init__(self, Vh, energy.varf_handler, bc, bc0)

        #variables needed for custom nonlinear newton solver
        self.energy_fun = energy.energy_fun
        self.constraint = energy.constraint
        self.constraint_vec = constraint_vec
        self.it = 0
        
        self.fwd_solver = ConstrainedNSolver()

    def solveFwd(self, state, x):
        """ Solve the nonlinear forward problem using Newton's method."""
        
        if self.solver is None:
            self.solver = self._createLUSolver()
            
        u = hp.vector2Function(x[hp.STATE], self.Vh[hp.STATE])
        m = hp.vector2Function(x[hp.PARAMETER], self.Vh[hp.PARAMETER])
        # p = dl.TestFunction(self.Vh[hp.ADJOINT])
        # w = dl.TrialFunction(self.Vh[hp.STATE])
        
        C = self.constraint(u)
        F = self.energy_fun(u, m)
        Fn = dl.assemble(F)
        # print('Fn = ',Fn,' in solveFwd')
                
        if self.fwd_solver.solver is None:
            self.fwd_solver.solver = self.solver
            
        uvec, niter = self.fwd_solver.solve(F, C, u, self.constraint_vec, self.bc, self.bc0)
        state.zero()
        state.axpy(1., uvec.vector())
        
        self.it += niter
        
    def export2XDMF(self, x, fid):
        fid.parameters["functions_share_mesh"] = True
        fid.parameters["rewrite_function_mesh"] = False
        
        ufun = hp.vector2Function(x[hp.STATE], self.Vh[hp.STATE], name='State')
        mfun = hp.vector2Function(x[hp.PARAMETER], self.Vh[hp.PARAMETER], name='Parameter')
        
        vel, press  = ufun.split(deepcopy=True)
        
        fid.write(vel, 0)
        fid.write(press, 0)
        fid.write(mfun, 0)
        
