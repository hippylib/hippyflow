# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.


import dolfin as dl
from hippylib import *
from .blockVector import BlockVector



class MultiPDEProblem(PDEProblem):
    def __init__(self, problems):
        self.Vh = problems[0].Vh
        self.problems = problems
        self.n_problems = len(problems)
                
    def generate_state(self):
        """ 
        Return a vector in the shape of the state 
        """
        return BlockVector(self.Vh[STATE], self.n_problems)
        
    def generate_parameter(self):
        return self.problems[0].generate_parameter()
    
    def init_parameter(self, m):
        """ 
        Initialize the parameter 
        """
        dummy = self.generate_parameter()
        m.init( dummy.mpi_comm(), dummy.local_range() )
    
    def solveFwd(self, state, x):
        """ 
        Solve the possibly nonlinear forward problem:

        Given :math:`m`, find :math:`u` such that
        
        .. math::
            \delta_p F(u, m, p; \hat{p}) = 0, \quad \\forall \hat{p}.

        """
        
        u, m, p = x
        
        if p is None:
            p = u.copy()
        
        for k in range(self.n_problems):
            self.problems[k].solveFwd(state.data[k], [u.data[k], m, p.data[k] ])
        
    def solveAdj(self, adj, x, adj_rhs):
        """ 
        Solve the linear adjoint problem: 
        Given :math:`m, u` find :math:`p` such that

        .. math::
            \delta_u F(u, m, p; \hat{u}) = 0, \quad \\forall \hat{u}

        """
        u, m, p = x
        for k in range(self.n_problems):
            self.problems[k].solveAdj(adj.data[k], [u.data[k], m, p.data[k] ], adj_rhs.data[k])
     
    def evalGradientParameter(self, x, out):
        """
        Given :math:`u,m,p` evaluate :math:`\delta_m F(u, m, p; \hat{m}),\: \\forall \hat{m}` 
        """
        tmp = self.generate_parameter()
        u, m, p = x
        out.zero()
        for k in range(self.n_problems):
            self.problems[k].evalGradientParameter([ u.data[k], m, p.data[k] ], tmp)
            out.axpy(1.,tmp)
         
    def setLinearizationPoint(self,x, gn_approx):
        """ 
        Set the values of the state and parameter for the incremental forward 
        and adjoint solvers 
        """
        u, m, p = x
        for k in range(self.n_problems):
            self.problems[k].setLinearizationPoint([ u.data[k], m, p.data[k] ], gn_approx)
                
    def solveIncremental(self, out, rhs, is_adj):
        """ 
        If :code:`is_adj = False`, solve the forward incremental system:
            
        Given :math:`u, m`, find :math:`\\tilde{u}` s.t.:
        
        .. math::
            \delta_{pu} F(u, m, p; \hat{p}, \\tilde{u}) = \mbox{rhs}, \quad \\forall \hat{p}.
        
        If :code:`is_adj = True`, solve the adjoint incremental system:
        
        Given :math:`u, m`, find :math:`\\tilde{p}` s.t.:
        
        .. math::
            \delta_{up} F(u, m, p; \hat{u}, \\tilde{p}) = \mbox{rhs}, \quad \\forall \hat{u}.

        """
        for k in range(self.n_problems):
            self.problems[k].solveIncremental(out.data[k], rhs.data[k], is_adj)

    def apply_ij(self,i,j, dir, out):   
        """
        Given :math:`u, m, p`; compute 
        
        .. math::
            \delta_{ij} F(u, m, p; \hat{i}, \\tilde{j}) \mbox{ in the direction } \\tilde{j} = \mbox{dir} \quad \\forall \hat{i}

        """
        out.zero()
        if i == PARAMETER:
            tmp = self.generate_parameter()
            if j == PARAMETER:
                for k in range(self.n_problems):
                    self.problems[k].apply_ij(i,j, dir,tmp)
                    out.axpy(1., tmp)
            else:
                for k in range(self.n_problems):
                    self.problems[k].apply_ij(i,j, dir.data[k],tmp)
                    out.axpy(1., tmp)
        else:
            assert type(out) is BlockVector
            if j == PARAMETER:
                for k in range(self.n_problems):
                    self.problems[k].apply_ij(i,j, dir,out.data[k] )
            else:
                for k in range(self.n_problems):
                    self.problems[k].apply_ij(i,j, dir.data[k],out.data[k])