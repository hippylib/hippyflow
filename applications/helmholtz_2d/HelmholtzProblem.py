
# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2021, The University of Texas at Austin 
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
import sys, os
sys.path.append( os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp

class PML:
    def __init__(self, mesh, box, box_pml, A):
        
        t = [None]*4
        
        for i in range(4):
            t[i] = box_pml[i]-box[i]
            if abs(t[i]) < dl.DOLFIN_EPS:
                t[i]= 1.
            
        self.sigma_x = dl.Expression("(x[0]<xL)*A*(x[0]-xL)*(x[0]-xL)/(tL*tL) + (x[0]>xR)*A*(x[0]-xR)*(x[0]-xR)/(tR*tR)",
                            xL=box[0], xR=box[2], A=A, tL=t[0], tR=t[2], degree = 2 )
        
        self.sigma_y = dl.Expression("(x[1]<yB)*A*(x[1]-yB)*(x[1]-yB)/(tB*tB) + (x[1]>yT)*A*(x[1]-yT)*(x[1]-yT)/(tT*tT)",
                            yB=box[1], yT=box[3], A=A, tB=t[1], tT=t[3], degree = 2 )
        
        physical_domain = dl.AutoSubDomain(lambda x, on_boundary: x[0] >= box[0] \
                                                              and x[0] <= box[2] \
                                                              and x[1] >= box[1] \
                                                              and x[1] <= box[3] )
        
        cell_marker = dl.MeshFunction("size_t", mesh, mesh.geometry().dim())
        cell_marker.set_all(0)
        physical_domain.mark(cell_marker, 1)
        self.dx = dl.Measure("dx", subdomain_data=cell_marker)

class SingleSourceHelmholtzProblem(PDEProblem):
    def __init__(self, Vh, sources_loc, wave_number, pml):
        self.Vh = Vh
        self.wave_number = wave_number
        self.PML = pml
        
        self.rhs_fwd = self.generate_state()
        
        if type(sources_loc) is dl.Point:
            ps0 = dl.PointSource(self.Vh[hp.STATE].sub(0), sources_loc, 1.)
            ps0.apply(self.rhs_fwd)

        else:
            for source in sources_loc:
                ps0 = dl.PointSource(self.Vh[hp.STATE].sub(0), source, 1.)
                ps0.apply(self.rhs_fwd)

        self.A  = None
        self.At = None
        self.C = None
        self.Wmu = None
        self.Wmm = None
        self.Wuu = None
        
        self.solver = self._createLUSolver()
        self.solver_fwd_inc = self._createLUSolver()
        self.solver_adj_inc = self._createLUSolver()
        
    def varf_handler(self,u,m,p):
        
        k = self.wave_number*dl.exp(m)
        ksquared = k**2
        
        sigma_x = self.PML.sigma_x
        sigma_y = self.PML.sigma_y
        
        Kr = ksquared - sigma_x*sigma_y
        Ki = -k*(sigma_x + sigma_y)
        
        Dr_xx = (ksquared+sigma_x*sigma_y)/(ksquared + sigma_x*sigma_x)
        Dr_yy = (ksquared+sigma_x*sigma_y)/(ksquared + sigma_y*sigma_y)
        Di_xx = k*(sigma_x - sigma_y)/(ksquared + sigma_x*sigma_x)
        Di_yy = k*(sigma_y - sigma_x)/(ksquared + sigma_y*sigma_y)
        
        Dr = dl.as_matrix([[Dr_xx, dl.Constant(0.)], [dl.Constant(0.), Dr_yy]])
        Di = dl.as_matrix([[Di_xx, dl.Constant(0.)], [dl.Constant(0.), Di_yy]])
        
        u1, u2 = dl.split(u)
        p1, p2 = dl.split(p)
        
        form_r = dl.inner(dl.grad(u1), dl.grad(p1))*self.PML.dx(1) \
                 - ksquared*u1*p1*self.PML.dx(1)
                 
        form_i = -dl.inner(dl.grad(u2), dl.grad(p2))*self.PML.dx(1) \
                     + ksquared*u2*p2*self.PML.dx(1)

        
        form_pml_r = dl.inner(Dr*dl.grad(u1), dl.grad(p1))*self.PML.dx(0) \
                 + dl.inner(Di*dl.grad(u2), dl.grad(p1))*self.PML.dx(0) \
                 - Kr*u1*p1*self.PML.dx(0) \
                 - Ki*u2*p1*self.PML.dx(0)
       
        form_pml_i = - dl.inner(Dr*dl.grad(u2), dl.grad(p2))*self.PML.dx(0) \
               + dl.inner(Di*dl.grad(u1), dl.grad(p2))*self.PML.dx(0) \
               + Kr*u2*p2*self.PML.dx(0) \
               - Ki*u1*p2*self.PML.dx(0)
       
        return form_r + form_i + form_pml_r + form_pml_i
        
    def generate_state(self):
        """ return a vector in the shape of the state """
        return dl.Function(self.Vh[hp.STATE]).vector()
    
    def generate_parameter(self):
        """ return a vector in the shape of the parameter """
        return dl.Function(self.Vh[hp.PARAMETER]).vector()
    
    def init_parameter(self, m):
        """ initialize the parameter """
        dummy = self.generate_parameter()
        m.init( dummy.mpi_comm(), dummy.local_range() )
    
    def solveFwd(self, state, x):
        """ Solve the possibly nonlinear Fwd Problem:
        Given m, find u such that
        \delta_p F(u,m,p;\hat_p) = 0 \for all \hat_p"""
        
        u = dl.TrialFunction(self.Vh[hp.STATE])
        m = hp.vector2Function(x[hp.PARAMETER], self.Vh[hp.PARAMETER])
        p = dl.TestFunction(self.Vh[hp.ADJOINT])
        A = dl.assemble( self.varf_handler(u,m,p) )
        
        self.solver.set_operator(A)
        self.solver.solve(state, self.rhs_fwd)

        
    def solveAdj(self, adj, x, adj_rhs):
        """ Solve the linear Adj Problem: 
            Given m, u; find p such that
            \delta_u F(u,m,p;\hat_u) = 0 \for all \hat_u
        """
        u = dl.TestFunction(self.Vh[hp.STATE])
        m = hp.vector2Function(x[hp.PARAMETER], self.Vh[hp.PARAMETER])
        p = dl.TrialFunction(self.Vh[hp.ADJOINT])
        Aadj = dl.assemble( self.varf_handler(u,m,p) )

        self.solver.set_operator(Aadj)
        self.solver.solve(adj, adj_rhs)
     
    def evalGradientParameter(self, x, out):
        """Given u,m,p; eval \delta_m F(u,m,p; \hat_m) \for all \hat_m """
        u = hp.vector2Function(x[hp.STATE], self.Vh[hp.STATE])
        m = hp.vector2Function(x[hp.PARAMETER], self.Vh[hp.PARAMETER])
        p = hp.vector2Function(x[hp.ADJOINT], self.Vh[hp.ADJOINT])
        dm = dl.TestFunction(self.Vh[hp.PARAMETER])
        res_form = self.varf_handler(u,m,p)
        out.zero()
        dl.assemble( dl.derivative(res_form, m, dm), tensor=out)
         
    def setLinearizationPoint(self,x, gn_approx):
        """ Set the values of the state and parameter
            for the incremental Fwd and Adj solvers """
        u = hp.vector2Function(x[hp.STATE], self.Vh[hp.STATE])
        m = hp.vector2Function(x[hp.PARAMETER], self.Vh[hp.PARAMETER])
        p = hp.vector2Function(x[hp.ADJOINT], self.Vh[hp.ADJOINT])
        x_fun = [u,m,p]
        
        f_form = self.varf_handler(u,m,p)
        
        g_form = [None,None,None]
        for i in range(3):
            g_form[i] = dl.derivative(f_form, x_fun[i])
            
        self.A = dl.assemble(dl.derivative(g_form[hp.ADJOINT],u))
        self.At = dl.assemble(dl.derivative(g_form[hp.STATE],p))
        self.C = dl.assemble(dl.derivative(g_form[hp.ADJOINT],m))
        
        self.solver_fwd_inc.set_operator(self.A)
        self.solver_adj_inc.set_operator(self.At)
        
        if gn_approx:
            self.Wmu = None
            self.Wuu = None
            self.Wmm = None
        else:
            self.Wmu = dl.assemble(dl.derivative(g_form[hp.PARAMETER],u))
            self.Wuu = dl.assemble(dl.derivative(g_form[hp.STATE],u))
            self.Wmm = dl.assemble(dl.derivative(g_form[hp.PARAMETER],m))
        
                
    def solveIncremental(self, out, rhs, is_adj):
        """ If is_adj = False:
            Solve the forward incremental system:
            Given u, m, find \tilde_u s.t.:
            \delta_{pu} F(u,m,p; \hat_p, \tilde_u) = rhs for all \hat_p.
            
            If is_adj = True:
            Solve the adj incremental system:
            Given u, m, find \tilde_p s.t.:
            \delta_{up} F(u,m,p; \hat_u, \tilde_p) = rhs for all \delta_u.
        """
        if is_adj:
            self.solver_adj_inc.solve(out, rhs)
        else:
            self.solver_fwd_inc.solve(out,rhs)
            

    
    def apply_ij(self,i,j, dir, out):   
        """
            Given u, m, p; compute 
            \delta_{ij} F(u,a,p; \hat_i, \tilde_j) in the direction \tilde_j = dir for all \hat_i
        """
        KKT = {}
        KKT[hp.STATE,hp.STATE] = self.Wuu
        KKT[hp.PARAMETER, hp.STATE] = self.Wmu
        KKT[hp.PARAMETER, hp.PARAMETER] = self.Wmm
        KKT[hp.ADJOINT, hp.STATE] = self.A
        KKT[hp.ADJOINT, hp.PARAMETER] = self.C
        
        if i >= j:
            KKT[i,j].mult(dir, out)
        else:
            KKT[j,i].transpmult(dir, out)
            
            
    def _createLUSolver(self):
        hp.PETScLUSolver(self.Vh[hp.STATE].mesh().mpi_comm() )
                

        
        