# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California, Merced.
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
import numpy as np
import math

import sys, os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR'))
import hippylib as hp


def newtonSolver_ParameterList():
    """
    Generate a ParameterList for newtonFwdSolve and InexactNewtonCG classes.
    type: :code:`newtonSolver_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["max_iter"]              = [20, "maximum number of iterations for nonlinear forward solve"]
    parameters["rel_tolerance"]         = [1e-6, "we converge when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
    parameters["abs_tolerance"]         = [1e-9, "we converge when sqrt(g,g) <= abs_tolerance"]
    parameters["gdu_tolerance"]         = [1e-18, "we converge when (g,du) <= gdu_tolerance"]
    parameters["LS"]                    = [hp.LS_ParameterList(), "Sublist containing LS globalization parameters"]
    parameters["print_level"]           = [-1, "print info to screen if set to > 0. Do not print if set to 0"]
    
    return hp.ParameterList(parameters)



class ConstrainedNSolver:
    """
    Newton's method to solve constrained optimization problems.
    The Newton system is solved iteratively either with a direct solver or an iterative solver.
    The iterative solver requires a user defined preconditioner
    Globalization is performed using the armijo sufficient reduction condition (backtracking).
    The stopping criterion is based on a control on the norm of the gradient and a control of the
    inner product between the gradient and the Newton direction.
       
    The user must provide the variational forms for the energy functional and the gradient. 
    The Hessian of the energy functional can be either provided by the user
    or computed by FEniCS using automatic differentiation.
    NOTE: Only works for linearly constrained problems
    """
    termination_reasons = ["Maximum number of Iteration reached",      #0
                           "Norm of the gradient less than tolerance", #1
                           "Maximum number of backtracking reached",   #2
                           "Norm of (g, du) less than tolerance"       #3
                           ]

    def __init__(self, parameters=newtonSolver_ParameterList()):
        self.parameters = parameters

        self.it = 0
        self.converged = False
        self.reason = 0
        self.solver = None

    def solve(self, F, C, u, constraint_vec, bcs = [], bc0 = []):
        """
        Input:
            :code:`F` represents the energy functional.
            :code:`C` represents the constraint.
            :code:`u` represents the initial guess.
            :code:`u` will be overwritten on return.
            :code:`constraint_vec`: vector used to check that constraint is satisfied.
            :code:`bcs` represents the Dirichlet boundary conditions on the unknown u.
            :code:`bc0` represents the Dirichlet boundary conditions for the step (du) in the Newton iterations.
        """

        max_iter = self.parameters["max_iter"]
        rtol = self.parameters["rel_tolerance"]
        atol = self.parameters["abs_tolerance"]
        gdu_tol = self.parameters["gdu_tolerance"]
        c_armijo = self.parameters["LS"]["c_armijo"]
        max_backtrack = self.parameters["LS"]["max_backtracking_iter"]
        prt_level = self.parameters["print_level"]

        if self.solver is None:
            self.solver = dl.PETScLUSolver(u.function_space().mesh().mpi_comm())

        if prt_level > 0:
            print ("Solving Nonlinear Problem")


        bk_converged = True


        L = F + C
        grad = dl.derivative(L, u)
        H = dl.derivative(grad, u)

        # Applying boundary conditions
        if type(bcs) is dl.DirichletBC:
            bcsl  = [bcs]
        else:
            bcsl = bcs
           
        [bc.apply(u.vector()) for bc in bcsl]


        # Setting variables
        Fn = dl.assemble(F)
        gn = dl.assemble(grad)
        g0_norm = gn.norm("l2")
        gn_norm = g0_norm
        tol = max(g0_norm*rtol, atol)
        du = dl.Vector()

        if prt_level > 0:
            print( "{0:>3} {1:>15} {2:>15} {3:>15} {4:>15}".format(
                   "Nit",  "Energy", "||g||", "(g,du)", "alpha") )
            print ( "{0:3d} {1:15e} {2:15e}  {3:15}   {4:15}".format(
                  0, Fn, g0_norm, "    -    ", "    -") )

        self.converged = False
        self.reason = 0
         
        for self.it in range(max_iter):
            
            [Hn, gn] = dl.assemble_system(H, grad, bc0)
            self.solver.set_operator(Hn)
            Hn.init_vector(du,1)
            
            # Ensure that at the end of the first iteration
            # the linear constraint is satisfied. 
            # If the constraint is not satisfied we find the minimum energy
            # to satisfy the constraint
            if self.it == 0:
                constraint_violation = gn*constraint_vec
                if constraint_violation.norm("l2") > 1.e-6:
                    #self.solver.solve(du, -constraint_violation.norm)   
                    self.solver.solve(du, -constraint_violation)
                    u.vector().axpy(1.,du)
                    Fn =  dl.assemble(F)
                    continue
            
            self.solver.solve(du,-gn)            
            du_gn = du.inner(gn)
            
            alpha = 1.0
            if (np.abs(du_gn) < gdu_tol):
                self.converged = True
                self.reason = 3
                u.vector().axpy(alpha,du)
                Fn = dl.assemble(F)
                gn_norm = gn.norm("l2")
                break
            


            u_backtrack = u.copy(deepcopy=True)
            bk_converged = False

            #Backtrack
            for j in range(max_backtrack):
                u.assign(u_backtrack)
                u.vector().axpy(alpha,du)
                Fnext = dl.assemble(F)
                if Fnext < Fn + alpha*c_armijo*du_gn:
                    Fn = Fnext
                    bk_converged = True
                    break
                alpha /= 2.

            if not bk_converged:
                self.reason = 2
                break

            gn_norm = gn.norm("l2")

            if prt_level > 0:
                print ( "{0:3d}  {1:15e} {2:15e} {3:15e} {4:15e}".format(
                      self.it+1,  Fn, gn_norm, du_gn, alpha) )

            if gn_norm < tol:
                self.converged = True
                self.reason = 1
                break

        self.it = self.it+1
        if prt_level > 0:
            if self.reason == 3:
                print ( "{0:3d}   {1:15e} {2:15e} {3:15e} {4:15e}".format(
                        self.it,  Fn, gn_norm, du_gn, alpha) )
            print( self.termination_reasons[self.reason] )
            if self.converged:
                print( "Newton converged in ", self.it, \
                    "nonlinear iterations." )
            else:
                print( "Newton did NOT converge in ", self.it, "iterations." )
                
            print ("Final norm of the gradient: ", gn_norm)
            print ("Value of the cost functional: ", Fn )

        return u, self.reason

