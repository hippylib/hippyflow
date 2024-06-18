# Copyright (c) 2023, The University of Texas at Austin
# & Georgia Institute of Technology
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the SOUPy package. For more information see
# https://github.com/hippylib/soupy/
#
# SOUPy is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 2007.


import dolfin as dl
import math
import numpy as np
import hippylib as hp

import ufl

STATE = 0
PARAMETER = 1
ADJOINT = 2
CONTROL = 3


class PDEVariationalControlProblem(hp.PDEVariationalProblem):
    def __init__(self, Vh, varf_handler, bc, bc0, is_fwd_linear = False, lu_method="mumps"):
        """
        Constructor

        :param Vh: List of function spaces the state, parameter, adjoint, and control
        :type Vh: list of :py:class:`dolfin.FunctionSpace`
        :param varf_handler: Variational form handler with :code:`__call__` method
        :param bc: List of Dirichlet boundary conditions for the state
        :param bc0: List of zeroed Dirichlet boundary conditions
        :param is_fwd_linear: Flag indicating whether the forward problem is linear
        :type is_fwd_linear: bool
        :param lu_method: Method for solving linear systems (default, mumps, etc.)
        :type lu_method: str
        """

        # assert for class assumptions here
        assert id(Vh[STATE]) == id(Vh[ADJOINT]), print('Need to have same STATE and ADJOINT spaces')
        assert len(Vh) == 4
        # assert Vh[STATE].mesh().mpi_comm().size == 1, print('Only worked out for serial codes')

        self.Vh = Vh
        self.varf_handler = varf_handler
        if type(bc) is dl.DirichletBC:
            self.bc = [bc]
        else:
            self.bc = bc
        if type(bc0) is dl.DirichletBC:
            self.bc0 = [bc0]
        else:
            self.bc0 = bc0

        self.A  = None
        self.At = None
        self.C = None
        self.Cz = None
        self.Wmu = None
        self.Wmm = None
        self.Wzu = None
        self.Wzz = None
        self.Wuu = None

        self.solver = None
        self.solver_fwd_inc = None
        self.solver_adj_inc = None
        self.lu_method = lu_method

        self.is_fwd_linear = is_fwd_linear
        self.n_calls = {"forward": 0,
                        "adjoint":0 ,
                        "incremental_forward":0,
                        "incremental_adjoint":0}
        self.n_linear_solves = 0
        self.nonlinear_solver_parameters = None

    def set_nonlinear_solver_parameters(self, parameters):
        """ Set the solver parameters used for `dolfin.NonlinearVariationalSolver`

        :param parameters: Solver parameters for `dolfin.NonlinearVariationalSolver`
        :type parameters: dict
        """
        self.nonlinear_solver_parameters = parameters

    def generate_state(self):
        """ Return a vector in the shape of the state. """
        return dl.Function(self.Vh[STATE]).vector()

    def generate_parameter(self):
        """ Return a vector in the shape of the parameter. """
        return dl.Function(self.Vh[PARAMETER]).vector()

    def generate_control(self):
        return dl.Function(self.Vh[CONTROL]).vector()

    def init_parameter(self, m):
        """ Initialize the parameter. """
        dummy = self.generate_parameter()
        # This should be addressed at some point
        # m.init( dummy.mpi_comm(), dummy.local_range() )
        m.init( dummy.local_range() )

    def init_control(self, z):
        """ Initialize the parameter. """
        dummy = self.generate_control()
        # This should be addressed at some point
        # z.init( dummy.mpi_comm(), dummy.local_range() )
        z.init(dummy.local_range())

    def solveFwd(self, state, x):
        """ Solve the possibly nonlinear forward problem:
        Given :math:`m, z`, find :math:`u` such that

            .. math:: \\delta_p F(u, m, p, z;\\hat{p}) = 0,\\quad \\forall \\hat{p}."""
        self.n_calls["forward"] += 1
        if self.solver is None:
            self.solver = self._createLUSolver()
        if self.is_fwd_linear:
            u = dl.TrialFunction(self.Vh[STATE])
            m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            p = dl.TestFunction(self.Vh[ADJOINT])
            z = hp.vector2Function(x[CONTROL], self.Vh[CONTROL])
            res_form = self.varf_handler(u, m, p, z)
            A_form = ufl.lhs(res_form)
            b_form = ufl.rhs(res_form)
            A, b = dl.assemble_system(A_form, b_form, bcs=self.bc)
            self.solver.set_operator(A)
            self.solver.solve(state, b)
            self.n_linear_solves += 1
        else:
            u = hp.vector2Function(x[STATE], self.Vh[STATE])
            m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            p = dl.TestFunction(self.Vh[ADJOINT])
            z = hp.vector2Function(x[CONTROL], self.Vh[CONTROL])
            res_form = self.varf_handler(u, m, p, z)
            jacobian_form = dl.derivative(res_form, u)
            nonlinear_problem = dl.NonlinearVariationalProblem(res_form, u, self.bc, jacobian_form)
            solver = dl.NonlinearVariationalSolver(nonlinear_problem)

            if self.nonlinear_solver_parameters is not None:
                solver.parameters.update(self.nonlinear_solver_parameters)

            num_iters, converged = solver.solve()
            state.zero()
            state.axpy(1., u.vector())
            self.n_linear_solves += num_iters


    def solveAdj(self, adj, x, adj_rhs):
        """ Solve the linear adjoint problem:
            Given :math:`m, z, u`; find :math:`p` such that

                .. math:: \\delta_u F(u, m, p, z;\\hat{u}) = 0, \\quad \\forall \\hat{u}.
        """
        self.n_calls["adjoint"] += 1
        if self.solver is None:
            self.solver = self._createLUSolver()

        u = hp.vector2Function(x[STATE], self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        z = hp.vector2Function(x[CONTROL], self.Vh[CONTROL])
        du = dl.TestFunction(self.Vh[STATE])
        dp = dl.TrialFunction(self.Vh[ADJOINT])
        varf = self.varf_handler(u, m, p, z)
        adj_form = dl.derivative( dl.derivative(varf, u, du), p, dp )
        Aadj, dummy = dl.assemble_system(adj_form, ufl.inner(u,du)*ufl.dx, self.bc0)
        self.solver.set_operator(Aadj)

        # Apply the zeroed Dirichlet boundary conditions
        # This is safer then the standard hippylib approach where
        # the boundary conditions are assumed to have been applied
        # before being passed into the :code:`solveAdj` method
        local_adj_rhs = adj_rhs.copy()
        for bc0 in self.bc0:
            bc0.apply(local_adj_rhs)

        self.solver.solve(adj, local_adj_rhs)
        self.n_linear_solves += 1

    def evalGradientParameter(self, x, out):
        """Given :math:`u, m, p, z`; evaluate :math:`\\delta_m F(u, m, p, z; \\hat{m}),\\, \\forall \\hat{m}.` """
        u = hp.vector2Function(x[STATE], self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = hp.vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        z = hp.vector2Function(x[CONTROL], self.Vh[CONTROL])
        dm = dl.TestFunction(self.Vh[PARAMETER])
        res_form = self.varf_handler(u, m, p, z)
        out.zero()
        dl.assemble( dl.derivative(res_form, m, dm), tensor=out)

    def evalGradientControl(self, x, out):
        """Given :math:`u, m, p, z`; evaluate :math:`\\delta_z F(u, m, p, z; \\hat{z}),\\, \\forall \\hat{z}.` """
        u = hp.vector2Function(x[STATE], self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = hp.vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        z = hp.vector2Function(x[CONTROL], self.Vh[CONTROL])
        dz = dl.TestFunction(self.Vh[CONTROL])
        res_form = self.varf_handler(u, m, p, z)
        out.zero()
        dl.assemble( dl.derivative(res_form, z, dz), tensor=out)

    def setLinearizationPoint(self,x, gauss_newton_approx):
        """ Set the values of the state and parameter
            for the incremental forward and adjoint solvers. """

        x_fun = [hp.vector2Function(x[i], self.Vh[i]) for i in range(4)]

        f_form = self.varf_handler(*x_fun)

        g_form = [None,None,None,None]
        for i in range(4):
            g_form[i] = dl.derivative(f_form, x_fun[i])

        self.A, dummy = dl.assemble_system(dl.derivative(g_form[ADJOINT],x_fun[STATE]), g_form[ADJOINT], self.bc0)
        self.At, dummy = dl.assemble_system(dl.derivative(g_form[STATE],x_fun[ADJOINT]),  g_form[STATE], self.bc0)
        self.C = dl.assemble(dl.derivative(g_form[ADJOINT],x_fun[PARAMETER]))
        self.Cz = dl.assemble(dl.derivative(g_form[ADJOINT],x_fun[CONTROL]))
        [bc.zero(self.C) for bc in self.bc0]
        [bc.zero(self.Cz) for bc in self.bc0]

        if self.solver_fwd_inc is None:
            self.solver_fwd_inc = self._createLUSolver()
            self.solver_adj_inc = self._createLUSolver()

        self.solver_fwd_inc.set_operator(self.A)
        self.solver_adj_inc.set_operator(self.At)

        if gauss_newton_approx:
            self.Wuu = None
            self.Wmu = None
            self.Wmm = None
            self.Wzu = None
            self.Wzz = None
        else:
            self.Wuu = dl.assemble(dl.derivative(g_form[STATE],x_fun[STATE]))
            [bc.zero(self.Wuu) for bc in self.bc0]
            # print("WUU NORM", np.linalg.norm(self.Wuu.array()))
            Wuu_t = hp.Transpose(self.Wuu)
            [bc.zero(Wuu_t) for bc in self.bc0]
            self.Wuu = hp.Transpose(Wuu_t)
            self.Wmu = dl.assemble(dl.derivative(g_form[PARAMETER],x_fun[STATE]))
            Wmu_t = hp.Transpose(self.Wmu)
            [bc.zero(Wmu_t) for bc in self.bc0]
            self.Wmu = hp.Transpose(Wmu_t)
            self.Wmm = dl.assemble(dl.derivative(g_form[PARAMETER],x_fun[PARAMETER]))

            self.Wzu = dl.assemble(dl.derivative(g_form[CONTROL],x_fun[STATE]))
            Wzu_t = hp.Transpose(self.Wzu)
            [bc.zero(Wzu_t) for bc in self.bc0]
            self.Wzu = hp.Transpose(Wzu_t)
            self.Wzz = dl.assemble(dl.derivative(g_form[CONTROL],x_fun[CONTROL]))


    def solveIncremental(self, out, rhs, is_adj):
        """ If :code:`is_adj == False`:

            Solve the forward incremental system:
            Given :math:`u, m, z`, find :math:`\\tilde{u}` such that

                .. math:: \\delta_{pu} F(u, m, p, z ; \\hat{p}, \\tilde{u}) = \\mbox{rhs},\\quad \\forall \\hat{p}.

            If :code:`is_adj == True`:

            Solve the adjoint incremental system:
            Given :math:`u, m, z`, find :math:`\\tilde{p}` such that

                .. math:: \\delta_{up} F(u, m, p, z; \\hat{u}, \\tilde{p}) = \\mbox{rhs},\\quad \\forall \\hat{u}.
        """

        # Apply the zeroed Dirichlet boundary conditions
        # This is safer then the standard hippylib approach where
        # the boundary conditions are assumed to have been applied
        # before being passed into the :code:`solveIncremental` method
        local_rhs = rhs.copy()
        for bc0 in self.bc0:
            bc0.apply(local_rhs)

        if is_adj:
            self.n_calls["incremental_adjoint"] += 1
            self.solver_adj_inc.solve(out, local_rhs)
        else:
            self.n_calls["incremental_forward"] += 1
            self.solver_fwd_inc.solve(out, local_rhs)
        self.n_linear_solves += 1

    def apply_ij(self,i,j, dir, out):
        """
            Given :math:`u, m, p, z`; compute
            :math:`\\delta_{ij} F(u, m, p, z; \\hat{i}, \\tilde{j})` in the direction :math:`\\tilde{j} =` :code:`dir`,
            :math:`\\forall \\hat{i}`.
        """
        KKT = {}
        KKT[STATE,STATE] = self.Wuu
        KKT[PARAMETER, STATE] = self.Wmu
        KKT[PARAMETER, PARAMETER] = self.Wmm
        KKT[ADJOINT, STATE] = self.A
        KKT[ADJOINT, PARAMETER] = self.C

        KKT[CONTROL, STATE] = self.Wzu
        KKT[CONTROL, CONTROL] = self.Wzz
        KKT[CONTROL, ADJOINT] = hp.Transpose(self.Cz)

        if i == ADJOINT and j == CONTROL:
            # Check Cz first since the index ordering is different with CONTROL
            # This avoids doing transpmult of hp.Tranpose(Cz)
            self.Cz.mult(dir, out)

        elif i >= j:
            if KKT[i,j] is None:
                out.zero()
            else:
                KKT[i,j].mult(dir, out)
        else:
            if KKT[j,i] is None:
                out.zero()
            else:
                KKT[j,i].transpmult(dir, out)

    def apply_ijk(self,i,j,k, x, jdir, kdir, out):
        x_fun = [hp.vector2Function(x[ii], self.Vh[ii]) for ii in range(4)]
        idir_fun = dl.TestFunction(self.Vh[i])
        jdir_fun = hp.vector2Function(jdir, self.Vh[j])
        kdir_fun = hp.vector2Function(kdir, self.Vh[k])

        res_form = self.varf_handler(*x_fun)
        form = dl.derivative(
               dl.derivative(
               dl.derivative(res_form, x_fun[i], idir_fun),
               x_fun[j], jdir_fun),
               x_fun[k], kdir_fun)

        out.zero()
        dl.assemble(form, tensor=out)

        if i in [STATE,ADJOINT]:
            [bc.apply(out) for bc in self.bc0]

    def _createLUSolver(self):
        if hasattr(self, 'lu_method'):
            return hp.PETScLUSolver(self.Vh[STATE].mesh().mpi_comm(), method=self.lu_method)
        else:
            return hp.PETScLUSolver(self.Vh[STATE].mesh().mpi_comm(), method="default")


class UniformDistribution:
    """
    Class for sampling from a uniform distribution to `dl.Vector` \
        Sampling uses the numpy class and hence is not mesh parallel safe \

    .. note:: Used only for tests
    """
    def __init__(self, Vh, a, b, rng_seed=1):
        """
        Constructor

        :param Vh: Function space for sample vectors
        :param a: Lower bound
        :param b: Upper bound
        :param ndim: Dimension of sample vectors
        :param rng_seed: Seed for the random state
        """
        assert Vh.mesh().mpi_comm().size == 1
        self.Vh = Vh
        self.a = a
        self.b = b
        self.ndim = self.Vh.dim()
        self._dummy = dl.Function(Vh).vector()
        self.random_state = np.random.RandomState(rng_seed)

    def init_vector(self, v):
        v.init( self._dummy.local_range() )

    def sample(self, out):
        v = self.random_state.rand(self.ndim) * (self.b-self.a) + self.a
        # v = np.random.rand(self.ndim) * (self.b-self.a) + self.a
        out.set_local(v)


def u_boundary(x, on_boundary):
    return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)



def setupPoissonPDEProblem(Vh, settings):
    """
    Set up the Poisson PDE problem. Used as an example PDE for unit tests
    """

    anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree = 1)
    anis_diff.set(settings['THETA0'], settings['THETA1'], settings['ALPHA'])
    m_mean_fun = dl.Function(Vh[PARAMETER])
    m_mean_fun.interpolate(dl.Constant(1.0))
    prior = hp.BiLaplacianPrior(Vh[PARAMETER], settings['GAMMA'], settings['DELTA'],\
                                anis_diff, mean=m_mean_fun.vector(), robin_bc=True)

    bc = dl.DirichletBC(Vh[STATE], dl.Expression("x[1]", degree=1), u_boundary)
    bc0 = dl.DirichletBC(Vh[STATE], dl.Constant(0.0), u_boundary)
    poisson_varf = PoissonVarfHandler(Vh, settings=settings)
    pde = PDEVariationalControlProblem(Vh, poisson_varf, bc, bc0,
            is_fwd_linear=settings["LINEAR"])

    control_dist = UniformDistribution(Vh[CONTROL],
            settings['STRENGTH_LOWER'],
            settings['STRENGTH_UPPER'])

    return pde, prior, control_dist



def poisson_control_settings():
    """
    Settings for the poisson PDE problem
    """
    settings = {}
    settings['nx'] = 20
    settings['ny'] = 20

    settings['STRENGTH_UPPER'] = 1.
    settings['STRENGTH_LOWER'] = -1.
    settings['LINEAR'] = True


    settings['N_WELLS_PER_SIDE'] = 5
    settings['LOC_LOWER'] = 0.25
    settings['LOC_UPPER'] = 0.75
    settings['WELL_WIDTH'] = 0.1

    settings['GAMMA'] = 1.0
    settings['DELTA'] = 20.0
    settings['THETA0'] = 2.0
    settings['THETA1'] = 0.5
    settings['ALPHA'] = math.pi/4

    return settings


class PoissonVarfHandler:
    """
    Variational form for the poisson PDE
    """
    def __init__(self,Vh,settings = poisson_control_settings()):
        """
        Constructor:

        :param Vh: List of function spaces for state, adjoint, parameter, control
        :param settings: Dictionary for settings for the PDE problem
        """
        self.linear = settings['LINEAR']

        # Set up the control right hand side
        well_grid = np.linspace(settings['LOC_LOWER'],settings['LOC_UPPER'],settings['N_WELLS_PER_SIDE'])
        well_grid_x, well_grid_y = np.meshgrid(well_grid, well_grid)
        mollifier_list = []

        for i in range(settings['N_WELLS_PER_SIDE']):
            for j in range(settings['N_WELLS_PER_SIDE']):
                mollifier_list.append(
                        dl.Expression("a*exp(-(pow(x[0]-xi,2)+pow(x[1]-yj,2))/(pow(b,2)))",
                            xi=well_grid[i],
                            yj=well_grid[j],
                            a=1/(2*math.pi*settings['WELL_WIDTH']**2),
                            b=settings['WELL_WIDTH'],
                            degree=2)
                        )

        self.mollifiers = dl.as_vector(mollifier_list)

        assert Vh[CONTROL].dim() == len(mollifier_list)


    def __call__(self,u,m,p,z):
        if self.linear:
            return dl.exp(m)*dl.inner(dl.grad(u),dl.grad(p))*dl.dx - dl.inner(self.mollifiers,z)*p*dl.dx
        else:
            return dl.exp(m)*dl.inner(dl.grad(u),dl.grad(p))*dl.dx + u**3*p*dl.dx  - dl.inner(self.mollifiers,z)*p*dl.dx



