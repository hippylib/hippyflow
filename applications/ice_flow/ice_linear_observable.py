import dolfin as dl
import numpy as np
import time

path_to_hippylib = '../../hippylib/'
import sys, os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR',path_to_hippylib))
from hippylib import *

sys.path.append('../')
from hippyflow import *

from energyFunctionalPDEVariationalProblem import *


class DiskBoundaries:
	class bottom_boundary(dl.SubDomain):
		def inside(self,x,on_boundary):
			return on_boundary and x[2] < 1e-8

	class side_boundary(dl.SubDomain):
		def inside(self,x,on_boundary):
			on_flat_side = (x[2] > 1e-8) and (x[2] - 0.1< 1e-8)
			return on_boundary and x[2] > 0 and on_flat_side

	class top_boundary(dl.SubDomain):
		def inside(self,x,on_boundary):
			on_flat_side = (x[2] > 1e-8) and (x[2] - 0.1< 1e-8)

			return on_boundary and x[2] > 0 and (not on_flat_side)

	class interior(dl.SubDomain):
		def inside(self,x,on_boundary):
			return not on_boundary

class ChoppedDomeBoundaries:
	class bottom_boundary(dl.SubDomain):
		def inside(self,x,on_boundary):
			return on_boundary and x[2] < 1.e-8
	class side_boundary(dl.SubDomain):
		def inside(self,x,on_boundary):
			on_flat_side = (x[0] -0.8 < 1e-8 or x[0] + 0.8 < 1e-8 or  x[1] -0.8 < 1e-8 or x[1] + 0.8 < 1e-8 )
			return on_boundary and x[2] > 0 and on_flat_side
	class top_boundary(dl.SubDomain):
		def inside(self,x,on_boundary):
			on_flat_side = (x[0] -0.8 < 1e-8 or x[0] + 0.8 < 1e-8 or  x[1] -0.8 < 1e-8 or x[1] + 0.8 < 1e-8 )
			return on_boundary and x[2] > 0 and (not on_flat_side)
	class interior(dl.SubDomain):
		def inside(self,x,on_boundary):
			return not on_boundary

def ice_linear_observable(mesh,mesh_file = '',formulation = 'energy',n_obs = 100,verbose = False,output_folder = ''):
	# Reduce quadrature degree
	dl.parameters['form_compiler']['quadrature_degree'] = 4

	# # Load mesh file in and partition in parallel is so desired.
	# mesh = dl.Mesh(split_comm)
	# with dl.XDMFFile(mesh_file) as fid:
	# 	fid.read(mesh)
	# Define subdomains
	# These are mesh dependent, for each mesh file different subdomains should be implemented
	if 'disk' in mesh_file:
		boundaries = DiskBoundaries()
		targets = sample_targets_disk(n_obs)
	elif 'chop' in mesh_file:
		boundaries = ChoppedDomeBoundaries() 
		targets = sample_near_surface_half_dome(n_obs)


	sub_domains = dl.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
	sub_domains.set_all(0)
	bottom = boundaries.bottom_boundary()
	bottom.mark(sub_domains,1)
	side = boundaries.side_boundary()
	side.mark(sub_domains,0)
	top = boundaries.top_boundary()
	top.mark(sub_domains,2)

	# Define facet normal and boundary integral measure
	normal = dl.FacetNormal(mesh)
	# Is this parallel safe?
	ds = dl.Measure("ds")(subdomain_data=sub_domains)

	if formulation == 'energy':
		rank = mesh.mpi_comm().rank	    
		P1 = dl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
		P2 = dl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
		TH = P2 * P1
		Vh2 = dl.FunctionSpace(mesh, TH) #periodic product space for state + adjoint
		Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
		Vh = [Vh2, Vh1, Vh2]
		
		ndofs = [Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()]
		if rank == 0:
			print ("Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs))

		# forcing term
		angle = 0.0*dl.pi/180.0
		grav = 9.81
		rho = 910.0

		# rheology
		n = 3.0
		A = dl.Constant(1e-16)

		# Forcing term
		f=dl.Constant( ( rho*grav*np.sin(angle), 0.0, -rho*grav*np.cos(angle) ) )

		# Dirichlet condition. The first sub(0) gets the velocity, sub(2) gets the z-compoenet of velocity
		bc = dl.DirichletBC(Vh[hp.STATE].sub(0).sub(2), dl.Constant(0.0), bottom)
		bc0 = dl.DirichletBC(Vh[hp.STATE].sub(0).sub(2), dl.Constant(0.0), bottom)


		# Define the Nonlinear stoke varfs
		nonlinearStokesFunctional = NonlinearStokesForm(n, A, normal, ds(1), f)
		
		# Create one-hot vector on pressure dofs
		constraint_vec = dl.interpolate( dl.Constant((0,0,0,1)), Vh[hp.STATE]).vector()

		pde = EnergyFunctionalPDEVariationalProblem(Vh, nonlinearStokesFunctional, constraint_vec, bc, bc0)
		
		if rank == 0:
			if verbose:
				pde.fwd_solver.parameters["print_level"] = 1
			else:
				pde.fwd_solver.parameters["print_level"] = -1
		else:
			pde.fwd_solver.parameters["print_level"] = -1

	else:
		U = dl.VectorElement("Lagrange",mesh.ufl_cell(), 2, dim=3) # velocity
		P = dl.FiniteElement("Lagrange",mesh.ufl_cell(), 1)       # pressure
		PP = dl.FunctionSpace(mesh,P)
		S = dl.FunctionSpace(mesh, dl.MixedElement([U,P]))
		Vh = [S, PP,S]

		s = dl.Function(S)
		s = dl.interpolate(dl.Constant((0.0,0.0,0.0,0.0)), S)
		f = dl.Constant((0,0,-1))

		ice_varf = ice_sheet_varf(Vh,sub_domains,f)
		bc_ = dl.DirichletBC(S.sub(0).sub(2),dl. Constant(0.0), boundaries.bottom_boundary())
		bc = [bc_]
		bc0 = [bc_]
		pde = IceVariationalProblem(Vh, ice_varf, bc, bc0, is_fwd_linear=False)

		def bottom_function(x,on_boundary):
			return on_boundary and x[2] < 1.e-8

		pde.manifold_dofs = subdomain_dofs(bottom_function,Vh[PARAMETER])
		pde.dsGamma = ice_varf.ds(2)
	
	B = assemblePointwiseObservation(Vh[STATE], targets)

	observable = LinearStateObservable(pde,B)

	return observable



class ice_sheet_varf:
	def __init__(self,Vh,sub_domains,f):
		self.Vh = Vh
		self.mesh = Vh[STATE].mesh()
		self.n = dl.FacetNormal(self.mesh)
		self.ds = dl.Measure("ds")(subdomain_data=sub_domains)
		self.f = f
		
		
	def D(self,u):
		return dl.sym(dl.grad(u))

	def mu(self,u,eps = 1e-3):
		if False:
			return dl.Constant(1.0)
		eps = dl.Constant(eps)
		return (dl.inner(self.D(u),self.D(u)) + eps)**(-2./3.)

	def tangent_projection(self):
		return dl.Identity(3) - dl.outer(self.n,self.n)
	
	
	def __call__(self,uu,m,pp):
		u, p = dl.split(uu)
		v, q = dl.split(pp)
		# F = dl.inner(self.mu(u)*self.D(u),self.D(v))*dl.dx -\
		# dl.inner(dl.exp(m)*(self.tangent_projection()*u),self.tangent_projection()*v)*self.ds(1)\
		# - dl.div(v)*p*dl.dx + q*dl.div(u)*dl.dx - dl.inner(self.f,v)*dl.dx +\
		# dl.inner(1e-2*p,q)*dl.dx + dl.inner(1e-2*u,v)*dl.dx
		F = dl.inner(self.mu(u)*self.D(u),self.D(v))*dl.dx -\
		dl.inner(dl.exp(m)*(self.tangent_projection()*u),self.tangent_projection()*v)*self.ds(1)\
		- dl.div(v)*p*dl.dx + q*dl.div(u)*dl.dx - dl.inner(self.f,v)*dl.dx
		return F

	def picard_varf(self,uu,m,pp,uu_k,eps = 1.0):
		u, p = dl.split(uu)
		v, q = dl.split(pp)
		u_k, _ = dl.split(uu_k)
		F = dl.inner(self.mu(u_k,eps)*self.D(u),self.D(v))*dl.dx -\
		dl.inner(dl.exp(m)*(self.tangent_projection()*u),self.tangent_projection()*v)*self.ds(1)\
		- dl.div(v)*p*dl.dx + q*dl.div(u)*dl.dx - dl.inner(self.f,v)*dl.dx +\
		dl.inner(1e-6*p,q)*dl.dx + dl.inner(1e-6*u,v)*dl.dx
		return F


def subdomain_dofs(subdomain,V):
	ff = dl.MeshFunction("size_t", V.mesh(),V.mesh().topology().dim() - 1)
	dl.AutoSubDomain(subdomain).mark(ff, 1)


	u = dl.Function(V)
	bc = dl.DirichletBC(V, dl.Constant(1.0), ff, 1)
	bc.apply(u.vector())

	dofs = np.where(u.vector()==1.0)[0]
	return dofs




class IceVariationalProblem(PDEVariationalProblem):

	def solveFwd(self,state,x,picard = False):
		u = vector2Function(x[STATE],self.Vh[STATE])
		m = vector2Function(x[PARAMETER],self.Vh[PARAMETER])
		p = dl.TestFunction(self.Vh[ADJOINT])
		if not picard:
			F = self.varf_handler(u,m,p)
			du = dl.TrialFunction(self.Vh[STATE])
			JF = dl.derivative(F, u, du)
			problem = dl.NonlinearVariationalProblem(F,u,self.bc,JF)
			solver = dl.NonlinearVariationalSolver(problem)

			prm = solver.parameters
			# print('newton solver parameters = ',solver.parameters['newton_solver'].keys())
			if False:
				prm['newton_solver']['absolute_tolerance'] = 1E-4
				prm['newton_solver']['report'] = True
				prm['newton_solver']['relative_tolerance'] = 1E-3
				prm['newton_solver']['maximum_iterations'] = 200
				prm['newton_solver']['relaxation_parameter'] = 1.0
				# print(dl.info(solver.parameters, True))
			if True:
				prm['nonlinear_solver']='snes' 
				prm['snes_solver']['line_search'] = 'basic'
				prm['snes_solver']['linear_solver']= 'lu'
				prm['snes_solver']['report'] = True
				prm['snes_solver']['error_on_nonconvergence'] = False
				prm['snes_solver']['absolute_tolerance'] = 1E-5
				prm['snes_solver']['relative_tolerance'] = 1E-5
				prm['newton_solver']['absolute_tolerance'] = 1E-3
				prm['newton_solver']['relative_tolerance'] = 1E-2
				prm['newton_solver']['maximum_iterations'] = 200
				prm['newton_solver']['relaxation_parameter'] = 1.0

				# print(dl.info(solver.parameters, True))
			iterations, converged = solver.solve()

			state.zero()
			state.axpy(1.,u.vector())
		if picard:
			uu_k = dl.Function(self.Vh[STATE])
			uu_k = dl.interpolate(dl.Constant((0.0,0.0,0.0,0.0)), self.Vh[STATE])
			u_trial = dl.TrialFunction(self.Vh[STATE])
			p = dl.TestFunction(self.Vh[ADJOINT])
			F = self.varf_handler.picard_varf(u_trial,m,p,uu_k)
			picard_iteration = 0
			res = 1
			while res > 1e-4 and picard_iteration < 50:
				Ap, bp = dl.assemble_system(dl.lhs(F), dl.rhs(F), self.bc)
				dl.solve(Ap,u.vector(),bp)
				res = dl.sqrt(dl.assemble(dl.inner(u-uu_k,u-uu_k)*dl.dx))
				print('||u-u_k||_Omega = ', res)
				# m_norm = dl.sqrt(dl.assemble(dl.inner(dl.exp(m),dl.exp(m))*dl.dx))
				# print('||m||_Omega = ',m_norm)
				uu_k.assign(u)
				picard_iteration += 1
			state.zero()
			state.axpy(1.,u.vector())


def log_mean_function(Vh,easy_factor = 4.):
	t0 = time.time()
	print('Setting expression')
	m_expression = dl.Expression('6.0*std::log(10)*(1.0 - (x[0]*x[0] +\
			x[1]*x[1])*std::cos(10*std::atan2(x[0],x[1])))',degree = 5)
	# m_expression = dl.Expression('7.0 + 2.0*std::sin(x[0]*2.*pi)*std::sin(x[1]*2.*pi)',degree = 5)
	
	duration = time.time() -t0
	# print('Setting expression took ',duration,'s, now interpolating')
	# sys.stdout.flush()
	t0 = time.time()
	prior_mean = dl.interpolate(m_expression,Vh[PARAMETER]).vector()
	if True:
		prior_mean /= easy_factor
	duration = time.time() -t0
	# print('Finished Interpolating and it took ',duration, 's')
	# sys.stdout.flush()
	return prior_mean



def sample_targets_half_dome(n_obs=300,seed =0):
	random_state = np.random.RandomState(seed= seed)
	targets = np.zeros((n_obs,3))
	for i in range(n_obs):
		r = random_state.uniform(0,0.8)
		theta = random_state.uniform(0,np.pi/2)
		phi = random_state.uniform(0,2*np.pi)
		targets[i,0] = r*np.sin(theta)*np.cos(phi)
		targets[i,1] = r*np.sin(theta)*np.sin(phi)
		targets[i,2] = r*np.cos(theta)
	return targets

def sample_near_surface_half_dome(n_obs=300,shell_thickness = 0.15, eps = 0.05,seed = 0):
	random_state = np.random.RandomState(seed= seed)
	targets = np.zeros((n_obs,3))
	for i in range(n_obs):
		accepted = False
		while not accepted:
			x = random_state.uniform(-0.8+eps,0.8-eps)
			y = random_state.uniform(-0.8+eps,0.8-eps)
			z = random_state.uniform(eps,1.0)
			inside_domain = (x**2 + y**2 + z**2 < 1. - eps)
			inside_shell = inside_domain and (x**2 + y**2 + z**2 > 1. - shell_thickness)
			accepted = inside_shell
			if accepted:
				targets[i,0] = x
				targets[i,1] = y
				targets[i,2] = z
	return targets


def sample_targets_disk(n_obs=300):
	targets = np.zeros((n_obs,3))
	for i in range(n_obs):
		r = np.random.uniform(0,0.8)
		theta = np.random.uniform(0,2*np.pi)
		z = np.random.uniform(0,0.09)
		targets[i,0] = r*np.cos(theta)
		targets[i,1] = r*np.sin(theta)
		targets[i,2] = z
	return targets
