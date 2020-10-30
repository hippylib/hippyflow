import dolfin as dl
import numpy as np
import ufl

path_to_hippylib = '../../hippylib/'
import sys, os
sys.path.append( os.environ.get('HIPPYLIB_PATH',path_to_hippylib))
from hippylib import *

sys.path.append( os.environ.get('HIPPYFLOW_PATH'))
from hippyflow import LinearStateObservable

def confusion_linear_observable(mesh,n_obs = 100,output_folder ='confusion_setup/',\
									formulation = 'rhs', verbose = False,seed = 0):
	'''

	'''
	class confusion_varf:
		def __init__(self,Vh,save_fields = False,formulation = 'rhs',output_folder = 'confusion_setup/'):
			'''
		
			'''
			self.Vh = Vh
			# Gaussian blob for the right hand side
			self.f = dl.interpolate( dl.Expression('max(0.5,exp(-25*(pow(x[0]-0.7,2) +  pow(x[1]-0.7,2))))',degree=5), Vh[STATE])
			# Compute velocity field
			self.v = self.computeVelocityField(Vh[STATE].mesh())
			if save_fields:
				if not os.path.isdir(output_folder):
					os.mkdir(output_folder)
				f_pvd = dl.File(output_folder+'f_blob.pvd')
				f_pvd << self.f
				v_pvd = dl.File(output_folder+'v_sol.pvd')
				v_pvd << self.v
			# Constant coefficients for the PDE
			self.c = dl.Constant(1.0)
			self.k = dl.Constant(0.01)

			self.formulation = formulation
			print(80*'#')
			print(formulation.center(80))

		def computeVelocityField(self,mesh):
			'''

			'''
			def v_boundary(x,on_boundary):
				return on_boundary

			def q_boundary(x,on_boundary):
				return x[0] < dl.DOLFIN_EPS and x[1] < dl.DOLFIN_EPS
			
			Xh = dl.VectorFunctionSpace(mesh,'Lagrange', 2)
			Wh = dl.FunctionSpace(mesh, 'Lagrange', 1)
			mixed_element = ufl.MixedElement([Xh.ufl_element(), Wh.ufl_element()])
			XW = dl.FunctionSpace(mesh, mixed_element)

			Re = dl.Constant(1e2)

			g = dl.Expression(('0.0','(x[0] < 1e-14) - (x[0] > 1 - 1e-14)'), degree=1)
			bc1 = dl.DirichletBC(XW.sub(0), g, v_boundary)
			bc2 = dl.DirichletBC(XW.sub(1), dl.Constant(0), q_boundary, 'pointwise')
			bcs = [bc1, bc2]

			vq = dl.Function(XW)
			(v,q) = ufl.split(vq)
			(v_test, q_test) = dl.TestFunctions (XW)

			def strain(v):
				return ufl.sym(ufl.grad(v))

			F = ( (2./Re)*ufl.inner(strain(v),strain(v_test))+ ufl.inner (ufl.nabla_grad(v)*v, v_test)
				   - (q * ufl.div(v_test)) + ( ufl.div(v) * q_test) ) * ufl.dx

			dl.solve(F == 0, vq, bcs, solver_parameters={"newton_solver":
												 {"relative_tolerance":1e-4, "maximum_iterations":150}})
			return vq.split()[0]


		def __call__(self,u,m,p):
			'''
				Return the variational form of the PDE
				Inputs
					-u: state variable
					-m: model parameter
					-p: adjoint variable
				Outputs:
					Variational form of the PDE
			'''
			h = dl.CellDiameter(Vh[STATE].mesh())
			v_norm = dl.sqrt( dl.dot(self.v,self.v) + 1e-6)
			if self.formulation == 'rhs':
				return (h/v_norm)*dl.dot( self.v, dl.nabla_grad(u)) * dl.dot( self.v, dl.nabla_grad(p)) * dl.dx\
					+ self.k*dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx \
				   + dl.inner(dl.nabla_grad(u), self.v*p)*dl.dx \
				   + self.c*u*u*u*p*dl.dx \
				   - dl.exp(m)*self.f*p*dl.dx
			elif self.formulation == 'cubic_nonlinearity':
				return (h/v_norm)*dl.dot( self.v, dl.nabla_grad(u)) * dl.dot( self.v, dl.nabla_grad(p)) * dl.dx\
					+ self.k*dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx \
				   + dl.inner(dl.nabla_grad(u), self.v*p)*dl.dx \
				   + self.c*dl.exp(m)*u*u*u*p*dl.dx \
				   - self.f*p*dl.dx	 

			elif self.formulation == 'diffusion':
				self.k = dl.Constant(0.1)
				return (h/v_norm)*dl.dot( self.v, dl.nabla_grad(u)) * dl.dot( self.v, dl.nabla_grad(p)) * dl.dx\
					+ self.k*dl.exp(m)*dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx \
				   + dl.inner(dl.nabla_grad(u), self.v*p)*dl.dx \
				   + self.c*u*u*u*p*dl.dx \
				   - self.f*p*dl.dx	 


	def u_boundary(x, on_boundary):
		'''

		'''
		# return on_boundary and x[0] < dl.DOLFIN_EPS
		return on_boundary

	########################################################################

	# Construct the linear observable

	# Define the function spaces
	Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
	Vh = [Vh1, Vh1, Vh1]
	if verbose:
		print( "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()) )

	# Define points for the observable
	np.random.seed(seed=seed)
	x_targets = np.random.uniform(0.05,0.2, n_obs)
	y_targets = np.random.uniform(0.1,0.9, n_obs)
	targets = np.array(list(zip(x_targets,y_targets)))
	if verbose:
		print( "Number of observation points: {0}".format(targets.shape[0]) )

	# Define Dirichlet boundary conditions
	u_bdr = dl.Constant(0.0)
	u_bdr0 = dl.Constant(0.0)
	bc = dl.DirichletBC(Vh[STATE], u_bdr, u_boundary)
	bc0 = dl.DirichletBC(Vh[STATE], u_bdr0, u_boundary)

	
	m0 = dl.interpolate(dl.Constant(0.0), Vh[PARAMETER]).vector()
	param_dimension = m0.get_local().shape[0]
	m0.set_local(np.random.randn(param_dimension))

	varf_handler = confusion_varf(Vh, output_folder = output_folder,formulation = formulation)

	pde = PDEVariationalProblem(Vh, varf_handler, bc, bc0, is_fwd_linear=False)

	B = assemblePointwiseObservation(Vh[STATE], targets)

	observable = LinearStateObservable(pde,B)

	return observable