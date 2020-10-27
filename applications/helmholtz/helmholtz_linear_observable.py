import dolfin as dl
import numpy as np
import ufl

path_to_hippylib = '../../hippylib/'
import sys, os
sys.path.append( os.environ.get('HIPPYLIB_PATH',path_to_hippylib))
from hippylib import *

sys.path.append( os.environ.get('HIPPYFLOW_PATH'))
from hippyflow import *

from HelmholtzProblem import SingleSourceHelmholtzProblem, PML

dl.parameters["form_compiler"]["quadrature_degree"] = 6

def helmholtz_linear_observable(mesh,box = None, box_pml = None, n_obs = 100,output_folder ='helmholtz_setup/', verbose = False,seed = 0):
	'''

	'''
	########################################################################
	assert box is not None
	assert box_pml is not None
	# Construct the linear observable

	rank = dl.MPI.rank(mesh.mpi_comm())
	nproc = dl.MPI.size(mesh.mpi_comm())

	# Define the function spaces
	if rank == 0:
		sep = "\n"+"#"*80+"\n"
		print( sep, "Set up the finite element spaces", sep)
		
	Vh2 = dl.VectorFunctionSpace(mesh, 'Lagrange', 2)
	Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
	Vh = [Vh2, Vh1, Vh2]
	dims = [Vhi.dim() for Vhi in Vh]
	
	if rank == 0:
		print( "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*dims) )

	# Initialize Expressions
	n_sources = 10
	sources_loc = []
	for yi in np.linspace(box[1], box[3], n_sources):
		sources_loc.append( dl.Point(box[0], yi) )
	
	
	c = 343.4                                            #m/s     speed of sound in air
	rho = 1.204                                          #kg/m^3  density of air
	all_frequencies = np.array([100])      
	assert PML is not None         	 #Freq. in Hertz
	pml = PML(mesh, box, box_pml, 50)

	ntargets = 20
	ndim = 2
	np.random.seed(seed=1)
	targets = np.zeros([ntargets, ndim] )
	targets[0:ntargets//2,0] = box[2]-.1        # integer division
	targets[0:ntargets//2,1] = np.linspace(box[1]+.1, box[3]-.1, ntargets//2)
	targets[ntargets//2:,0]  = np.linspace(box[0]+.1, box[2]-.1, ntargets//2)
	targets[ntargets//2:,1]  = box[3]-.1
	
	if rank == 0:
		print( "Number of observation points: {0}".format(ntargets) )

	f = all_frequencies[0]
	omega = 2.*np.pi*f
	wave_number = dl.Constant(omega/(c*rho)) 
	pde = SingleSourceHelmholtzProblem(Vh, sources_loc, wave_number, pml)

	B = assemblePointwiseObservation(Vh[STATE], targets)

	observable = LinearStateObservable(pde,B)

	return observable