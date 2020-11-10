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

def helmholtz_linear_observable(mesh,box = None, box_pml = None, sqrt_n_obs = 10,output_folder ='helmholtz_setup/',frequency = 300, verbose = False,seed = 0):
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
	# n_sources = 1
	# sources_loc = []
	# for xi in np.linspace(box[0], box[2], n_sources):
	# 	sources_loc.append( dl.Point(xi, box[3]-0.15) )

	source_loc_ = ((box[0]+.1+(box[2]-0.1)/2)/2,box[3]-0.15)
	
	sources_loc = [dl.Point(*source_loc_)]
	
	
	c = 343.4                                            #m/s     speed of sound in air
	rho = 1.204                                          #kg/m^3  density of air
	all_frequencies = np.array([frequency])  
	assert PML is not None         	 #Freq. in Hertz
	pml = PML(mesh, box, box_pml, 50)

	ndim = 2
	x_targets = np.linspace(source_loc_[0]-0.2,source_loc_[0]+.2,sqrt_n_obs)
	y_targets = np.linspace(box[3]-0.25,box[3]-0.05,sqrt_n_obs)
	targets = []
	for xi in x_targets:
		for yi in y_targets:
			targets.append((xi,yi))
	targets = np.array(targets)


	# np.random.seed(seed=1)
	# targets = np.zeros([ntargets, ndim] )
	# targets[:,0]  = np.linspace(box[0]+.1, (box[2]-.1)/2, ntargets)
	# targets[:,1]  = box[3]-.1
	
	ntargets = len(targets)
	if rank == 0:
		print( "Number of observation points: {0}".format(ntargets) )

	f = all_frequencies[0]
	omega = 2.*np.pi*f
	wave_number = dl.Constant(omega/(c*rho)) 
	pde = SingleSourceHelmholtzProblem(Vh, sources_loc, wave_number, pml)

	B = assemblePointwiseObservation(Vh[STATE], targets)

	observable = LinearStateObservable(pde,B)

	return observable