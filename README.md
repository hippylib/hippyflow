		Dimension reduced surrogate construction for parametric PDE maps

		      ___                       ___         ___               
		     /__/\        ___          /  /\       /  /\        ___   
		     \  \:\      /  /\        /  /::\     /  /::\      /__/|  
		      \__\:\    /  /:/       /  /:/\:\   /  /:/\:\    |  |:|  
		  ___ /  /::\  /__/::\      /  /:/~/:/  /  /:/~/:/    |  |:|  
		 /__/\  /:/\:\ \__\/\:\__  /__/:/ /:/  /__/:/ /:/   __|__|:|  
		 \  \:\/:/__\/    \  \:\/\ \  \:\/:/   \  \:\/:/   /__/::::\  
		  \  \::/          \__\::/  \  \::/     \  \::/       ~\~~\:\ 
		   \  \:\          /__/:/    \  \:\      \  \:\         \  \:\
		    \  \:\         \__\/      \  \:\      \  \:\         \__\/
		     \__\/                     \__\/       \__\/              
                                                              

			      ___                       ___           ___     
			     /  /\                     /  /\         /__/\    
			    /  /:/_                   /  /::\       _\_ \:\   
			   /  /:/ /\  ___     ___    /  /:/\:\     /__/\ \:\  
			  /  /:/ /:/ /__/\   /  /\  /  /:/  \:\   _\_ \:\ \:\ 
			 /__/:/ /:/  \  \:\ /  /:/ /__/:/ \__\:\ /__/\ \:\ \:\
			 \  \:\/:/    \  \:\  /:/  \  \:\ /  /:/ \  \:\ \:\/:/
			  \  \::/      \  \:\/:/    \  \:\  /:/   \  \:\ \::/ 
			   \  \:\       \  \::/      \  \:\/:/     \  \:\/:/  
			    \  \:\       \__\/        \  \::/       \  \::/   
			     \__\/                     \__\/         \__\/    
			                                                      
                                                      

[![Build Status](https://api.travis-ci.com/hippylib/hippyflow.svg?branch=main)](https://travis-ci.com/github/hippylib/hippyflow)
[![DOI](https://zenodo.org/badge/301823282.svg)](https://zenodo.org/badge/latestdoi/301823282)
[![License](https://img.shields.io/github/license/hippylib/hippyflow)](./LICENSE.md)
[![Top language](https://img.shields.io/github/languages/top/hippylib/hippyflow)](https://www.python.org)
![Code size](https://img.shields.io/github/languages/code-size/hippylib/hippyflow)
[![Issues](https://img.shields.io/github/issues/hippylib/hippyflow)](https://github.com/hippylib/hippyflow/issues)
[![Latest commit](https://img.shields.io/github/last-commit/hippylib/hippyflow)](https://github.com/hippylib/hippyflow/commits/master)


* This code is used to construct dimension reduced neural network surrogeates for parametric mappings governed by PDEs

* [`hIPPYlib`](https://github.com/hippylib/hippylib) is used for the data generation and construction of model based projectors

* `Tensorflow` and `keras` are used for the construction of neural network surrogates

* [`hessianlearn`](https://github.com/tomoleary/hessianlearn) is used for second order optimization of keras neural network models.


# Model Based Projectors:

`hIPPYflow` implements software infrastructure for input and output dimension reduction strategies for parametric mappings governed by PDEs. Given a parametric PDE Variational Problem implemented in `hIPPYlib` (using `FEniCS` for finite element representation), and a PDE observable, this code automates the construction of dominant subspaces of the input and output for these mappings. 


<p align="center">
	<img src="https://github.com/tomoleary/images/blob/main/hippyflow/parametric_mapping.png" width="75%" /> 
</p>


`hIPPYflow` implements both active subspace (AS) and Karhunen Loeve expansion (KLE) for input dimension reduction. `hIPPYflow` implements proper orthogonal decomposition (POD) for output dimension reduction.

AS computes the dominant eigenvalue-eigenvector pairs of the following operator:
<p align="center">
	<img src="https://github.com/tomoleary/images/blob/main/hippyflow/active_subspace.png" width="70%" /> 
</p>
KLE computes the dominant eigenvalue-eigenvector pairs of the covariance of the  parameter distribution 
<p align="center">
	<img src="https://github.com/tomoleary/images/blob/main/hippyflow/kle.png" width="50%" /> 
</p>
POD computes the dominant eigenvalue-eigenvector pairs of the expectation of the data outer-product matrix:
<p align="center">
	<img src="https://github.com/tomoleary/images/blob/main/hippyflow/pod.png" width="65%" /> 
</p>

These constructs also implement the generation of training data to be used in surrogate construction, as well as projection error tests that exemplify how good the different model projectors are at capturing key information, and help to detect the "intrinsic dimensionality" of the mappings from inputs to outputs.

## Example Usage (reduced basis construction)

* Install [`hIPPYlib`](https://github.com/hippylib/hippylib), set `HIPPYLIB_PATH`, `HIPPYFLOW_PATH` environmental variables.

```python
import dolfin as dl
import ufl
import numpy as np
import sys, os
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
from hippylib import *

sys.path.append(os.environ.get('HIPPYFLOW_PATH'))
from hippyflow import *

# Set up PDE Variational Problem and observable using a function
def build_observable(mesh, **kwargs):
	# Set up the PDE problem in hIPPYlib
	rank = dl.MPI.rank(mesh.mpi_comm())			
	Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
	Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
	Vh = [Vh2, Vh1, Vh2]
	# Initialize Expressions
	f = dl.Constant(0.0)
		
	u_bdr = dl.Expression("x[1]", degree=1)
	u_bdr0 = dl.Constant(0.0)
	bc = dl.DirichletBC(Vh[STATE], u_bdr, u_boundary)
	bc0 = dl.DirichletBC(Vh[STATE], u_bdr0, u_boundary)
	
	def pde_varf(u,m,p):
		return ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - f*p*ufl.dx

	pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

	# Instance observable operator (in this case pointwise observation of state)
	x_targets = np.linspace(0.1,0.9,10)
	y_targets = np.linspace(0.1,0.9,10)
	targets = []
	for xi in x_targets:
		for yi in y_targets:
			targets.append((xi,yi))
	targets = np.array(targets)

	B = assemblePointwiseObservation(Vh[STATE], targets)
	return LinearStateObservable(pde,B)

# Set up mesh
ndim = 2
nx = 10
ny = 10
mesh = dl.UnitSquareMesh(nx, ny)
# Instance observable
observable_kwargs = {} # No kwargs given in this example
observable = build_observable(mesh,**observable_kwargs)

# Instance probability distribution for the parameter
prior = BiLaplacian2D(Vh[PARAMETER],gamma = 0.1, delta = 1.0)

# Instance Active Subspace Operator
AS = ActiveSubspaceProjector(observable,prior)
# Compute and save input reduced basis to file:
AS.construct_input_subspace()

# Instance POD Operator to compute POD basis and training data
POD = PODProjector(observable,prior)
POD.construct_subspace()
output_directory = 'location/for/training/data/'
POD.generate_training_data(output_directory)

```


# Dimension Reduced Neural Network Strategies


* Given information about dominant subspaces of the input and output spaces for the parametric mappings, `hIPPYflow` implements dimension reduced neural network surrogates. These surrogates allow for parsimonious representations of input-output mappings that can achieve good accuracy for very few training data. Few data is a key feature of many high dimensional PDE based inference problems. 

* Neural network models are implemented in `keras`. Training can be handled directly by `keras`, or using second order optimizers implemented in `hessianlearn`.


## Derivative Informed Projected Neural Networks (DIPNets)

* Active subspace decomposition preserves low dimensional geometry of the parametric mapping.

* Geometry preserving dimension reduction allows for efficient parametrization of neural networks that can generalize well given limited training data.

* Useful for outer-loop applications (e.g. uncertainty quantification, Bayesian inverse problems, Bayesian optimal experimental design, optimization under uncertainty, etc.) where repeated evaluation of expensive PDE-based maps is a major computational bottleneck and limitation in practice.

<p align="center">
	<img src="https://github.com/tomoleary/images/blob/main/hippyflow/dipnet.png" width="65%" /> 
</p>

* Using ResNet for nonlinearity allows for adaptive training, and experimentally superior performance, (DIPResNet)

<p align="center">
	<img src="https://github.com/tomoleary/images/blob/main/hippyflow/dipresnet.png" width="65%" /> 
</p>



# References

These publications use the hippyflow library

- \[1\] O'Leary-Roseberry, T., Villa, U., Chen P., Ghattas O.,

[**Derivative-Informed Projected Neural Networks for High-Dimensional Parametric Maps Governed by PDEs**](https://www.sciencedirect.com/science/article/pii/S0045782521005302).
Computer Methods in Applied Mechanics and Engineering. Volume 388, 1 January 2022, 114199.
([Download](https://arxiv.org/pdf/2011.15110.pdf))<details><summary>BibTeX</summary><pre>
@article{OLearyRoseberryVillaChenEtAl2022,
  title={Derivative-informed projected neural networks for high-dimensional parametric maps governed by {PDE}s},
  author={O’Leary-Roseberry, Thomas and Villa, Umberto and Chen, Peng and Ghattas, Omar},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={388},
  pages={114199},
  year={2022},
  publisher={Elsevier}
}
}</pre></details>

- \[2\] O'Leary-Roseberry, T., Du, X., Chaudhuri, A., Martins, J., Willcox, K., Ghattas, O.,
[**Adaptive Projected Residual Networks for Learning Parametric Maps from Sparse Data**](https://arxiv.org/abs/2112.07096).
arXiv:2112.07096.
([Download](https://arxiv.org/pdf/2112.07096.pdf))<details><summary>BibTeX</summary><pre>
@article{OLearyRoseberryDuChaudhuriEtAl2021,
  title={Adaptive Projected Residual Networks for Learning Parametric Maps from Sparse Data},
  author={O'Leary-Roseberry, Thomas and Du, Xiaosong, and Chaudhuri, Anirban, and Martins Joaqium R. R. A., and Willcox, Karen, and Ghattas, Omar},
  journal={arXiv preprint arXiv:2112.07096},
  year={2021}
}
}</pre></details>

