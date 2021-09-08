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


## Model Based Projectors:

* `hIPPYflow` implements software infrastructure for input and output dimension reduction strategies for parametric mappings governed by PDEs. Given a parametric PDE Variational Problem implemented in `hIPPYlib` (using `FEniCS` for finite element representation), and a PDE observable, this code automates the construction of dominant subspaces of the input and output for these mappings. `hIPPYflow` implements both active subspace (AS) and Karhunen Loeve expansion (KLE) for input dimension reduction. `hIPPYflow` implements proper orthogonal decomposition (POD) for output dimension reduction.

* These constructs also implement the generation of training data to be used in surrogate construction, as well as projection error tests that exemplify how good the different model projectors are at capturing key information, and help to detect the "intrinsic dimensionality" of the mappings from inputs to outputs.

## Dimension Reduced Neural Network Strategies

* Given information about dominant subspaces of the input and output spaces for the parametric mappings, `hIPPYflow` implements dimension reduced neural network surrogates. These surrogates allow for parsimonious representations of input-output mappings that can achieve good accuracy for very few training data. Few data is a key feature of many high dimensional PDE based inference problems. 

* Neural network models are implemented in `keras`. Training can be handled directly by `keras`, or using second order optimizers implemented in `hessianlearn`.


# References

These publications use the hippyflow library

- \[1\] O'Leary-Roseberry, T., Villa, U., Chen P., Ghattas O.,
[**Derivative-Informed Projected Neural Networks for High-Dimensional Parametric Maps Governed by PDEs**](https://arxiv.org/abs/2011.15110).
arXiv:2011.15110.
([Download](https://arxiv.org/pdf/2011.15110.pdf))<details><summary>BibTeX</summary><pre>
@article{o2020derivative,
  title={Derivative-Informed Projected Neural Networks for High-Dimensional Parametric Maps Governed by PDEs},
  author={O'Leary-Roseberry, Thomas and Villa, Umberto and Chen, Peng and Ghattas, Omar},
  journal={arXiv preprint arXiv:2011.15110},
  year={2020}
}
}</pre></details>

