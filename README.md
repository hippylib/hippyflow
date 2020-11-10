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
			                                                      
                                                      

* This code is used to construct dimension reduced neural network surrogeates for parametric mappings governed by PDEs

* [`hIPPYlib`](https://github.com/hippylib/hippylib) is used for the data generation and construction of model based projectors

* `Tensorflow` and `keras` are used for the construction of neural network surrogates

* [`hessianlearn`](https://github.com/tomoleary/hessianlearn) is used for second order optimization of keras neural network models.


## Model Based Projectors:

* `hippyflow` implements software infrastructure for input and output dimension reduction strategies for parametric mappings governed by PDEs. Given a parametric PDE Variational Problem implemented in `hIPPYlib` (using `FEniCS` for finite element representation), and a PDE observable, this code automates the construction of dominant subspaces of the input and output for these mappings. `hippyflow` implements both active subspace (AS) and Karhunen Loeve expansion (KLE) for input dimension reduction. `hippyflow` implements proper orthogonal decomposition (POD) for output dimension reduction.

* These constructs also implement the generation of training data to be used in surrogate construction, as well as projection error tests that exemplify how good the different model projectors are at capturing key information, and help to detect the "intrinsic dimensionality" of the mappings from inputs to outputs.

## Dimension Reduced Neural Network Strategies

* Given information about dominant subspaces of the input and output spaces for the parametric mappings, `hippyflow` implements dimension reduced neural network surrogates. These surrogates allow for parsimonious representations of input-output mappings that can achieve good accuracy for very few training data. Few data is a key feature of many high dimensional PDE based inference problems. 

* Neural network models are implemented in `keras`. Training can be handled directly by `keras`, or using second order optimizers implemented in `hessianlearn`.