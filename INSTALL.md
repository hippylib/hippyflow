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
			                                                      
                                                      

* PDE data generation is handled by `FEniCS` and `hIPPYlib`. For this [`hIPPYlib`](https://github.com/hippylib/hippylib) must be installed. 

With conda

* `conda create -n hippyflow -c uvilla -c conda-forge fenics==2019.1.0 tensorflow matplotlib scipy jupyter mpi4py keras`

* `pip install pympler` pympler is used to check memory usage during subspace creation

Assumes that the environmental variables `HIPPYLIB_PATH` and `HIPPYFLOW_PATH` have been set.

* `export HIPPYLIB_PATH=path/to/hippylib`
* `export HIPPYFLOW_PATH=path/to/hippyflow`



Neural network training is handled by `keras` / second order methods in hessianlearn

When using second order optimizers, the code assumes that the environmental variable `HESSIANLEARN_PATH` has been set to the path to [`hessianlearn`](https://github.com/tomoleary/hessianlearn).

* `export HESSIANLEARN_PATH=path/to/hessianlearn`

In some cases (such as on Unix clusters) the dependencies for `hIPPYlib` and `tensorflow` clash. In this case different conda environments may be needed for the data generation phase and the neural network training phase. These two phases can be easily de-coupled in this case. 

On Mac, there has been no issue creating a singular conda environment to handle all dependencies.