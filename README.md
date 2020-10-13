# hippyflow
* Integrating hippylib with tensorflow for qoi surrogate construction

* Assumes that the environmental variable `HIPPYLIB_BASE_DIR` has been set

With conda

* `conda create -n hippyflow -c uvilla -c conda-forge fenics==2019.1.0 tensorflow matplotlib scipy jupyter mpi4py keras`


Neural network training is handled by keras / second order methods in hessianlearn

* Assume that the environmental variable `HESSIANLEARN_PATH` has been set



To Do:

pygalmesh mesh generator, currently I am only using this inside of the ice_problem/ directory, but eventually I will have to export paths to link to pygalmesh across many different examples. For the time being it is clashing with the very sensitive `hippyflow` anaconda environment that I have so I am creating another conda environment just for the mesh generation. The build is as follows, must link to the `spec_file.txt` which is currently inside of the ice_problem directory. Do this build in the directory that you want pygalmesh in.:

* `conda create --name pygalmesh --file path/to/spec_file.txt`

* `git clone https://github.com/nschloe/pygalmesh.git`

* `export EIGEN_INCLUDE_DIR=/opt/anaconda3/envs/pygalmesh/include/eigen3`

* `cd pygalmesh`

* `python setup.py install`

* `export PYGALMESH_PATH=/path/to/pygalmesh`

* `conda install -c conda-forge h5py`