# Copyright (c) 2020-2021, The University of Texas at Austin 
# & Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYflow package. For more information see
# https://github.com/hippylib/hippyflow/
#
# hIPPYflow is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.


   
#!/bin/bash

set -ev 


PYTHON=python3

DOCKER="docker run --rm -v $(pwd):/home/fenics/hippylib -w"

${DOCKER} /home/fenics/hippylib/hippylib/test $IMAGE "$PYTHON -m unittest discover -v"
${DOCKER} /home/fenics/hippylib/hippylib/test $IMAGE "$PYTHON -m unittest discover -v -p 'ptest_*' "

# ${DOCKER} /home/fenics/hippylib $IMAGE "$PYTHON -c 'import hippyflow' && cd hippyflow/test/ && $PYTHON test_derivativeSubspace.py"
# If the paths are set as environmental variables then I do not need to import hippyflow or hippylib here
${DOCKER} /home/fenics/hippylib $IMAGE "$PYTHON -c  cd hippyflow/test/ && $PYTHON test_derivativeSubspace.py"
