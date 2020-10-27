# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

# UPDATE THE COPYRIGHT

import dolfin as dl
from hippylib import *

class BlockVector(object):
    """
    A class to store multiple vectors.
    """
    def __init__(self, *args):
        
        if len(args) == 1:
            #Copy constructor
            other = args[0]
            self.nv = other.nv
            self.data = [None]*self.nv
            for i in range(self.nv):
                self.data[i] = other.data[i].copy()
                
        elif len(args) == 2:
            Vh = args[0]
            nv = args[1]
            self.nv = nv
            self.data = [None]*self.nv
            if isinstance(Vh, dl.FunctionSpace):
                for i in range(self.nv):
                    self.data[i] = dl.Function(Vh).vector()
            else:
                for i in range(self.nv):
                    self.data[i] = dl.Function(Vh[i]).vector()
            
    def initialize(self, Vh):
        if isinstance(Vh, dl.FunctionSpace):
            for ii in range(self.nv):
                self.data[ii].init(Vh.mesh().mpi_comm(), Vh.dofmap().ownership_range())
        else:
            for ii in range(self.nv):
                self.data[ii].init(Vh[ii].mesh().mpi_comm(), Vh[ii].dofmap().ownership_range())
            
        
    def randn_perturb(self,std_dev):
        """
        Add a random perturbation :math:`\eta_i \sim \mathcal{N}(0, \mbox{std_dev}^2 I)`
        to each of the snapshots.
        """
        for d in self.data:
            parRandom.normal_perturb(std_dev, d)

    
    def axpy(self, a, other):
        """
        Compute :code:`x = x + a*other` snapshot per snapshot.
        """
        assert self.nv == other.nv
        for i in range(self.nv):
            self.data[i].axpy(a,other.data[i])
        
    def zero(self):
        """
        Zero out each subvector.
        """
        for d in self.data:
            d.zero()
                        
    def __imul__(self, alpha):
        """
        Scale by scalar
        """
        for d in self.data:
            d *= alpha
        return self
    
    def copy(self):
        """
        Return a copy 
        """
        return BlockVector(self)
    
    def export(self,Vh, fid, xname):
        for xi in self.data:
            xfun = vector2Function(xi,Vh, name=xname)
            fid << xfun
