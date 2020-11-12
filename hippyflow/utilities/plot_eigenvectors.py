# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin 
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


import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as cls
import dolfin as dl
import numpy as np
from matplotlib import animation

"""
Plotting utilities for notebooks
"""

def _mesh2triang(mesh):
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

def _mplot_cellfunction(cellfn):
    C = cellfn.array()
    tri = _mesh2triang(cellfn.mesh())
    return plt.tripcolor(tri, facecolors=C)

def _mplot_function(f, vmin, vmax, logscale):
    mesh = f.function_space().mesh()
    if (mesh.geometry().dim() != 2):
        raise AttributeError('Mesh must be 2D')
    # DG0 cellwise function
    if f.vector().size() == mesh.num_cells():
        C = f.vector().get_local()
        if logscale:
            return plt.tripcolor(_mesh2triang(mesh), C, vmin=vmin, vmax=vmax, norm=cls.LogNorm() )
        else:
            return plt.tripcolor(_mesh2triang(mesh), C, vmin=vmin, vmax=vmax)
    # Scalar function, interpolated to vertices
    elif f.value_rank() == 0:
        C = f.compute_vertex_values(mesh)
        if logscale:
            return plt.tripcolor(_mesh2triang(mesh), C, vmin=vmin, vmax=vmax, norm=cls.LogNorm() )
        else:
            return plt.tripcolor(_mesh2triang(mesh), C, shading='gouraud', vmin=vmin, vmax=vmax)
    # Vector function, interpolated to vertices
    elif f.value_rank() == 1:
        w0 = f.compute_vertex_values(mesh)
        if (len(w0) != 2*mesh.num_vertices()):
            raise AttributeError('Vector field must be 2D')
        X = mesh.coordinates()[:, 0]
        Y = mesh.coordinates()[:, 1]
        U = w0[:mesh.num_vertices()]
        V = w0[mesh.num_vertices():]
        C = np.sqrt(U*U+V*V)
        return plt.quiver(X,Y,U,V, C, units='x', headaxislength=7, headwidth=7, headlength=7, scale=4, pivot='middle')
    
def plot(obj, colorbar=True, subplot_loc=None, mytitle=None, show_axis='off', vmin=None, vmax=None, logscale=False, cmap=None):
    """
    Plot a generic dolfin object (if supported)
    """
    if subplot_loc is not None:
        plt.subplot(subplot_loc)
#    plt.gca().set_aspect('equal')
    if isinstance(obj, dl.Function):
        pp = _mplot_function(obj, vmin, vmax, logscale)
    elif isinstance(obj, dl.MeshFunction):
        pp = _mplot_cellfunction(obj)
    elif isinstance(obj, dl.Mesh):
        if (obj.geometry().dim() != 2):
            raise AttributeError('Mesh must be 2D')
        pp = plt.triplot(_mesh2triang(obj), color='#808080')
        colorbar = False
    else:
        raise AttributeError('Failed to plot %s'%type(obj))
    
    plt.axis(show_axis)
        
    if colorbar:
        plt.colorbar(pp, fraction=.1, pad=0.2)
    else:
        plt.gca().set_aspect('equal')
        
    if mytitle is not None:
        plt.title(mytitle, fontsize=30)
        
    if cmap:
        plt.set_cmap(cmap)
    else:
        plt.set_cmap('viridis')
        
    return pp
        

def plot_pts(points, values, colorbar=True, subplot_loc=None, mytitle=None, show_axis='on', vmin=None, vmax=None, xlim=(0,1), ylim=(0,1),cmap=None):
    """
    Plot a cloud of points
    """  
    if subplot_loc is not None:
        plt.subplot(subplot_loc)
    
    pp = plt.scatter(points[:,0], points[:,1], c=values.get_local(), marker=",", s=20, vmin=vmin, vmax=vmax)
        
    plt.axis(show_axis)
        
    if colorbar:
        plt.colorbar(pp, fraction=.1, pad=0.2)
    else:
        plt.gca().set_aspect('equal')
        
    if mytitle is not None:
        plt.title(mytitle, fontsize=20)
        
    if xlim is not None:
        plt.xlim(xlim)
        
    if ylim is not None:
        plt.ylim(ylim)
        
    if cmap:
        plt.set_cmap(cmap)
    else:
        plt.set_cmap('viridis')
        
    return pp



def plot_eigenvector(Vh, U, mytitle, which = 0, cmap = None,outname = 'eigenvectors.pdf'):
    """
    Plot specified vectors in a :code:MultiVector
    """
    plt.figure(figsize= (9,6.5) )
    
    title_stamp = mytitle + " {0}" 
    u = dl.Function(Vh)

    assert which < U.nvec()
    if (U[which])[0] >= 0:
        s = 1./U[which].norm("linf")
    else:
        s = -1./U[which].norm("linf")
    u.vector().zero()
    u.vector().axpy(s, U[which])
    plot(u, mytitle=title_stamp.format(which+1), vmin=-1, vmax=1, cmap = cmap)


    plt.savefig(outname)
    