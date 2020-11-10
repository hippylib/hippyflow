# Copyright (c) 2020, The University of Texas at Austin 
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

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
from pylab import *
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')


def spectrum_plot(lambdas,keys = None,
                 axis_label = ['i','$\lambda$','Spectrum'],
                 ylims = None,
                 out_name = None):
    # Truncate above 1e-10
    lambdas = lambdas[lambdas > 1e-10] 

    # try:
    fig, ax = plt.subplots(figsize=(10,5))
    indices = np.arange(lambdas.shape[0])
    ax.semilogy(indices, lambdas)
    ax.set_xlabel(axis_label[0],fontsize = 30)
    for label in ax.get_yticklabels():
        label.set_rotation(90) 

    if ylims is not None:
        ax.set_ylim(ylims)

    ax.set_ylabel(axis_label[1],fontsize = 35)
    ax.set_title(axis_label[2],fontsize = 35)
    # legend = ax.legend(loc='upper right',fontsize = 25)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(axis='both', which='minor', labelsize=25)

    plt.savefig(out_name, bbox_inches = 'tight')

    # 	if out_name is not None:
    # 		pp = PdfPages(out_name+'.pdf')
    # 		pp.savefig(fig)
    # 		pp.close()
    # try:
    #     plt.show()
    # except:
    #     pass
    return fig


def generic_semilogy_plot(xs,ys,keys = None,
                 axis_label = ['x axis','y axis','My Generic Semilogy Plot'],
                 ylims = None,
                 out_name = None):

    # try:
    fig, ax = plt.subplots(figsize=(10,5))

    ax.semilogy(xs, ys)
    ax.set_xlabel(axis_label[0],fontsize = 30)
    for label in ax.get_yticklabels():
        label.set_rotation(90) 

    if ylims is not None:
        ax.set_ylim(ylims)

    ax.set_ylabel(axis_label[1],fontsize = 35)
    ax.set_title(axis_label[2],fontsize = 35)
    # legend = ax.legend(loc='upper right',fontsize = 25)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(axis='both', which='minor', labelsize=25)

    plt.savefig(out_name, bbox_inches = 'tight')

    #   if out_name is not None:
    #       pp = PdfPages(out_name+'.pdf')
    #       pp.savefig(fig)
    #       pp.close()
    # try:
    #     plt.show()
    # except:
    #     pass
    return fig