# Copyright (c) 2020-2022, The University of Texas at Austin 
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
try:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts, amsmath, mathrsfs}')
except:
    print('Error loading latex, will not be used in plots')
    pass

def spectrum_plot(lambdas, axis_label = ['i','$\lambda$','Spectrum'],\
                 ylims = None, out_name = None):
    """
    This is a generic spectrum plot
        
        - :code:`lambdas` - numpy array of eigenvalues
        - :code:`axis_label` - list of three strings: [x label, y label, title]
        - :code:`ylims` - y limits for truncating plot
        - :code:`out_name` - path for saving plot
    """

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
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(axis='both', which='minor', labelsize=25)
    plt.savefig(out_name, bbox_inches = 'tight')

    return fig


def generic_semilogy_plot(xs,ys,axis_label = ['x axis','y axis','My Generic Semilogy Plot'],
                 ylims = None, out_name = None):
    """
    This is a generic semilogy plot
        
        - :code:`xs` - numpy array of x data
        - :code:`ys` - numpy array of y data
        - :code:`axis_label` - list of three strings: [x label, y label, title]
        - :code:`ylims` - y limits for truncating plot
        - :code:`out_name` - path for saving plot
    """
    fig, ax = plt.subplots(figsize=(10,5))

    ax.semilogy(xs, ys)
    ax.set_xlabel(axis_label[0],fontsize = 30)
    for label in ax.get_yticklabels():
        label.set_rotation(90) 

    if ylims is not None:
        ax.set_ylim(ylims)

    ax.set_ylabel(axis_label[1],fontsize = 35)
    ax.set_title(axis_label[2],fontsize = 35)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(axis='both', which='minor', labelsize=25)

    plt.savefig(out_name, bbox_inches = 'tight')

    return fig


def plot_accs_vs_data(data_dictionary,std_data_dictionary = {},\
            axis_label = ['Training Data','Accuracy','Testing Accuracy vs Training Data'],\
                     out_name = 'max_accs',show_plot = True,full_ylim = False,legend_loc = 'best'):
    """
    This plotting function plots accuracy data vs number of training data used during training
        - :code:`data_dictionary` - dictionary of data used for plotting, keys are label names,
            values are n_data, accuracies
        - :code:`std_data_dictionary` - a dictionary of standard deviations used for error bars in plotting
            keys are label names, values are n_data, stds
        - :code:`axis_label` - python list of three strings: [x label, y label, title]
        - :code:`out_name` - string path for where to save the plot
        - :code:`show_plot` - Boolean for showing the plotting
        - :code:`full_ylim` - Boolean to show full y axis, or to truncate based on data
        - :code:`legend_loc` - string for legend_loc passed into matplotlib legend function
    """
    fig = plt.figure(figsize=(10,5))
    plt.gcf().subplots_adjust(bottom=0.15)
    ax = fig.add_subplot(111)
    
    min_y = 1.
    max_y = 1.
    
    for i,key in enumerate(data_dictionary.keys()):
        n_datas,max_accs = data_dictionary[key]
        label = ''
        for piece in key.split('_'):
            label+=piece+' '
        ax.plot(n_datas,max_accs,label = label,color = 'C'+str(i))
        if key in std_data_dictionary.keys():
            _,std = std_data_dictionary[key]
            ax.fill_between(n_datas,max_accs+std,max_accs - std,alpha = 0.2)
        min_y = min(min_y,min(max_accs))
        max_y = max(max_y,max(max_accs))
    
    ax.set_xlabel(axis_label[0],fontsize = 25)

    ax.set_ylabel(axis_label[1],fontsize = 25)
    ax.set_title(axis_label[2],loc = 'center',fontsize = 25)
    if full_ylim:
        ax.set_ylim(bottom = 0.0,top = 1.0+1e-2)
    else:
        ax.set_ylim(bottom = min_y,top = max_y)
    
    
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.legend(loc = legend_loc,fontsize = 20)
    plt.savefig(out_name+'.pdf', bbox_inches = 'tight')
    if show_plot:
        plt.show()


def plot_singular_values_with_std(s,s_std,title = 'Average singular values with std',outname= 'out_plot.pdf',show = False):
    

    # The reduced SVD is factorized in numpy as:

#     print('Error = ',np.linalg.norm(J - U@np.diag(s)@V))
#     print('Error = ',np.linalg.norm(J - (U*s)@V))

    # Plot the singular values

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')


    fig, ax = plt.subplots()
    indices = np.arange(1,len(s)+1)
#     ax.plot(indices,s)
    ax.semilogy(indices,s)
    ax.fill_between(indices,s-s_std,s+s_std,alpha = 0.2)

    ax.set_xlabel('i',fontsize = 20)
    ax.set_ylabel('$\sigma_i$',fontsize = 20)
    ax.set_title(title,fontsize = 20)

    ax.grid()
    plt.tight_layout()
    plt.savefig(outname)
    if show:
        plt.show()


def subspace_angle_video(angleses,keys = None,
                 axis_label = ['i','Angle $(^o)$',('Subspace Angles between $V(m_0)$ and $V(m_{','})$')],
                 out_name = 'subspace_angle_video'):

    matplotlib.use("Agg")
    import matplotlib.animation as manimation

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Subspace angles', artist='hippyflow',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=5, metadata=metadata)


    fig, ax = plt.subplots(figsize=(10,5))
    y_max = np.max([np.max(angles) for angles in angleses])
    y_min = np.max([np.min(angles) for angles in angleses])

    max_index = np.max([len(angles) for angles in angles])

    indices = np.arange(len(angleses[0]))
    with writer.saving(fig, out_name+'.mp4',dpi = 200):
        try:
            from tqdm import tqdm
            for i,angles in enumerate(tqdm(angleses)):
                ax.set_ylim([y_min, y_max])
                ax.set_xlim([0,max_index])
                ax.set_xlabel(axis_label[0],fontsize = 25)
                ax.set_ylabel(axis_label[1],fontsize = 25)
                ax.set_title(axis_label[2][0]+str(i)+axis_label[2][1],fontsize = 25)
                ax.plot(indices, angles)
                writer.grab_frame()
                ax.cla()

        except:
            for i,angles in enumerate(angleses):
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel(axis_label[0],fontsize = 25)
                ax.set_ylabel(axis_label[1],fontsize = 25)
                ax.set_title(axis_label[2][0]+str(i)+axis_label[2][1],fontsize = 25)
                ax.plot(indices, angles)
                writer.grab_frame()
                ax.cla()
