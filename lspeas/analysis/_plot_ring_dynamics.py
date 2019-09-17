""" PLOT LSPEAS RING DYNAMICS: Script to plot data collected in _get_ring_dynamics.py
Copyright 2019, Maaike A. Koenrades
"""
from lspeas.analysis.utils_analysis import _initaxis
from lspeas.analysis._get_ring_dynamics import print_stats_var_over_time_
import openpyxl
from stentseg.utils.datahandling import select_dir
import sys, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np 
import itertools
from matplotlib import gridspec
import visvis as vv
import scipy
from scipy import io


class PlotRingDynamics():
    """ Read the stored mat files
    """
    filedir = select_dir(r'C:\Users\Maaike\SURFdrive\UTdrive\LSPEAS\Analysis\Ring motion\matfiles_from_readsheets_python', 
                    r'D:\Profiles\koenradesma\SURFdrive\UTdrive\LSPEAS\Analysis\Ring motion\matfiles_from_readsheets_python')
    dirsaveIm =  select_dir(r'C:\Users\Maaike\Desktop','D:\Profiles\koenradesma\Desktop')
    
    
    def __init__(self):
        self.filedir =  PlotRingDynamics.filedir
        self.dirsaveIm = PlotRingDynamics.dirsaveIm
        self.folderringdynamics = '_get_ring_dynamics'
        self.folderringmotion = '_plot_ring_motion'
        
        self.fontsize1 = 17 # 
        self.fontsize2 = 16 # 
        self.fontsize3 = 13 # legend title
        self.fontsize4 = 10 # legend contents
        self.fontsize5 = 14 # xticks
        self.fontsize6 = 15 # ylabel
        
        self.colorsdirections = [              # from phantom paper in vivo plots
                            'k',
                            '#d73027', # 1 red
                            '#fc8d59', # orange
                            '#91bfdb', # blue
                            '#4575b4' # 5
                            ]


    def plot_displacement_mean_over_time(self, ylim=[0, 2], 
                    saveFig=False, showlegend=True):
        """ Plot mean displacement rings for full ring and all 4 directions using the 
        mean of the patients
        """
        
        # init figure
        f1 = plt.figure(figsize=(10, 9)) # from 11.6 to 10 to place legend
        xlabels = ['D', '1M', '6M', '12M', '24M']
        xrange = range(1,1+len(xlabels)) # not start from x=0 to play with margin xlim
        
        # init axis
        ax1 = f1.add_subplot(3,2,1)
        plt.xticks(xrange, xlabels, fontsize = self.fontsize5)
        ax2 = f1.add_subplot(3,2,2)
        plt.xticks(xrange, xlabels, fontsize = self.fontsize5)
        ax3 = f1.add_subplot(3,2,3)
        plt.xticks(xrange, xlabels, fontsize = self.fontsize5)
        ax4 = f1.add_subplot(3,2,4)
        plt.xticks(xrange, xlabels, fontsize = self.fontsize5)
        ax5 = f1.add_subplot(3,2,5)
        plt.xticks(xrange, xlabels, fontsize = self.fontsize5)
        ax6 = f1.add_subplot(3,2,6)
        plt.xticks(xrange, xlabels, fontsize = self.fontsize5)
        # ax7 = f1.add_subplot(4,2,7)
        # plt.xticks(xrange, xlabels, fontsize = self.fontsize5)
        # ax8 = f1.add_subplot(4,2,8)
        # plt.xticks(xrange, xlabels, fontsize = self.fontsize5)
        
        axes = [ax1, ax2, ax3, ax4, ax5, ax6] #, ax7, ax8]
        
        yname2 = 'mm'
        yname = 'Displacement'
        
        for ax in axes:
            ax.set_ylabel('{} ({})'.format(yname, yname2), fontsize=self.fontsize6)
            ax.set_ylim(ylim)
            ax.set_xlim([0.8, len(xlabels)+0.2]) # xlim margins 0.2
        
        # set markers
        markers = ['p','D', '^', 'o', 's']
        colorsdirections = self.colorsdirections
        # capsize = 8
        lss = ['--', '-','-','-','-']
        # lw = 1
        # lw2 = 1.1
        # alpha = 0.7
        
        Qnames = ['Full ring', 'QP','QA','QL','QR']
        Qs =['','Q1','Q3','Q2','Q4']
        Rs = ['R1', 'R2']
        
        filedir = os.path.join(self.filedir, self.folderringdynamics, 'Displacement')
        
        def plot_displacement_line_errorbar(filedir, name, ax, Q, R, xrange, labelname, ls, 
                    marker, color, alpha=0.7, capsize=8, lw=1, lw2=1.1):
            """
            """
            filename = 'Displacement{}{}_{}_pts'.format(Q,R, name)
            matdict = scipy.io.loadmat(os.path.join(filedir, filename+'.mat'))
            var = matdict[filename]
            # plot
            ax.plot(xrange, np.nanmean(var, axis=0), ls=ls, lw=lw, marker=marker, color=color, 
                    label=labelname, alpha=alpha)
            ax.errorbar(xrange, np.nanmean(var, axis=0), yerr=np.nanstd(var, axis=0), fmt=None,
                         ecolor=color, capsize=capsize, elinewidth=lw2, capthick=lw2)
        
        # read mat files
        for i, labelname in enumerate(Qnames):
            R = Rs[0]
            plot_displacement_line_errorbar(filedir, 'x', ax1, Qs[i], R, xrange, labelname, lss[i], 
                                markers[i], colorsdirections[i])
            plot_displacement_line_errorbar(filedir, 'y', ax3, Qs[i], R, xrange, labelname, lss[i], 
                                markers[i], colorsdirections[i])
            plot_displacement_line_errorbar(filedir, 'z', ax5, Qs[i], R, xrange, labelname, lss[i], 
                                markers[i], colorsdirections[i])
            R = Rs[1]
            plot_displacement_line_errorbar(filedir, 'x', ax2, Qs[i], R, xrange, labelname, lss[i], 
                                markers[i], colorsdirections[i])
            plot_displacement_line_errorbar(filedir, 'y', ax4, Qs[i], R, xrange, labelname, lss[i], 
                                markers[i], colorsdirections[i])
            plot_displacement_line_errorbar(filedir, 'z', ax6, Qs[i], R, xrange, labelname, lss[i], 
                                markers[i], colorsdirections[i])
            
            
        if showlegend:
            ax2.legend(loc='upper right', fontsize=self.fontsize3, numpoints=1, title='Legend')
            
        _initaxis(axes, axsize=self.fontsize5)
        
        if saveFig:
            plt.savefig(os.path.join(self.dirsaveIm, 
            'plot_ring_quadrants_displacement.png'), papertype='a0', dpi=600)

    def plot_meancurvaturechange(self, ylim=[0, 0.1], saveFig=False, showlegend=True):
        """
        """
        # init figure
        f1 = plt.figure(figsize=(10, 4.6)) # from 11.6 to 10 to place legend
        xlabels = ['D', '1M', '6M', '12M', '24M']
        xrange = range(1,1+len(xlabels)) # not start from x=0 to play with margin xlim
        
        # init axis
        ax1 = f1.add_subplot(1,2,1)
        plt.xticks(xrange, xlabels, fontsize = self.fontsize5)
        ax2 = f1.add_subplot(1,2,2)
        plt.xticks(xrange, xlabels, fontsize = self.fontsize5)

        axes = [ax1, ax2]
        
        yname2 = 'cm$^{-1}$'
        yname = 'Curvature change'
        
        for ax in axes:
            ax.set_ylabel('{} ({})'.format(yname, yname2), fontsize=self.fontsize6)
            ax.set_ylim(ylim)
            ax.set_xlim([0.8, len(xlabels)+0.2]) # xlim margins 0.2
        
        # set markers
        markers = ['p','D', '^', 'o', 's']
        colorsdirections = self.colorsdirections
        lss = ['--', '-','-','-','-']
        
        Qnames = ['Full ring', 'QP','QA','QL','QR']
        Qs =['','Q1','Q3','Q2','Q4']
        Rs = ['R1', 'R2']
        
        filedir = os.path.join(self.filedir, self.folderringdynamics, 'Curvature')
        
        def plot_curvaturechange_line_errorbar(filedir, name, ax, Q, R, xrange, labelname, ls, 
                    marker, color, alpha=0.7, capsize=8, lw=1, lw2=1.1):
            """
            """
            filename = 'Curvature{}{}_{}_pts'.format(Q,R, name)
            matdict = scipy.io.loadmat(os.path.join(filedir, filename+'.mat'))
            var = matdict[filename]
            # plot
            ax.plot(xrange, np.nanmean(var, axis=0), ls=ls, lw=lw, marker=marker, color=color, 
                    label=labelname, alpha=alpha)
            ax.errorbar(xrange, np.nanmean(var, axis=0), yerr=np.nanstd(var, axis=0), fmt=None,
                         ecolor=color, capsize=capsize, elinewidth=lw2, capthick=lw2)
        
        # read mat files
        for i, labelname in enumerate(Qnames):
            R = Rs[0]
            plot_curvaturechange_line_errorbar(filedir, 'meanchange', ax1, Qs[i], R, xrange, labelname, lss[i], 
                                markers[i], colorsdirections[i])
            R = Rs[1]
            plot_curvaturechange_line_errorbar(filedir, 'meanchange', ax2, Qs[i], R, xrange, labelname, lss[i], 
                                markers[i], colorsdirections[i])
            
        if showlegend:
            ax2.legend(loc='upper right', fontsize=self.fontsize3, numpoints=1, title='Legend')
            
        _initaxis(axes, axsize=self.fontsize5)
        
        if saveFig:
            plt.savefig(os.path.join(self.dirsaveIm, 
            'plot_ring_quadrants_meanchangecurvature.png'), papertype='a0', dpi=600)
        
        
    
def read_var_dynamics_over_time(filedir, folder, analysisname, typename, dec=1, stats=True, showvar=True):
    """
    analysisname: type of analysis in mat filename, e.g. Displacement
    typename: x, y, z, 3d or mean, max or maxchange, maxchangeP etc
    Q: Quartile name Q1, Q2, Q3, Q4 or ''
    R: 'R1' or R2 or R1R2
    folder: _get_ring_dynamics or _plot_ring_motion
    """
    if analysisname == 'Distances':
        QRs = ['R1R2','Q1R1R2','Q2R1R2','Q3R1R2','Q4R1R2']
    else:
        QRs = ['R1','Q1R1','Q2R1','Q3R1','Q4R1', 'R2','Q1R2','Q2R2','Q3R2','Q4R2']
    filedir = os.path.join(filedir, folder, analysisname)
    vars = {}
    varslist = []
    for QR in QRs:
        filename = '{}{}_{}_pts'.format(analysisname, QR, typename)
        matdict = scipy.io.loadmat(os.path.join(filedir, filename+'.mat'))
        var = matdict[filename]
        vars[filename] = var
        varslist.append(var)
        if stats:
            print_stats_var_mean_over_time_(var, varname=filename, dec=dec, showvar=showvar)
    if stats:
        filename = '{}{}_{}_pts'.format(analysisname, 'Overall', typename)
        var = np.asarray(varslist)
        print_stats_var_mean_over_time_(var, varname=filename, dec=dec, showvar=False)
            
    return vars

def print_stats_var_mean_over_time_(varOverTime_pts, varname='varOverTime_pts', dec=1, 
            showvar=True, median=False):
    """ To print stats for paper
    varOverTime_pts is an array of size 15x5 (pts x timepoints)
    dec to set number of decimals
    print avgerage over time, i.e., from 15x5 to 1 mean value instead of per time point (1x5)
    """
    if showvar:
        print(varOverTime_pts)
    print(varname)
    
    if median:
        if dec == 3:
            print('Overall median, Q1, Q3: {:.3f} [{:.3f}, {:.3f}]'.format(
                                                np.nanmedian(varOverTime_pts),
                                                np.nanpercentile(varOverTime_pts, 25),
                                                np.nanpercentile(varOverTime_pts, 75)
                                                ))
        else:
            print('Overall median, Q1, Q3: {:.1f} [{:.1f}, {:.1f}]'.format(
                                                np.nanmedian(varOverTime_pts),
                                                np.nanpercentile(varOverTime_pts, 25),
                                                np.nanpercentile(varOverTime_pts, 75)
                                                ))
    else:
        if dec == 3:
            print('Overall average±std, min, max: {:.3f} ± {:.3f} ({:.3f}-{:.3f})'.format(
                                                np.nanmean(varOverTime_pts),
                                                np.nanstd(varOverTime_pts),
                                                np.nanmin(varOverTime_pts),
                                                np.nanmax(varOverTime_pts)
                                                ))
        else:
            print('Overall average±std, min, max: {:.1f} ± {:.1f} ({:.1f}-{:.1f})'.format(
                                                np.nanmean(varOverTime_pts),
                                                np.nanstd(varOverTime_pts),
                                                np.nanmin(varOverTime_pts),
                                                np.nanmax(varOverTime_pts)
                                                ))
    print()

if __name__ == '__main__':
    
    foo = PlotRingDynamics()
    
    ## Ring motion manuscript 
    # plot displacement x,y,z per quadrant, overall and per ring
    if False:
        foo.plot_displacement_mean_over_time(ylim=[0, 1.31], saveFig=True, showlegend=False)
    # plot curvature quadrants and overall per ring
    if False:
        foo.plot_meancurvaturechange(ylim=[0, 0.05], saveFig=True, showlegend=False)
    
    
    ## Stats for manuscript
    # get displacement stats overall
    if False:
        varOverTime_pts = read_var_dynamics_over_time(foo.filedir,foo.folderringdynamics, 'Displacement', 'y')
    # distance r1r2
    if False:
        varOverTime_pts = read_var_dynamics_over_time(foo.filedir,foo.folderringdynamics, 'Distances', 'mean')
    
    