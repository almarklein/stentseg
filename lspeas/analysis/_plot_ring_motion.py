""" LSPEAS: Script to analyze and plot motion during the cardiac cycle from
excel sheet pulsatility_expansion
Author: M.A. Koenrades. Created 2018.
"""

from lspeas.analysis.utils_analysis import readRingExcel, _initaxis
from stentseg.utils import PointSet
import openpyxl
from stentseg.utils.datahandling import select_dir
import sys, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np 
import itertools
from matplotlib import gridspec
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from stentseg.utils.picker import pick3d
from stentseg.utils.visualization import DrawModelAxes, show_ctvolume, plot_points
from stentseg.utils import PointSet, _utils_GUI, visualization
from lspeas.utils.vis import showModel3d

class ExcelMotionAnalysis():
    """ Create graphs from excel data
    """
    
    exceldir = select_dir(r'C:\Users\Maaike\SURFdrive\UTdrive\LSPEAS\Analysis', 
                    r'D:\Profiles\koenradesma\SURFdrive\UTdrive\LSPEAS\Analysis')
    dirsaveIm =  select_dir(r'C:\Users\Maaike\Desktop','D:\Profiles\koenradesma\Desktop')
    basedir = select_dir(r'D:\LSPEAS\LSPEAS_ssdf', 
                     r'F:\LSPEAS_ssdf_backup',r'G:\LSPEAS_ssdf_backup')
    
    def __init__(self):
        self.exceldir =  ExcelMotionAnalysis.exceldir
        self.dirsaveIm = ExcelMotionAnalysis.dirsaveIm
        self.basedir = ExcelMotionAnalysis.basedir
        self.workbook_stent = 'LSPEAS_pulsatility_expansion_avgreg_subp_v15.6.xlsx'
        #self.workbook_variables = 'LSPEAS_Variables.xlsx'
        self.patients =['LSPEAS_001', 
                        'LSPEAS_002',	
                        'LSPEAS_003', 
                        'LSPEAS_005',	
                        'LSPEAS_008', 
                        'LSPEAS_009',	
                        'LSPEAS_011', 
                        'LSPEAS_015',	
                        'LSPEAS_017',	
                        'LSPEAS_018', 
                        'LSPEAS_020', 
                        'LSPEAS_021', 
                        'LSPEAS_022', 
                        'LSPEAS_019',
                        'LSPEAS_025', 
                        #'LSPEAS_023', 
                        #'LSPEAS_024', 
                        #'LSPEAS_004'
                        ]
        
        self.relPosAll = [] # deforms-> pdisplacement
        self.points_plotted = []
        
        self.fontsize1 = 14
        self.fontsize2 = 15
    
    def plot_displacement(self, patient='LSPEAS_001', ctcodes=['discharge', '24months'],
                        analysis=['ant'], ring='R1', 
                        ylim=[-0.5, 0.5], ylimRel=[0,0.8], saveFig=False):
        """
        Plot relative displacement with respect to position at avgreg and 
        all positions at each phase. 
        Rows is row as in excel sheet; 
        analysis='ant' or 'post' or 'left' or 'right';
        ctcodes= [discharge, 1month, 6months, 12months, 24months]
        """
        # init figure
        self.f1 = plt.figure(figsize=(12.5, 9.1)) # 11.6, 5.8 or 4.62
        xlabels = ['0', '10', '20', '30', '40', '50', '60', '70','80','90']
        xrange = range(1,1+len(xlabels)) # not start from x=0 to play with margin xlim
        fontsize1 = self.fontsize1
        fontsize2 = self.fontsize2
        
        # init axis
        factor = 1
        gs = gridspec.GridSpec(2, 3, width_ratios=[factor, factor, factor]) 
        ax1 = plt.subplot(gs[0])
        plt.xticks(xrange, xlabels, fontsize = fontsize1)
        ax2 = plt.subplot(gs[1])
        plt.xticks(xrange, xlabels, fontsize = fontsize1)
        ax3 = plt.subplot(gs[2])
        plt.xticks(xrange, xlabels, fontsize = fontsize1)
        ax4 = plt.subplot(gs[3])
        plt.xticks(xrange, xlabels, fontsize = fontsize1)
        
        ax1.set_ylabel('Relative position x (mm)', fontsize=fontsize2)
        ax1.set_ylim(ylim)
        # ax1.set_yticks(np.arange(ylim[0], ylim[1], 0.2))
        ax1.set_xlim([0.8, len(xlabels)+0.2]) # xlim margins 0.2
        ax1.set_xlabel('Phase in cardiac cycle', fontsize=fontsize2)
        
        ax2.set_ylabel('Relative position y (mm)', fontsize=fontsize2)
        ax2.set_ylim(ylim)
        # ax2.set_yticks(np.arange(ylim[0], ylim[1], 0.2))
        ax2.set_xlim([0.8, len(xlabels)+0.2]) # xlim margins 0.2
        ax2.set_xlabel('Phase in cardiac cycle', fontsize=fontsize2)
        
        ax3.set_ylabel('Relative position z (mm)', fontsize=fontsize2)
        ax3.set_ylim(ylim)
        # ax3.set_yticks(np.arange(ylim[0], ylim[1], 0.2))
        ax3.set_xlim([0.8, len(xlabels)+0.2]) # xlim margins 0.2
        ax3.set_xlabel('Phase in cardiac cycle', fontsize=fontsize2)
        
        ax4.set_ylabel('Displacement 3D (mm)', fontsize=fontsize2) # relative pos wrt avgreg pos
        ax4.set_ylim(ylimRel)
        # ax4.set_yticks(np.arange(ylimRel[0], ylimRel[1], 0.2))
        ax4.set_xlim([0.8, len(xlabels)+0.2]) # len(xlabels)*factor+0.7
        ax4.set_xlabel('Phase in cardiac cycle', fontsize=fontsize2)
        
        # plot init
        lw = 1
        # ls = '-'
        # marker = 'o'
        alpha = 0.9
        
        colors = create_iter_colors(type=2)
        colors3d = create_iter_colors(type=3)
        markers = create_iter_markers(type=3)
        lstyles = create_iter_ls(type=3)
        
        # read workbook
        sheetname = 'Output_'+ patient[-3:]
        workbook_stent = os.path.join(self.exceldir, self.workbook_stent)
        # read sheet
        wb = openpyxl.load_workbook(workbook_stent, data_only=True)
        sheet = wb.get_sheet_by_name(sheetname)  
        
        ctnames = []
        for ctcode in ctcodes:
            ctname = ctcode_printname(ctcode)
            ctnames.append(ctname)
            color1 = next(colors)
            color3d = next(colors3d)
            marker = next(markers)
            out = readRingExcel(sheet, patient,ctcode, ring=ring, motion=True, nphases=10) # ant post left right
            #todo: built in check in readRingExcel for ant post
            for a in analysis:
                shortname = a # e.g. 'ant'
                if a == 'ant':
                    pavg, pdisplacement = out[0], out[1]
                elif a == 'post':
                    pavg, pdisplacement = out[2], out[3]
                elif a == 'left':
                    pavg, pdisplacement = out[4], out[5]
                elif a == 'right':
                    pavg, pdisplacement = out[6], out[7]
                #pphases = pdisplacement + pavg
                
                self.relPosAll.append([pavg,pdisplacement])
                
                # get vector magnitudes of rel displacement
                vec_rel_displ_magn = np.linalg.norm(pdisplacement, axis=1) # vector magnitude for each phase
                
                # plot
                ls = next(lstyles)
                for i, text in enumerate(['x', 'y', 'z', '3D']): # x,y,z and magnitude
                    if i == 0:
                        ax1.plot(xrange, pdisplacement[:,i], ls=ls, lw=lw, marker=marker, color=color1, 
                            label='%s-%s' % (ctname,shortname), alpha=alpha)
                    elif i == 1:
                        ax2.plot(xrange, pdisplacement[:,i], ls=ls, lw=lw, marker=marker, color=color1, 
                            label='%s-%s' % (ctname,shortname), alpha=alpha)
                    elif i == 2:
                        ax3.plot(xrange, pdisplacement[:,i], ls=ls, lw=lw, marker=marker, color=color1, 
                            label='%s-%s' % (ctname,shortname), alpha=alpha)
                    elif i == 3:
                        ax4.plot(xrange, vec_rel_displ_magn, ls=ls, lw=lw, marker=marker, color=color3d, 
                                label='%s-%s' % (ctname,shortname), alpha=alpha)
            
        ax1.legend(loc='upper right', fontsize=10, numpoints=2, title='Analysis:')
        ax2.legend(loc='upper right', fontsize=10, numpoints=2, title='Analysis:')
        ax3.legend(loc='upper right', fontsize=10, numpoints=2, title='Analysis:')
        ax4.legend(loc='upper right', fontsize=10, numpoints=2, title='Analysis:')
        _initaxis([ax1,ax2,ax3,ax4])
    
        if saveFig:
            savename = 'plot_displacement_{}_{}_{}.png'.format(patient[8:],ctnames,analysis)
            savename = savename.replace('[', '')
            savename = savename.replace(']', '')
            savename = savename.replace('\'', '')
            savename = savename.replace(', ', '_')
            plt.savefig(os.path.join(self.dirsaveIm, savename), papertype='a0', dpi=600) 

    
    def show_displacement_3d(self,patient='LSPEAS_001', ctcode='discharge',
                        analysis=['ant', 'post'], ring='R1', 
                        showVol='MIP', showModelavgreg=True, **kwargs):
        """ Show the displacement of a ring point or points in 3d space with 
        respect to the position at mid cardiac cycle = avgreg
        """
        # load vol for patient and show 3d
        self.f, self.a1, self.label = showModel3d(self.basedir, patient, ctcode, 
                cropname='ring', showVol=showVol,showmodel=showModelavgreg, **kwargs)
        # read workbook for point positions
        sheetname = 'Output_'+ patient[-3:]
        workbook_stent = os.path.join(self.exceldir, self.workbook_stent)
        # read sheet
        wb = openpyxl.load_workbook(workbook_stent, data_only=True)
        sheet = wb.get_sheet_by_name(sheetname) 
        out = readRingExcel(sheet, patient,ctcode, ring=ring, motion=True, nphases=10) # ant post left right
        #todo: built in check in readRingExcel for ant post
        # show for given points by analysis and ring
        for a in analysis:
            if a == 'ant':
                pavg, pdisplacement = out[0], out[1]
            elif a == 'post':
                pavg, pdisplacement = out[2], out[3]
            elif a == 'left':
                pavg, pdisplacement = out[4], out[5]
            elif a == 'right':
                pavg, pdisplacement = out[6], out[7]
            pphases = pdisplacement + pavg
            
            # show these point positions
            points = plot_points(pphases, mc='y', ax=self.a1)
            self.points_plotted.append(points)
            
            #show the points at mid cycle = avgreg
            plot_points(pavg, mc='b', ax=self.a1)
        
        ctname = ctcode_printname(ctcode)
        savename = 'displacement3d_{}_{}_{}_{}.png'.format(patient[8:],ctname,ring,analysis)
        savename = savename.replace('[', '')
        savename = savename.replace(']', '')
        savename = savename.replace('\'', '')
        savename = savename.replace(', ', '_')
        print(savename)
        

def create_iter_colors(type=1):
    if type == 1:
        colors = itertools.cycle([  
                                '#a6cee3', # 1
                                '#fb9a99', # 2 
                                '#33a02c', # 3
                                '#fdbf6f', # 4 
                                '#1f78b4', # 5
                                '#e31a1c', # 6
                                '#b2df8a', # 7 
                                '#cab2d6', # 8 
                                '#ff7f00', # 9
                                '#6a3d9a', # 10
                                '#ffff99', # 11
                                '#b15928'])# 12
    
    elif type == 2:
        colors =  itertools.cycle([
                                    '#d73027', # 1 red
                                    '#91bfdb', # blue
                                    '#fee090', # yellow
                                    #'#4575b4' # 5
                                    ])
    elif type == 3:
        colors =  itertools.cycle([
                                    '#fc8d59', # orange
                                    '#4575b4', # 5 dark blue
                                    'k'
                                    ])
    
    return colors


def create_iter_markers(type=1):
    if type == 1:
        markers = itertools.cycle([
                                    'o', 'o', 'o', 'o',
                                    '^', '^', '^', '^',
                                    's', 's', 's', 's',
                                    'd', 'd', 'd', 'd'  ])
    if type == 2:
        markers = itertools.cycle([
                                    'o', 'o', 'o',
                                    '^', '^', '^',
                                    's', 's', 's'  ])
    
    
    if type == 3:
        markers = itertools.cycle(['o', '^', 's','d'])
    
    return markers


def create_iter_ls(type=1):
    if type == 1:
        lstyles = itertools.cycle(['-'])
    
    elif type == 2:
        lstyles = itertools.cycle(['-', '-', '-', '--','--','--'])
    
    elif type == 3:
        lstyles = itertools.cycle(['-', '--']) #-.

    return lstyles

def ctcode_printname(ctcode):
    name = ''
    if ctcode == 'discharge':
        name = 'D'
    elif ctcode == '1month':
        name = '1M'
    elif ctcode == '6months':
        name = '6M'
    elif ctcode == '12months':
        name = '12M'
    elif ctcode == '24months':
        name = '24M'
    return name
    

if __name__ == '__main__':
    
    foo = ExcelMotionAnalysis()
    
    patients = foo.patients
    
    foo.plot_displacement(patient='LSPEAS_020', ctcodes=['discharge','24months'], 
        analysis=['post', 'right'], ring='R1', ylim=[-0.65, 0.75], ylimRel=[0,1], saveFig=True)
    
    # foo.show_displacement_3d('LSPEAS_020', '24months',
    #                         analysis=['post', 'right'], ring='R1', 
    #                         clim=(0,2600),showVol='MIP', showModelavgreg=True)
    
