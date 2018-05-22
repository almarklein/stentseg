""" LSPEAS: Script to plot motion during the cardiac cycle 
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

class ExcelMotionAnalysis():
    """ Create graphs from excel data
    """
    
    exceldir = select_dir(r'C:\Users\Maaike\SURFdrive\UTdrive\LSPEAS\Analysis', 
                    r'D:\Profiles\koenradesma\SURFdrive\UTdrive\LSPEAS\Analysis')
    dirsaveIm =  select_dir(r'C:\Users\Maaike\Desktop','D:\Profiles\koenradesma\Desktop')
    
    def __init__(self):
        self.exceldir =  ExcelMotionAnalysis.exceldir
        self.dirsaveIm = ExcelMotionAnalysis.dirsaveIm
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
                        'LSPEAS_023', 
                        'LSPEAS_024', 
                        'LSPEAS_004']
        
        self.relPosAll = [] # deforms-> pdisplacement
        
        self.fontsize1 = 14
        self.fontsize2 = 15
        
    
    def plot_displacement(self, patient='LSPEAS_001', ctcode='discharge',
                        analysis=['ant'], ring='R1', 
                        ylim=[-0.5, 0.5], ylimRel=[0,0.8], saveFig=False):
        """
        Plot relative displacement with respect to position at avgreg and 
        all positions at each phase. 
        Rows is row as in excel sheet; analysis='ant' or 'post' or 'left' or 'right'
        """
        # init figure
        self.f1 = plt.figure(figsize=(11.6, 5.8)) # 11.6, 4.62
        xlabels = ['0', '10', '20', '30', '40', '50', '60', '70','80','90']
        xrange = range(1,1+len(xlabels)) # not start from x=0 to play with margin xlim
        fontsize1 = self.fontsize1
        fontsize2 = self.fontsize2
        
        # init axis
        factor = 1
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, factor]) 
        ax1 = plt.subplot(gs[0])
        plt.xticks(xrange, xlabels, fontsize = fontsize1)
        ax2 = plt.subplot(gs[1])
        plt.xticks(xrange, xlabels, fontsize = fontsize1)
        
        ax1.set_ylabel('Relative position x/y/z (mm)', fontsize=fontsize2)
        ax1.set_ylim(ylim)
        ax1.set_xlim([0.8, len(xlabels)+0.2]) # xlim margins 0.2
        ax1.set_xlabel('Phase in cardiac cycle', fontsize=fontsize2)
        
        ax2.set_ylabel('Displacement 3D (mm)', fontsize=fontsize2) # relative pos wrt avgreg
        ax2.set_ylim(ylimRel)
        ax2.set_xlim([0.8, len(xlabels)*factor+0.7]) # xlim margins 0.2
        ax2.set_xlabel('Phase in cardiac cycle', fontsize=fontsize2)
        
        # plot init
        lw = 1
        # ls = '-'
        # marker = 'o'
        alpha = 0.9
        
        colors = create_iter_colors(type=2)
        markers = create_iter_markers(type=3)
        lstyles = create_iter_ls(type=2)
        
        # read workbook
        sheetname = 'Output_'+ patient[-3:]
        workbook_stent = os.path.join(self.exceldir, self.workbook_stent)
        # read sheet
        wb = openpyxl.load_workbook(workbook_stent, data_only=True)
        sheet = wb.get_sheet_by_name(sheetname)  
        
        for a in analysis:
            shortname = a # e.g. 'ant'
            out = readRingExcel(sheet, ctcode, ring='R1', motion=True, nphases=10) # ant post left right
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
            for i, text in enumerate(['x', 'y', 'z', '3D']): # x,y,z and magnitude
                # for x y z and 3d new color and marker
                color1 = next(colors)
                marker = next(markers)
                ls = next(lstyles)
                if i == 3: # plot once the magnitudes
                    ax2.plot(xrange, vec_rel_displ_magn, ls=ls, lw=lw, marker=marker, color=color1, 
                            label='%s:%s_%s' % (patient[7:],shortname,text), alpha=alpha)
                    break
                ax1.plot(xrange, pdisplacement[:,i], ls=ls, lw=lw, marker=marker, color=color1, 
                        label='%s:%s_%s' % (patient[7:],shortname,text), alpha=alpha)
            
        ax1.legend(loc='upper right', fontsize=10, numpoints=2, title='Analysis:')
        ax2.legend(loc='upper right', fontsize=10, numpoints=2, title='Analysis:')
        _initaxis([ax1,ax2])
    
        if saveFig:
            savename = 'plot_displacement_{}_{}.png'.format(patient[7:],analysis)
            savename = savename.replace('[', '')
            savename = savename.replace(']', '')
            savename = savename.replace('\'', '')
            savename = savename.replace(', ', '_')
            plt.savefig(os.path.join(self.dirsaveIm, savename), papertype='a0', dpi=600) 



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
                                    '#d73027', # 1
                                    '#fc8d59',
                                    #'#fee090', # yellow
                                    '#91bfdb',
                                    '#4575b4' # 5
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
        lstyles = itertools.cycle(['-', '-', '-', '-', '--','--','--','--'])
    
    elif type == 3:
        lstyles = itertools.cycle(['-', '--', '.-'])

    return lstyles



if __name__ == '__main__':
    
    foo = ExcelMotionAnalysis()
    
    patients = None # None = all in self.patients
    
    foo.plot_displacement(patient='LSPEAS_001', ctcode='discharge', 
        analysis=['ant'], ring='R1', ylim=[-1, 1], ylimRel=[0,1], saveFig=False)
    

