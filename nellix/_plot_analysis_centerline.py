""" Author: M.A. Koenrades
Created November 2017
Module to create plots from automated analysis of motion on the centerlines
Reads Excel output
"""
from stentseg.utils import PointSet
import openpyxl
from stentseg.utils.datahandling import select_dir
import sys, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import itertools
from matplotlib import gridspec
from lspeas.analysis.utils_analysis import _initaxis 

class ExcelAnalysisNellix():
    """ Create graphs from excel data
    """
    
    # exceldir = select_dir(r'F:\Nellix_chevas\CT_SSDF\SSDF')
    exceldir = r'E:\Nellix_chevas\CT_SSDF\SSDF_automated'
    dirsaveIm =  select_dir(r'C:\Users\Maaike\Desktop','D:\Profiles\koenradesma\Desktop')
    
    def __init__(self):
        self.exceldir =  ExcelAnalysisNellix.exceldir
        self.dirsaveIm = ExcelAnalysisNellix.dirsaveIm
        self.workbook_analysis = 'ChevasStoreOutput'
        self.patients =['chevas_01', 'chevas_02',	'chevas_03', 'chevas_04',	
                        'chevas_05', 'chevas_06',	'chevas_07', 'chevas_08',
                        'chevas_09', 'chevas_10', 'chevas_11'
                        ]
    
        self.distsAll = [] # distances between all stents that were analyzed
        self.distsRelAll = [] # relative from avgreg distance
        self.angsAll = []
        self.angsRelAll = []
        self.tortsAll = []
        self.tortsRelAll = []
        self.posAll = []
        self.relPosAll = []
        self.fontsize1 = 14
        self.fontsize2 = 15
    
    def plot_displacement(self, patient='chevas_01', analysis=['NelLprox'], rows=[8,9], 
                ylim=[-0.5, 0.5], ylimRel=[0,0.8], saveFig=False):
        """
        Plot relative displacement with respect to position at avgreg and 
        all positions at each phase. 
        Rows is row as in excel sheet; analysis='NelRprox' or 'LRAprox' or 'LRAdist'
        """
        # init figure
        self.f1 = plt.figure(figsize=(11.6, 5.8)) # 11.6, 4.62
        xlabels = ['0', '10', '20', '30', '40', '50', '60', '70','80','90']
        xrange = range(1,1+len(xlabels)) # not start from x=0 to play with margin xlim
        xrange2 = np.asarray([1,2,3,4,5,6,7,8,9,10])
        xrange2 = xrange2 + 0.5
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
        workbook_stent = os.path.join(self.exceldir,patient, self.workbook_analysis+patient[7:]+'.xlsx')
        # read sheet
        wb = openpyxl.load_workbook(workbook_stent, data_only=True)
        sheetnames = wb.get_sheet_names()
        for a in analysis:
            for sheetname in sheetnames:
                i = 0
                if sheetname.startswith('Motion_{}'.format(a) ):
                    sheet = wb.get_sheet_by_name(sheetname)  
                    shortname = sheetname[7:] # e.g. 'RRAprox'
                    
                    pavg, pphases, pdisplacement = readDisplacement(sheet,rows=rows,colStart=1)
                    self.relPosAll.append(pdisplacement)
                    
                    # get vector magnitudes between phases
                    vec_between_phases = []
                    for i, p in enumerate(pphases[:-1]):
                        vec_between_phases.append(p-pphases[i+1])
                    # add vector between last and first phase
                    vec_last_to_first = pphases[-1] - pphases[0]
                    vec_between_phases.append(vec_last_to_first)
                    # to array for magnitude
                    vec_between_phases = np.asarray(vec_between_phases)
                    vec_between_phases_magn = np.linalg.norm(vec_between_phases, axis=1) # vector magnitude for each phase
                    
                    # plot
                    for i, text in enumerate(['x', 'y', 'z', '3D']): # x,y,z and magnitude
                        color1 = next(colors)
                        marker = next(markers)
                        ls = next(lstyles)
                        if i == 3: # plot once the magnitudes
                            ax2.plot(xrange2, vec_between_phases_magn, ls=ls, lw=lw, marker=marker, color=color1, 
                                    label='%s:%s_%s' % (patient[7:],shortname,text), alpha=alpha)
                            break
                        # for x y and z new color
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
        
    def plot_distances_between_points(self, patients=None, analysis='NelNel', rows=[9,5], 
                ylim=[0, 32], ylimRel=[-0.5,0.5], saveFig=False):
        """
        Plot relative distance change with respect to distance at avgreg and 
        absolute distance at each phase. 
        Rows is row as in excel sheet; analysis='NelNel' or 'ChimNel'
        """
        # init figure
        self.f1 = plt.figure(figsize=(11.6, 5.8)) # 11.6, 4.62
        xlabels = ['0', '10', '20', '30', '40', '50', '60', '70','80','90']
        xrange = range(1,1+len(xlabels)) # not start from x=0 to play with margin xlim
        fontsize1 = self.fontsize1
        fontsize2 = self.fontsize2
        
        # init axis
        factor = 1.33
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, factor]) # plot right wider
        ax1 = plt.subplot(gs[0])
        plt.xticks(xrange, xlabels, fontsize = fontsize1)
        ax2 = plt.subplot(gs[1])
        plt.xticks(xrange, xlabels, fontsize = fontsize1)
        
        ax1.set_ylabel('Distance (mm)', fontsize=fontsize2) # absolute distances
        ax1.set_ylim(ylim)
        ax1.set_xlim([0.8, len(xlabels)+0.2]) # xlim margins 0.2
        ax1.set_xlabel('Phase in cardiac cycle', fontsize=fontsize2)
        
        ax2.set_ylabel('Relative distance change (mm)', fontsize=fontsize2) # relative dist from avgreg
        ax2.set_ylim(ylimRel)
        ax2.set_xlim([0.8, len(xlabels)*factor+0.2]) # xlim margins 0.2; # longer for legend
        ax2.set_xlabel('Phase in cardiac cycle', fontsize=fontsize2)
        
        # plot init
        lw = 1
        # ls = '-'
        # marker = 'o'
        alpha = 0.9
        
        colors = create_iter_colors()
        markers = create_iter_markers()
        lstyles = create_iter_ls(type=1)
        
        if patients == None:
            patients = self.patients
        
        for patient in patients:
            # read workbook
            workbook_stent = os.path.join(self.exceldir,patient, self.workbook_analysis+patient[7:]+'.xlsx')
            # read sheet
            wb = openpyxl.load_workbook(workbook_stent, data_only=True)
            sheetnames = wb.get_sheet_names()
            for sheetname in sheetnames:
                sheet = None
                if analysis == 'ChimNel':
                    if (sheetname.startswith('Dist_RRA') or
                            sheetname.startswith('Dist_LRA') or
                            sheetname.startswith('Dist_SMA')):
                        sheet = wb.get_sheet_by_name(sheetname)  
                        shortname = sheetname[5:-1] # e.g. 'RRA_Nel' 
                elif analysis == 'NelNel':
                    if sheetname.startswith('Dist_Nel'):
                        sheet = wb.get_sheet_by_name(sheetname)
                        shortname = 'NelNel'
                if not sheet is None:        
                    dists, distsRel = readDistancesBetweenPoints(sheet,rows=rows,colStart=1)
                    self.distsAll.append(dists)
                    self.distsRelAll.append(distsRel)
                    color1 = next(colors)
                    marker = next(markers)
                    ls = next(lstyles)
                    # plot
                    ax1.plot(xrange, dists, ls=ls, lw=lw, marker=marker, color=color1, 
                            label='%s:%s' % (patient[7:],shortname), alpha=alpha)
                    ax2.plot(xrange, distsRel, ls=ls, lw=lw, marker=marker, color=color1, 
                            label='%s:%s' % (patient[7:],shortname), alpha=alpha)
                
        ax2.legend(loc='upper right', fontsize=10, numpoints=2, title='Analysis:')
        _initaxis([ax1,ax2])
        
        if saveFig:
            plt.savefig(os.path.join(self.dirsaveIm, 
            'plot_distances_between_points_{}.png'.format(analysis)), papertype='a0', dpi=600)


    def plot_angles_chimney(self, patients=None, analysis='Ang', rows=[7,10,6], 
                ylim=[0, 47], ylimRel=[-2,2], saveFig=False):
        """ Plot Angulation or Tortuosity. analysis='Ang' or 'Tort'
        """
        # init figure
        self.f2 = plt.figure(figsize=(11.6, 5.8)) # 11.6, 4.62
        xlabels = ['0', '10', '20', '30', '40', '50', '60', '70','80','90']
        xrange = range(1,1+len(xlabels)) # not start from x=0 to play with margin xlim
        fontsize1 = self.fontsize1
        fontsize2 = self.fontsize2
        
        # init axis
        factor = 1.33
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, factor]) # plot right wider
        ax1 = plt.subplot(gs[0])
        plt.xticks(xrange, xlabels, fontsize = fontsize1)
        ax2 = plt.subplot(gs[1])
        plt.xticks(xrange, xlabels, fontsize = fontsize1)
        
        if analysis=='Ang':
            ax1.set_ylabel('Angulation ($\degree$)', fontsize=fontsize2) # absolute
        elif analysis=='Tort':
            ax1.set_ylabel('Tortuosity (AU)', fontsize=fontsize2) # absolute
        ax1.set_ylim(ylim)
        ax1.set_xlim([0.8, len(xlabels)+0.2]) # xlim margins 0.2
        ax1.set_xlabel('Phase in cardiac cycle', fontsize=fontsize2)
        
        if analysis=='Ang':
            ax2.set_ylabel('Relative angulation change ($\degree$)', fontsize=fontsize2) # relative dist from avgreg
        elif analysis=='Tort':
            ax2.set_ylabel('Relative tortuosity change (AU)', fontsize=fontsize2) # relative dist from avgreg
        ax2.set_ylim(ylimRel)
        ax2.set_xlim([0.8, len(xlabels)*factor+0.2]) # xlim margins 0.2; # longer for legend
        ax2.set_xlabel('Phase in cardiac cycle', fontsize=fontsize2)
        
        # plot init
        lw = 1
        # ls = '-'
        # marker = 'o'
        alpha = 0.9
        
        colors = create_iter_colors()
        markers = create_iter_markers()
        lstyles = create_iter_ls(type=1)
        
        if patients == None:
            patients = self.patients
        
        for patient in patients:
            # read workbooks and sheets
            workbook_stent = os.path.join(self.exceldir,patient, self.workbook_analysis+patient[7:]+'.xlsx')
            wb = openpyxl.load_workbook(workbook_stent, data_only=True)
            sheetnames = wb.get_sheet_names()
            for sheetname in sheetnames:
                if sheetname.startswith(analysis):
                    sheet = wb.get_sheet_by_name(sheetname)
                    if analysis=='Ang':
                        angs, angsRel = readAnglesChimney(sheet, rows=rows,colStart=1, analysis=analysis)
                        self.angsAll.append(angs)
                        self.angsRelAll.append(angsRel)
                        y = angs
                        yrel = angsRel
                    elif analysis=='Tort': 
                        torts, tortsRel = readAnglesChimney(sheet, rows=rows,colStart=1, analysis=analysis)
                        self.tortsAll.append(torts)
                        self.tortsRelAll.append(tortsRel)
                        y = torts
                        yrel = tortsRel
                    shortname = sheetname[-3:] # LRA or RRA or SMA
                    color1 = next(colors)
                    marker = next(markers)
                    ls = next(lstyles)
                    # plot
                    ax1.plot(xrange, y, ls=ls, lw=lw, marker=marker, color=color1, 
                            label='%s:%s' % (patient[7:],shortname), alpha=alpha)
                    ax2.plot(xrange, yrel, ls=ls, lw=lw, marker=marker, color=color1, 
                            label='%s:%s' % (patient[7:],shortname), alpha=alpha)
                
        ax2.legend(loc='upper right', fontsize=10, numpoints=2, title='Analysis:')
        _initaxis([ax1,ax2])
        
        if saveFig:
            plt.savefig(os.path.join(self.dirsaveIm, 
            'plot_angles_chimney{}.png'.format(analysis)), papertype='a0', dpi=600)
        
        

def readDisplacement(sheet,rows=[8, 9],colStart=1, nphases=10):
    """ Read position of a point at avgreg and at each phase in cycle
    rows as excel rows
    """ 
    rowStart1 = rows[0]-1
    rowStart2 = rows[1]-1
    # read mean position avgreg
    obj = sheet.rows[rowStart2][colStart]
    pavg = obj.value.strip('()')
    pavg = pavg.split(',')
    pavg = float(pavg[0]), float(pavg[1]), float(pavg[2]) # x,y,z
    pavg = np.asarray(pavg)
    
    def convert_str_tuple(coord):
        return float(coord[0]), float(coord[1]), float(coord[2])
    
    objs = sheet.rows[rowStart1][colStart:colStart+nphases]
    pphases = [obj.value.strip('()') for obj in objs]
    pphases = [coord.split(',') for coord in pphases]
    pphases = [convert_str_tuple(coord) for coord in pphases] 
    pphases = np.asarray(pphases)
    
    pdisplacement = pphases - pavg
    
    return pavg, pphases, pdisplacement

def readDistancesBetweenPoints(sheet,rows=[9,5],colStart=1, nphases=10):
    """ read distances over cardiac cycle in excel
    """
    # read distances
    rowStart1 = rows[0]-1
    rowStart2 = rows[1]-1
    dists = sheet.rows[rowStart1][colStart:colStart+nphases] # B to K, 10 phases 
    dists = [obj.value for obj in dists]
    dists = np.asarray(dists)
    obj = sheet.rows[rowStart2][colStart] # avgreg distance
    avgdist = obj.value # get distance at mid heart cycle
    # relative distances from avgreg
    distsRel = dists - avgdist
    
    return dists, distsRel


def readAnglesChimney(sheet,rows=[7,10,6],colStart=1, analysis='Ang'):
    """ read angles over cardiac cycle in excel
    """
    # read distances
    rowStart1 = rows[0]-1 # dif
    rowStart2 = rows[1]-1 # angles
    rowStart3 = rows[2]-1 # ang mid cycle
    
    # colStart = 1 # B
    maxAngleDif = sheet.rows[rowStart1][colStart:colStart+3] # B to D 
    maxAngleDif = [obj.value for obj in maxAngleDif]
    phaseAngleMin = maxAngleDif[1]
    phaseAngleMax = maxAngleDif[2]
    maxAngleDif = maxAngleDif[0]
    # angles / tort
    angs = sheet.rows[rowStart2][colStart:colStart+10] # B to K, 10 phases 
    if analysis == 'Ang':
        angs = [180-obj.value for obj in angs] # so that 0 is straight: 70 is scherpere hoek dan 40
        angs = np.asarray(angs)
        # get angle mid cycle
        obj = sheet.rows[rowStart3][colStart]
        avgang = 180-obj.value
    elif analysis == 'Tort':
        angs = [obj.value for obj in angs] 
        angs = np.asarray(angs)
        # get tort mid cycle
        obj = sheet.rows[rowStart3][colStart]
        avgang = obj.value
    
    # relative angles from avgreg angle (or tort)
    angsRel = angs - avgang # change with respect to mid cycle angle
    
    return angs, angsRel

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
    
    foo = ExcelAnalysisNellix()
    
    # patients=['chevas_09', 'chevas_09_thin']
    patients = None # None = all in self.patients
    
    # # Distances
    # foo.plot_distances_between_points(patients=patients,analysis='NelNel', ylim=[8,17], saveFig=True) 
    # foo.plot_distances_between_points(patients=patients,analysis='ChimNel', ylim=[5,40], saveFig=True)
    # 
    # # Angles
    # foo.plot_angles_chimney(patients=patients,analysis='Ang', rows=[7,10,6], ylim=[0, 47], ylimRel=[-2,2], saveFig=True)
    # 
    # # Tortuosity
    # foo.plot_angles_chimney(patients=patients,analysis='Tort', rows=[4,7,3], ylim=[0.99, 1.11], ylimRel=[-0.01,0.01], saveFig=True)
    
    # Displacement
    patient = 'chevas_07'
    foo.plot_displacement(patient=patient, analysis=['RRAprox','NelRprox'], 
                rows=[8,9], ylim=[-0.5, 0.5], ylimRel=[0,0.8], saveFig=True)
    
    
    
    