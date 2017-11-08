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
    
    exceldir = select_dir(r'F:\Nellix_chevas\CT_SSDF\SSDF')
    dirsaveIm =  select_dir(r'C:\Users\Maaike\Desktop','D:\Profiles\koenradesma\Desktop')
    
    def __init__(self):
        self.exceldir =  ExcelAnalysisNellix.exceldir
        self.dirsaveIm = ExcelAnalysisNellix.dirsaveIm
        self.folder_analysis = 'Mirthe2'
        self.workbook_analysis = 'storeOutput.xlsx'
        self.patients =['chevas_01', 'chevas_02',	'chevas_03', 'chevas_04',	
                        'chevas_05', 'chevas_06',	'chevas_07', 'chevas_08']
                        #'chevas_09', 'chevas_10', 'chevas_11']
    
        self.distsAll = [] # distances between all stents that were analyzed
        self.distsRelAll = [] # relative from avgreg distance
        self.angsAll = []
        self.angsRelAll = []
        self.tortsAll = []
        self.tortsRelAll = []
        self.colors = itertools.cycle(['#a6cee3','#1f78b4','#b2df8a','#33a02c',
        '#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928'])
        self.markers = itertools.cycle(['o', '^', 's'])
        self.lstyles = itertools.cycle(['-', '--'])
        self.fontsize1 = 14
        self.fontsize2 = 15
        
    def plot_distances_between_points(self, patients=None, analysis='NelNel', rows=[9,10,11], 
                ylim=[0, 32], ylimRel=[-0.5,0.5], saveFig=False):
        """
        Plot relative distance change with respect to distance at avgreg and 
        absolute distance at each phase. 
        Rows is row as in excel sheet; analysis='NelNel' or 'ChimNel'
        """
        # init figure
        f1 = plt.figure(num=3, figsize=(11.6, 4.62)) # 11.6, 4.6
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
        marker = 'o'
        alpha = 0.9
        
        if patients == None:
            patients = self.patients
        
        for patient in patients:
            # read workbooks and sheets
            workbook_stent = os.path.join(self.exceldir,patient,self.folder_analysis, self.workbook_analysis)
            wb = openpyxl.load_workbook(workbook_stent, data_only=True)
            sheetnames = wb.get_sheet_names()
            for sheetname in sheetnames:
                if sheetname.startswith(analysis):
                    sheet = wb.get_sheet_by_name(sheetname)
                    dists, distsRel = readDistancesBetweenPoints(sheet,rows=rows,colStart=1)
                    self.distsAll.append(dists)
                    self.distsRelAll.append(distsRel)
                    if sheetname.startswith('Chim'):
                        shortname = sheetname[4:]
                    else:
                        shortname = sheetname
                    color1 = next(self.colors)
                    # marker = next(self.markers)
                    ls = next(self.lstyles)
                    # plot
                    ax1.plot(xrange, dists, ls=ls, lw=lw, marker=marker, color=color1, 
                            label='%s:%s' % (patient[-2:],shortname), alpha=alpha)
                    ax2.plot(xrange, distsRel, ls=ls, lw=lw, marker=marker, color=color1, 
                            label='%s:%s' % (patient[-2:],shortname), alpha=alpha)
                
        ax2.legend(loc='center right', fontsize=10, numpoints=2, title='Analysis:')
        _initaxis([ax1,ax2])
        
        if saveFig:
            plt.savefig(os.path.join(self.dirsaveIm, 
            'plot_distances_between_points_{}.png'.format(analysis)), papertype='a0', dpi=600)


    def plot_angles_chimney(self, patients=None, analysis='Ang', rows=[4,5,6,7,8], 
                ylim=[0, 45], ylimRel=[-2,2], saveFig=False):
        """ Plot Angulation or Tortuosity. analysis='Ang' or 'Tort'
        """
        # init figure
        f1 = plt.figure(num=3, figsize=(11.6, 4.62)) # 11.6, 4.6
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
        marker = 'o'
        alpha = 0.9
        
        if patients == None:
            patients = self.patients
        
        for patient in patients:
            # read workbooks and sheets
            workbook_stent = os.path.join(self.exceldir,patient,self.folder_analysis, self.workbook_analysis)
            wb = openpyxl.load_workbook(workbook_stent, data_only=True)
            sheetnames = wb.get_sheet_names()
            for sheetname in sheetnames:
                if sheetname.startswith(analysis):
                    sheet = wb.get_sheet_by_name(sheetname)
                    if analysis=='Ang':
                        angs, angsRel = readAnglesChimney(sheet, rows=rows,colStart=1)
                        self.angsAll.append(angs)
                        self.angsRelAll.append(angsRel)
                        y = angs
                        yrel = angsRel
                    elif analysis=='Tort': 
                        torts, tortsRel = readAnglesChimney(sheet, rows=rows,colStart=1)#todo
                        self.tortsAll.append(torts)
                        self.tortsRelAll.append(tortsRel)
                        y = torts
                        yrel = tortsRel
                    shortname = sheetname[-3:] # LRA or RRA or SMA
                    color1 = next(self.colors)
                    # marker = next(self.markers)
                    ls = next(self.lstyles)
                    # plot
                    ax1.plot(xrange, y, ls=ls, lw=lw, marker=marker, color=color1, 
                            label='%s:%s' % (patient[-2:],shortname), alpha=alpha)
                    ax2.plot(xrange, yrel, ls=ls, lw=lw, marker=marker, color=color1, 
                            label='%s:%s' % (patient[-2:],shortname), alpha=alpha)
                
        ax2.legend(loc='center right', fontsize=10, numpoints=2, title='Analysis:')
        _initaxis([ax1,ax2])
        
        if saveFig:
            plt.savefig(os.path.join(self.dirsaveIm, 
            'plot_angles_chimney{}.png'.format(analysis)), papertype='a0', dpi=600)
        
        

def readDistancesBetweenPoints(sheet,rows=[9,10,11],colStart=1):
    """ read distances over cardiac cycle in excel
    """
    # read distances
    rowStart1 = rows[0]-1
    rowStart2 = rows[1]-1
    rowStart3 = rows[2]-1
    # colStart = 1 # B
    dists = sheet.rows[rowStart1][colStart:colStart+10] # B to K, 10 phases 
    dists = [obj.value for obj in dists]
    dists = np.asarray(dists)
    obj = sheet.rows[rowStart2][colStart] # avgreg position
    n1 = obj.value.split(',')
    n1 = float(n1[0]), float(n1[1]), float(n1[2]) # x,y,z
    obj2 = sheet.rows[rowStart3][colStart] # avgreg position
    n2 = obj2.value.split(',')
    n2 = float(n2[0]), float(n2[1]), float(n2[2]) # x,y,z
    # calc distance at mid heart cycle
    vector = PointSet(np.column_stack(n1))-PointSet(np.column_stack(n2))
    avgdist = vector.norm()
    # relative distances from avgreg
    distsRel = dists - avgdist
    
    return dists, distsRel


def readAnglesChimney(sheet,rows=[4,5,6,7,8],colStart=1):
    """ read angles over cardiac cycle in excel
    """
    # read distances
    rowStart1 = rows[0]-1 # dif
    rowStart2 = rows[1]-1 # angles
    rowStart3 = rows[2]-1 # n1
    rowStart4 = rows[3]-1 # n2
    rowStart5 = rows[4]-1 # n3
    
    # colStart = 1 # B
    maxAngleDif = sheet.rows[rowStart1][colStart:colStart+3] # B to D 
    maxAngleDif = [obj.value for obj in maxAngleDif]
    phaseAngleMin = maxAngleDif[1]
    phaseAngleMax = maxAngleDif[2]
    maxAngleDif = maxAngleDif[0]
    # angles
    angs = sheet.rows[rowStart2][colStart:colStart+10] # B to K, 10 phases 
    angs = [180-obj.value for obj in angs] # 70 is scherpere hoek dan 40
    angs = np.asarray(angs)
    # position of the 3 nodes
    obj = sheet.rows[rowStart3][colStart] # avgreg position
    n1 = obj.value.split(',')
    n1 = float(n1[0]), float(n1[1]), float(n1[2]) # x,y,z
    obj2 = sheet.rows[rowStart4][colStart] # avgreg position
    n2 = obj2.value.split(',')
    n2 = float(n2[0]), float(n2[1]), float(n2[2]) # x,y,z
    obj3 = sheet.rows[rowStart5][colStart] # avgreg position
    n3 = obj3.value.split(',')
    n3 = float(n3[0]), float(n3[1]), float(n3[2]) # x,y,z
    # calc angle at mid heart cycle
    vector1 = PointSet(np.column_stack(n1))-PointSet(np.column_stack(n2))
    vector2 = PointSet(np.column_stack(n2))-PointSet(np.column_stack(n3))
    phi = vector1.angle(vector2)
    avgang = phi[0]*180.0/np.pi # radialen to degrees; pi rad = 180 degrees
    # relative angles from avgreg angle
    angsRel = angs - avgang # is bijv. 10 graden voor bijna recht
    
    return angs, angsRel

if __name__ == '__main__':
    
    foo = ExcelAnalysisNellix()
    
    # foo.colors = itertools.cycle(['#a6cee3','#1f78b4','#b2df8a','#33a02c',
    #     '#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928'])
    foo.colors = itertools.cycle(['#6a3d9a']) # set colors if plotting selection of pts
    
    # foo.plot_distances_between_points(patients=['chevas_07'],analysis='NelNel', ylim=[8,17], saveFig=True)
    foo.plot_distances_between_points(patients=['chevas_07'],analysis='ChimNel', ylim=[5,40], saveFig=True)
    # foo.plot_angles_chimney(patients=['chevas_02'],analysis='Ang', rows=[4,5,6,7,8], 
    #     ylim=[0, 45], ylimRel=[-2,2], saveFig=True)
