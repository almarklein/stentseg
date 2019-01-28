""" LSPEAS: Script to analyze and plot motion during the cardiac cycle from
excel sheet or from the dynamic models
Author: M.A. Koenrades. Created 2018.
"""

from lspeas.analysis.utils_analysis import readRingExcel, _initaxis, cols2num
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

class MotionAnalysis():
    """ Create graphs from excel data or from the segmented models
    """
    
    exceldir = select_dir(r'C:\Users\Maaike\SURFdrive\UTdrive\LSPEAS\Analysis', 
                    r'D:\Profiles\koenradesma\SURFdrive\UTdrive\LSPEAS\Analysis')
    dirsaveIm =  select_dir(r'C:\Users\Maaike\Desktop','D:\Profiles\koenradesma\Desktop')
    basedir = select_dir(r'D:\LSPEAS\LSPEAS_ssdf', 
                     r'F:\LSPEAS_ssdf_backup',r'G:\LSPEAS_ssdf_backup')
    
    def __init__(self):
        self.exceldir =  MotionAnalysis.exceldir
        self.dirsaveIm = MotionAnalysis.dirsaveIm
        self.basedir = MotionAnalysis.basedir
        self.workbook_stent = 'LSPEAS_pulsatility_expansion_avgreg_subp_v2.0.xlsx'
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
                        'LSPEAS_019', 
                        'LSPEAS_020', 
                        'LSPEAS_021', 
                        'LSPEAS_022', 
                        'LSPEAS_025', 
                        #'LSPEAS_023', 
                        #'LSPEAS_024', 
                        #'LSPEAS_004'
                        ]
        
        self.relPosDispNodes = [] # [pavg,pdisplacement]
        self.points_plotted = []
        self.analysis = {}
        self.modelssdf = [] # for s of models (spine point analysis)
        
        self.fontsize1 = 17 # 14
        self.fontsize2 = 17 # 15
    
    def _init_fig_plot_displacement(self, ylim, ylimRel):
        """
        """
        # init figure
        self.f1 = plt.figure(figsize=(12.5, 9.1)) # 11.6, 5.8 or 4.62
        # xlabels = ['0', '10', '20', '30', '40', '50', '60', '70','80','90']
        xlabels = ['0', '', '20', '', '40', '', '60', '','80','']
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
        
        self.ax1, self.ax2, self.ax3, self.ax4 = ax1, ax2, ax3, ax4
        self.xrange = xrange
    
    def plot_displacement(self, patient='LSPEAS_001', ctcodes=['discharge', '24months'],
                        analysis=['ant'], ring='R1', 
                        ylim=[-0.5, 0.5], ylimRel=[0,0.8], saveFig=False):
        """
        Plot relative displacement with respect to position at avgreg and 
        all positions at each phase.  for 1 patient
        Rows is row as in excel sheet; 
        analysis='ant' or 'post' or 'left' or 'right';
        ctcodes= [discharge, 1month, 6months, 12months, 24months]
        """
        # init figure
        self._init_fig_plot_displacement(ylim, ylimRel)
        ax1, ax2, ax3, ax4 = self.ax1, self.ax2, self.ax3, self.ax4
        xrange = self.xrange
        
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
            #marker = next(markers)
            ls = next(lstyles)
            out = readRingExcel(sheet, patient,ctcode, ring=ring, motion=True, nphases=10) 
            # out = ant post left right
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
                
                self.relPosDispNodes.append([pavg,pdisplacement])
                
                # get vector magnitudes of rel displacement for each phase
                vec_rel_displ_magn = np.linalg.norm(pdisplacement, axis=1)
                
                # plot
                marker = next(markers)
                
                # plot x
                ax1.plot(xrange, pdisplacement[:,0], ls=ls, lw=lw, marker=marker, color=color1, 
                    label='%s-%s' % (ctname,shortname), alpha=alpha)
                # plot y
                ax2.plot(xrange, pdisplacement[:,1], ls=ls, lw=lw, marker=marker, color=color1, 
                    label='%s-%s' % (ctname,shortname), alpha=alpha)
                # plot z
                ax3.plot(xrange, pdisplacement[:,2], ls=ls, lw=lw, marker=marker, color=color1, 
                    label='%s-%s' % (ctname,shortname), alpha=alpha)
                # plot 3d vector magnitude
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
    
    
    def plot_displacement_selected_points(self, colortype=2, colortype3d=3,
                    markertype=3, linetype=3,
                    ylim=[-0.5, 0.5], ylimRel=[0,0.8], saveFig=False):
        """ plot displacement for point that were selected in 
        show_displacement_3d_select_points
        """
        # init figure
        self._init_fig_plot_displacement(ylim, ylimRel)
        ax1, ax2, ax3, ax4 = self.ax1, self.ax2, self.ax3, self.ax4
        xrange = self.xrange
        
        # plot init
        lw = 1
        alpha = 0.9
        
        colors = create_iter_colors(type=colortype)
        colors3d = create_iter_colors(type=colortype3d)
        markers = create_iter_markers(type=markertype)
        lstyles = create_iter_ls(type=linetype)
        
        # loop through stored node analysis
        for key in self.analysis.keys(): # per scan
            # get order for plot since dict has no order
            order = []
            key2s = []
            for key2 in self.analysis[key].keys(): # per analyzed point
                [i, selectn1,pdisplacement] = self.analysis[key][key2]
                order.append(i)
                key2s.append(key2)
            for j in sorted(order): # 1,2,3..
                index = order.index(j)
                key2 = key2s[index]
                [i, selectn1,pdisplacement] = self.analysis[key][key2]
                color1 = next(colors)
                color3d = next(colors3d)
                ls = next(lstyles)
                
                # get vector magnitudes of rel displacement for each phase
                vec_rel_displ_magn = np.linalg.norm(pdisplacement, axis=1)
                
                # plot
                marker = next(markers)
                
                # plot x
                ax1.plot(xrange, pdisplacement[:,0], ls=ls, lw=lw, marker=marker, color=color1, 
                    label='%s' % (key2), alpha=alpha)
                # plot y
                ax2.plot(xrange, pdisplacement[:,1], ls=ls, lw=lw, marker=marker, color=color1, 
                    label='%s' % (key2), alpha=alpha)
                # plot z
                ax3.plot(xrange, pdisplacement[:,2], ls=ls, lw=lw, marker=marker, color=color1, 
                    label='%s' % (key2), alpha=alpha)
                # plot 3d vector magnitude
                ax4.plot(xrange, vec_rel_displ_magn, ls=ls, lw=lw, marker=marker, color=color3d, 
                        label='%s' % (key2), alpha=alpha)
        
        #ax1.legend(loc='upper right', fontsize=10, numpoints=2, title='Analysis:')
        #ax2.legend(loc='upper right', fontsize=10, numpoints=2, title='Analysis:')
        ax3.legend(loc='upper right', fontsize=13, numpoints=2, title='Legend:')
        ax4.legend(loc='upper right', fontsize=10, numpoints=2, title='Analysis:')
        _initaxis([ax1,ax2,ax3,ax4])
    
        if saveFig:
            savename = 'plot_displacement.png'
            savename = savename.replace('[', '')
            savename = savename.replace(']', '')
            savename = savename.replace('\'', '')
            savename = savename.replace(', ', '_')
            plt.savefig(os.path.join(self.dirsaveIm, savename), papertype='a0', dpi=600)
        
    
    def show_displacement_3d(self,patient='LSPEAS_001', ctcode='discharge',
                        analysis=['R1ant', 'R1post'], 
                        showVol='MIP', showModelavgreg=True, locationcheck=True, **kwargs):
        """ Show the displacement of a ring point or points in 3d space with 
        respect to the position at mid cardiac cycle = avgreg
        """
        # load vol for patient and show 3d
        self.f, self.a1, self.label, self.s = showModel3d(self.basedir, patient, ctcode, 
                cropname='ring', showVol=showVol,showmodel=showModelavgreg, **kwargs)
        # read workbook for point positions
        sheetname = 'Output_'+ patient[-3:]
        workbook_stent = os.path.join(self.exceldir, self.workbook_stent)
        # read sheet
        wb = openpyxl.load_workbook(workbook_stent, data_only=True)
        sheet = wb.get_sheet_by_name(sheetname) 
        # ring 1
        out = readRingExcel(sheet, patient,ctcode, ring='R1', motion=True, 
                nphases=10, locationcheck=locationcheck) # ant post left right
        # ring 2 (or distal zenith points R1 for z stent)
        out2 = readRingExcel(sheet, patient,ctcode, ring='R2', motion=True, 
                nphases=10, locationcheck=locationcheck) # ant post left right
        #todo: built in check in readRingExcel for ant post
        # show for given points by analysis and ring
        for a in analysis:
            if a == 'R1ant':
                pavg, pdisplacement = out[0], out[1]
            elif a == 'R1post':
                pavg, pdisplacement = out[2], out[3]
            elif a == 'R1left':
                pavg, pdisplacement = out[4], out[5]
            elif a == 'R1right':
                pavg, pdisplacement = out[6], out[7]
            elif a == 'R2ant':
                pavg, pdisplacement = out2[0], out2[1]
            elif a == 'R2post':
                pavg, pdisplacement = out2[2], out2[3]
            elif a == 'R2left':
                pavg, pdisplacement = out2[4], out2[5]
            elif a == 'R2right':
                pavg, pdisplacement = out2[6], out2[7]
            
            pphases = pdisplacement + pavg
            
            # show these point positions
            points = plot_points(pphases, mc='y', ax=self.a1)
            self.points_plotted.append(points)
            
            #show the points at mid cycle = avgreg
            plot_points(pavg, mc='b', ax=self.a1)
        
        ctname = ctcode_printname(ctcode)
        savename = 'displacement3d_{}_{}_{}.png'.format(patient[8:],ctname,analysis)
        savename = savename.replace('[', '')
        savename = savename.replace(']', '')
        savename = savename.replace('\'', '')
        savename = savename.replace(', ', '_')
        print(savename)
    
    def show_displacement_3d_select_points(self,patient='LSPEAS_001', ctcode='discharge',
                        showVol='MIP', showModelavgreg=True, basedir=None, cropname='ring',
                        analysis=['Anaconda R1ant', 'Anaconda R1post'], **kwargs):
        """ Same as for show_displacement_3d but user can select which nodes from 3d image
        The position and deforms are stored
        analysis = labels for points selected (in order) to store in self for 
        later plotting with legend
        """
        # load vol for patient and show 3d
        if basedir is None: # use default from init
            basedir = self.basedir
        self.f, self.a1, self.label, self.s = showModel3d(basedir, patient, ctcode, 
                cropname=cropname, showVol=showVol,showmodel=showModelavgreg,
                graphname='model', **kwargs)
        
        # combine models if multiple were stored in ssdf
        cnt = 0
        for key in dir(self.s):
            if key.startswith('model'):
                if cnt == 0: # first get graph
                    graph = self.s[key]
                else:
                    graph.add_nodes_from(self.s[key].nodes(data=True)) # also attributes
                    graph.add_edges_from(self.s[key].edges(data=True))
                cnt += 1
        
        # create clickable nodes
        node_points = _utils_GUI.interactive_node_points(graph, scale=0.7)
        selected_nodes = list()
        
        # Initialize label
        t0 = vv.Label(self.a1, '\b{Node nr|location}: ', fontSize=11, color='w')
        t0.position = 0.1, 25, 0.5, 20  # x (frac w), y, w (frac), h
        t0.bgcolor = None
        t0.visible = True
        
        # create dict for this scan for storing displacement vectors of selected nodes
        self.analysis[patient+ctcode] = {}
        
        def on_key(event, node_points):
            if event.key == vv.KEY_DOWN:
                # hide nodes
                for node_point in node_points:
                    node_point.visible = False
                    t0.visible = False
            if event.key == vv.KEY_UP:
                # show nodes
                for node_point in node_points:
                    node_point.visible = True
            if event.key == vv.KEY_ESCAPE:
                # get deforms of nodes selected and plot displacement
                for i, n in enumerate(selected_nodes):
                    # get node
                    selectn1 = n.node # position avgreg mid cycle
                    # get deforms of node
                    n1Deforms = graph.node[selectn1]['deforms']
                    pphases = n1Deforms + selectn1
                    # show these point positions (relative to avgreg)
                    points = plot_points(pphases, mc='y', ax=self.a1)
                    self.points_plotted.append(points)
                    #show the point at mid cycle = avgreg
                    plot_points(selectn1, mc='b', ax=self.a1)
                    self.analysis[patient+ctcode][analysis[i]] = [i, selectn1,n1Deforms]
                    # self.analysis[patient+ctcode+analysis[i]] = [selectn1,n1Deforms]
        
        # Bind event handlers
        self.f.eventKeyDown.Bind(lambda event: on_key(event, node_points) )
        # bind callback functions to node points
        _utils_GUI.node_points_callbacks(node_points, selected_nodes, t0=t0) 
        
        # Print user instructions
        print('')
        print('Esc = finish to plot displacement of selected nodes')
        print('x = axis invisible/visible')

    
    def loadssdf(self,patient='LSPEAS_001', ctcode='discharge',
                        basedir=None, cropname='ring'):
        """ load ssdf to store in self
        """
        if basedir is None:
            basedir = self.basedir
        s = loadmodel(basedir, patient, ctcode, cropname, modelname='modelavgreg')
        self.modelssdf.append(s)
    
    def amplitude_of_node_points_models(self, graphname='modelspine'):
        """ to calculate the displacement of the spine points (error) for the 
        models in the ssdfs loaded to self with loadssdf
        """ 
        from stentseg.motion.displacement import _calculateAmplitude
        from lspeas.utils import normality_shapiro
        from lspeas.utils.normality_shapiro import normality_check
        
        xamplitudes = []
        yamplitudes = []
        zamplitudes = []
        xyzamplitudes = []
        
        for s in self.modelssdf:
            for key in dir(s):
                if key.startswith(graphname):
                    graph = s[key]
                    for i, n in enumerate(sorted(graph.nodes())):
                        relpositions = graph.node[n]['deforms']
                        dmax, p1, p2 = _calculateAmplitude(relpositions, dim = 'x')
                        xamplitudes.append(dmax)
                        dmax, p1, p2 = _calculateAmplitude(relpositions, dim = 'y')
                        yamplitudes.append(dmax)
                        dmax, p1, p2 = _calculateAmplitude(relpositions, dim = 'z')
                        zamplitudes.append(dmax)
                        dmax, p1, p2 = _calculateAmplitude(relpositions, dim = 'xyz')
                        xyzamplitudes.append(dmax)
                    print(i)
        
        # check normality
        W, pValue, normality = normality_check(xamplitudes, alpha=0.05, showhist=True)
        print('Amplitude distribution in x of tested graphnodes gives a pValue of {}'.format(pValue))
        
        # get stats
        xamplitudemean = np.mean(xamplitudes)
        yamplitudemean = np.mean(yamplitudes)
        zamplitudemean = np.mean(zamplitudes)
        xyzamplitudemean = np.mean(xyzamplitudes)
        
        xamplitudestd = np.std(xamplitudes)
        yamplitudestd = np.std(yamplitudes)
        zamplitudestd = np.std(zamplitudes)
        xyzamplitudestd = np.std(xyzamplitudes)
        
        xamplitudemin, xamplitudemax = np.min(xamplitudes), np.max(xamplitudes)
        yamplitudemin, yamplitudemax = np.min(yamplitudes), np.max(yamplitudes)
        zamplitudemin, zamplitudemax = np.min(zamplitudes), np.max(zamplitudes)
        xyzamplitudemin, xyzamplitudemax = np.min(xyzamplitudes), np.max(xyzamplitudes)
        
        xstats = [xamplitudemean, xamplitudestd, xamplitudemin, xamplitudemax]
        ystats = [yamplitudemean, yamplitudestd, yamplitudemin, yamplitudemax]
        zstats = [zamplitudemean, zamplitudestd, zamplitudemin, zamplitudemax]
        xyzstats = [xyzamplitudemean, xyzamplitudestd, xyzamplitudemin, xyzamplitudemax]
        
        return xstats, ystats, zstats, xyzstats
        
        
#todo: wip
    def plot_pulsatility_line_per_patient(self, patients=None, ylim=[0, 2], ylim_perc=[0,5], saveFig=True):
        """ Plot pulsatility rings individul patients lines
        plot in absolute change in mm and as radial strain percentage
        
        """
        
        exceldir = self.exceldir
        workbook_stent = self.workbook_stent
        workbook_vars = self.workbook_variables
        
        wb = openpyxl.load_workbook(os.path.join(exceldir, workbook_stent), data_only=True)
        wbvars = openpyxl.load_workbook(os.path.join(exceldir, workbook_vars), data_only=True)
        
        # init figure
        f1 = plt.figure(num=3, figsize=(11.6, 9.2)) # 4.6
        xlabels = ['D', '1M', '6M', '12M', '24M']
        xrange = range(1,1+len(xlabels)) # not start from x=0 to play with margin xlim
        
        # init axis
        ax1 = f1.add_subplot(2,2,1)
        plt.xticks(xrange, xlabels, fontsize = 14)
        ax2 = f1.add_subplot(2,2,2)
        plt.xticks(xrange, xlabels, fontsize = 14)
        ax3 = f1.add_subplot(2,2,3)
        plt.xticks(xrange, xlabels, fontsize = 14)
        ax4 = f1.add_subplot(2,2,4)
        plt.xticks(xrange, xlabels, fontsize = 14)
        
        ax1.set_ylabel('Pulsatility R1 (mm)', fontsize=15) #Radial distension?
        ax2.set_ylabel('Pulsatility R2 (mm)', fontsize=15)
        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)
        ax1.set_xlim([0.8, len(xlabels)+0.2]) # xlim margins 0.2
        ax2.set_xlim([0.8, len(xlabels)+0.2])
        # plots in perc change cycle
        ax3.set_ylabel('Pulsatility R1 (%)', fontsize=15) # mean distance pp vv
        ax4.set_ylabel('Pulsatility R2 (%)', fontsize=15) # mean distance pp vv
        ax3.set_ylim(ylim_perc)
        ax4.set_ylim(ylim_perc)
        ax3.set_xlim([0.8, len(xlabels)+0.2]) # xlim margins 0.2
        ax4.set_xlim([0.8, len(xlabels)+0.2])
        
        # lines and colors; 12-class Paired
        colors = itertools.cycle(['#a6cee3','#1f78b4','#b2df8a','#33a02c',
        '#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928'])
        markers = ['D', 'o', '^', 's', '*']
        lw = 1
        
        # read data
        rowStart = 13 # 13 = row 14 excel
        colStart = ['B', 'G'] # B, G
        colStart = cols2num(colStart)
        colStartPerc = [col+1 for col in colStart]
        
        if patients is None:
            patients = self.patients
        # loop through patient sheets
        for i, patient in enumerate(patients):
            sheet = wb.get_sheet_by_name(patient)
            # read R1/R2
            if patient == ('LSPEAS_023' or 'LSPEAS_024'):
                continue
            else:    
                R1 = sheet.rows[rowStart][colStart[0]:colStart[0]+5] # +4 is read until 12M
                R2 = sheet.rows[rowStart][colStart[1]:colStart[1]+5] # +4 is read until 12M
                R1mm = sheet.rows[rowStartmm][colStart[0]:colStart[0]+5] # +4 is read until 12M
                R2mm = sheet.rows[rowStartmm][colStart[1]:colStart[1]+5] # +4 is read until 12M
            R1 = [obj.value for obj in R1]
            R2 = [obj.value for obj in R2]
            R1mm = [obj.value for obj in R1mm]
            R2mm = [obj.value for obj in R2mm]
            # read preop applied oversize
            sheet_preop = wbvars.get_sheet_by_name(self.workbook_variables_presheet)
            rowPre = 8 # row 9 in excel
            for j in range(18): # read sheet column with patients
                pt = sheet_preop.rows[rowPre+j][1]
                pt = pt.value
                if pt == patient[-3:]:
                    devicesize = sheet_preop.rows[rowPre+j][4].value # 4 = col E
                    break
            # plot
            ls = '-'
            color = next(colors)
            # if i > 11: # through 12 colors
            #     marker = next(mStyles)
            if devicesize == 25.5:
                marker = markers[0]
                olb = 'OLB25' 
            elif devicesize == 28:
                marker = markers[1]
                olb = 'OLB28' 
            elif devicesize == 30.5:
                marker = markers[2]
                olb = 'OLB30'
            elif devicesize == 32:
                marker = markers[3]
                olb = 'OLB32'
            else:
                marker = markers[4]
                olb = 'OLB34'
            if patient == 'LSPEAS_004': # FEVAR
                color = 'k'
                ls = ':'
                olb = 'OLB32'
            elif patient == 'LSPEAS_023': # endurant
                color = 'k'
                ls = '-.'
                olb = 'body28'
            
            # when scans are not scored in excel do not plot '#DIV/0!'
            R1 = [el if not isinstance(el, str) else None for el in R1]
            R2 = [el if not isinstance(el, str) else None for el in R2]
            R1mm = [el if not isinstance(el, str) else None for el in R1mm]
            R2mm = [el if not isinstance(el, str) else None for el in R2mm]
            # get deployment% (in excel percentage is used)
            R1 = [100*(1-el) if not el is None else None for el in R1]
            R2 = [100*(1-el) if not el is None else None for el in R2]
            
            alpha = 1
            if preop:
                xaxis = xrange[1:]
            else:
                xaxis = xrange
            # plot postop rdc
            ax1.plot(xaxis, R1, ls=ls, lw=lw, marker=marker, color=color, 
                    label='%s: %s' % (patient[-2:], olb), alpha=alpha)
            ax2.plot(xaxis, R2, ls=ls, lw=lw, marker=marker, color=color,
                    label='%s: %s' % (patient[-2:], olb), alpha=alpha)
            # label='%i: ID %s' % (i+1, patient[-3:]), alpha=alpha)
            if preop:
                # plot preop rdc
                if not isinstance(preR1, str): # not yet scored in excel so '#DIV/0!'
                    ax1.plot(xrange[:2], [preR1,R1[0]], ls=ls, lw=lw, 
                        marker=marker, color=color, alpha=alpha)
                if not isinstance(preR2, str):
                    ax2.plot(xrange[:2], [preR2,R2[0]], ls=ls, lw=lw, 
                        marker=marker, color=color, alpha=alpha)
            
            # plot in mm postop
            ax3.plot(xaxis, R1mm, ls=ls, lw=lw, marker=marker, color=color, 
                    label='%s: %s' % (patient[-2:], olb), alpha=alpha)
            ax4.plot(xaxis, R2mm, ls=ls, lw=lw, marker=marker, color=color, 
                    label='%s: %s' % (patient[-2:], olb), alpha=alpha)
            if preop:
                # plot in mm preop
                if not isinstance(preR1mm, str): # not yet scored in excel so '#DIV/0!'
                    ax3.plot(xrange[:2], [preR1mm,R1mm[0]], ls=ls, lw=lw, 
                        marker=marker, color=color, alpha=alpha)
                if not isinstance(preR2mm, str):
                    ax4.plot(xrange[:2], [preR2mm,R2mm[0]], ls=ls, lw=lw, 
                        marker=marker, color=color, alpha=alpha)
            
        ax1.legend(loc='lower right', fontsize=9, numpoints=1, title='Patients')
        _initaxis([ax1, ax2, ax3, ax4])
        
        if saveFig:
            plt.savefig(os.path.join(self.dirsaveIm, 
            'plot_ring_deployment.png'), papertype='a0', dpi=600)






def create_iter_colors(type=1):
    if type == 1:
        colors = itertools.cycle([  
                                '#a6cee3', # 1
                                '#1f78b4', # 2 
                                '#b2df8a', # 3
                                '#33a02c', # 4 
                                '#fb9a99', # 5
                                '#e31a1c', # 6
                                '#fdbf6f', # 7 
                                '#ff7f00', # 8 
                                '#cab2d6', # 9
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
    elif type == 4:
        colors =  itertools.cycle([
                                    '#d73027', # 1 red
                                    '#91bfdb', # blue
                                    'k', # black
                                    '#fc8d59', # orange
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
        markers = itertools.cycle(['o', '^'])
    
    if type == 4:
        markers = itertools.cycle(['o', '^', 's','d'])
    
    if type == 5:
        markers = itertools.cycle(['o', 'o', '.', 's'])
    
    return markers


def create_iter_ls(type=1):
    if type == 1:
        lstyles = itertools.cycle(['-'])
    
    elif type == 2:
        lstyles = itertools.cycle(['-', '-', '-', '--','--','--'])
    
    elif type == 3:
        lstyles = itertools.cycle(['-', '--']) #-.
    
    elif type == 4:
        lstyles = itertools.cycle(['-', '--', '-.'])

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
    
def get_devicesize(ptcode):
    """
    """


if __name__ == '__main__':
    
    foo = MotionAnalysis()
    
    # foo.plot_displacement(patient='LSPEAS_020', ctcodes=['discharge','24months'], 
    #     analysis=['post', 'right'], ring='R1', ylim=[-0.65, 0.75], ylimRel=[0,1], saveFig=False)
    
    # foo.show_displacement_3d('LSPEAS_020', '24months',
    #                         analysis=['R1post', 'R1right'], 
    #                         clim=(0,2600),showVol='MIP', showModelavgreg=True)
    # 
    # foo.show_displacement_3d('LSPEAS_023', 'discharge',
    #                         analysis=['R1ant', 'R2ant'], 
    #                         clim=(0,2600),showVol='MIP', showModelavgreg=True, locationcheck=False)
    
    if False:
        # Select points from 3d image
        # step 1 select and visualize 3d
        foo.show_displacement_3d_select_points(patient='lspeasf_C_01', ctcode='d', cropname='ring',
                analysis=['Ring right', 'Spine'], #labels for plot legend 'Ring posterior', 'Spine',
                showVol='MIP', basedir=None)
        # step 2 plot 2d
        foo.plot_displacement_selected_points(colortype=4, colortype3d=3,
                markertype=5, linetype=4,
                ylim=[-0.7, 0.85], ylimRel=[0,0.8], saveFig=False)
    
    if False:
        # Get displacement amplitude of spinepoints
        # load models to analyze
        foo.loadssdf(patient='lspeasf_c_01', ctcode='d', basedir=None, cropname='ring')
        foo.loadssdf(patient='lspeas_020', ctcode='discharge', basedir=None, cropname='ring')
        foo.loadssdf(patient='lspeas_023', ctcode='discharge', basedir=None, cropname='ring')
        foo.loadssdf(patient='chevas_09', ctcode='12months', basedir=None, cropname='prox')
        # analyze nodepoint amplitudes of loaded models
        xstats, ystats, zstats, xyzstats = foo.amplitude_of_node_points_models(graphname='modelspine')
        print(xstats)
        print(ystats)
        print(zstats)
        #print(xyzstats)

