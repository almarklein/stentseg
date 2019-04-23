""" Module to analyze ring-stent dynamics during the heart cycle from (dynamic) models
* ring curvature
* distance between ring1 and ring2
* displacement ring parts/quartiles

Uses excel pulsatility_and_expansion to get location of peaks and valleys
M.A. Koenrades. Created April 2019
"""

import sys, os
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from stentseg.utils.picker import pick3d
from stentseg.utils.visualization import DrawModelAxes, show_ctvolume, plot_points
from stentseg.utils import PointSet, _utils_GUI, visualization
from stentseg.motion.vis import get_graph_in_phase
import visvis as vv
import numpy as np
import copy
import math
from stentseg.motion.displacement import calculateMeanAmplitude
import xlsxwriter
import openpyxl
from lspeas.utils.curvature import measure_curvature
from lspeas.utils.get_anaconda_ringparts import get_model_struts,get_model_rings, add_nodes_edge_to_newmodel 
import pirt

class _Do_Analysis_Rings:
    """ Analyze motion and curvature of dynamic ring models
    Functions for chimneys/branches and main stents
    """
    exceldir = select_dir(r'C:\Users\Maaike\SURFdrive\UTdrive\LSPEAS\Analysis', 
                    r'D:\Profiles\koenradesma\SURFdrive\UTdrive\LSPEAS\Analysis')
    
    # select the ssdf basedir
    basedir = select_dir(
        r'D:\LSPEAS\LSPEAS_ssdf',
        r'F:\LSPEAS_ssdf_backup',
        r'F:\LSPEAS_ssdf_BACKUP')
    
    def __init__(self,ptcode,ctcode,cropname,cropvol='stent', nstruts=8,
            showVol='MIP',clim=(0,2500), color='b', mw=5, **kwargs):
        """
        Init motion analysis on ring models 
        """
        self.exceldir =  _Do_Analysis_Rings.exceldir
        self.basedir = _Do_Analysis_Rings.basedir
        self.workbook_stent = 'LSPEAS_pulsatility_expansion_avgreg_subp_v2.1.xlsx'
        
        self.ptcode = ptcode
        self.ctcode = ctcode
        self.cropname = cropname
        self.cropvol = cropvol
        
        # Load CT image data for reference, and deform data to measure motion
        try:
            # If we run this script without restart, we can re-use volume and deforms
            self.vol
            self.deforms
        except AttributeError:
            self.vol = loadvol(self.basedir, self.ptcode, self.ctcode, self.cropvol, 'avgreg').vol
            s_deforms = loadvol(self.basedir, self.ptcode, self.ctcode, self.cropname, 'deforms')
            deforms = [s_deforms[key] for key in dir(s_deforms) if key.startswith('deform')]
            deforms = [pirt.DeformationFieldBackward(*fields) for fields in deforms]
            self.deforms = deforms
            
        # Load ring model
        try:
            self.model
        except AttributeError:
            self.s_model = loadmodel(self.basedir, self.ptcode, self.ctcode, self.cropname, 'modelavgreg')
            self.model = self.s_model.model 
        
        # figure
        f = vv.figure(); vv.clf()
        f.position = 0.00, 22.00,  944.00, 1018.00
        self.a = vv.gca()
        self.a.axis.axisColor = 0,0,0#1,1,1
        self.a.axis.visible = False
        self.a.bgcolor = 1,1,1#0,0,0
        self.a.daspect = 1, 1, -1
        t = show_ctvolume(self.vol, showVol=showVol, clim=clim, removeStent=False, 
                        climEditor=True, isoTh=225, **kwargs)
        self.label = pick3d(self.a, self.vol)
        vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
        self.model.Draw(mc=color, mw = mw, lc=color, alpha = 0.5)
        
        vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
        
        f.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event, [self.a], axishandling=False) )
        f.eventKeyDown.Bind(lambda event: _utils_GUI.ViewPresets(event, [self.a]) )
        self.fig = f
        
        # get 2 seperate rings
        modelsout = get_model_struts(self.model, nstruts=nstruts)
        model_R1R2 = modelsout[2]
        modelR1, modelR2  = get_model_rings(model_R1R2)
        
        # vis rings
        fig = vv.figure();
        fig.position = 8.00, 30.00,  944.00, 1002.00
        vv.clf()
        a0 = vv.subplot(211)
        t2 = show_ctvolume(self.vol, showVol=showVol, clim=clim, removeStent=False, 
                        climEditor=True, isoTh=225, **kwargs)
        modelR1.Draw(mc='g', mw = 10, lc='g') # R1 = green
        vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
        vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
        a0.axis.axisColor= 0,0,0
        a0.bgcolor= 1,1,1
        a0.daspect= 1, 1, -1  # z-axis flipped
        a0.axis.visible = False
        
        a1 = vv.subplot(212)
        t3 = show_ctvolume(self.vol, showVol=showVol, clim=clim, removeStent=False, 
                        climEditor=True, isoTh=225, **kwargs)
        modelR2.Draw(mc='c', mw = 10, lc='c') # R2 = cyan
        vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
        vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
        a1.axis.axisColor= 0,0,0
        a1.bgcolor= 1,1,1
        a1.daspect= 1, 1, -1  # z-axis flipped
        a1.axis.visible = False
        
        a0.camera = a1.camera
        
        fig.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event, [a0, a1], axishandling=False) )
        fig.eventKeyDown.Bind(lambda event: _utils_GUI.ViewPresets(event, [a0, a1]) )
        self.fig2 = fig
        
        
        
        
        
        





if __name__ == '__main__':
    
    # Select dataset
    ptcode = 'LSPEAS_001'
    ctcode = 'discharge'
    cropname = 'ring'
    
    showVol  = 'ISO'  # MIP or ISO or 2D or None
    nstruts = 8 # needed to get seperate ring models for R1 R2
    
    foo = _Do_Analysis_Rings(ptcode,ctcode,cropname,nstruts=nstruts,showVol=showVol)
    
    
    