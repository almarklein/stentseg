""" LSPEAS RING DYNAMICS: Script to collect ring dynamics data from
automated analysis in ring_dynamics.py (excel sheet per patient)
Copyright 2019, Maaike A. Koenrades
"""

from lspeas.analysis.utils_analysis import readRingExcel, _initaxis, cols2num, read_deforms
import openpyxl
from openpyxl.utils import column_index_from_string
from stentseg.utils.datahandling import select_dir
import sys, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np 
import itertools
from matplotlib import gridspec
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
import scipy
from scipy import io

class ExcelAnalysisRingDynamics():
    """ Create graphs from excel file per patient created by ring_dynamics.py
    """
    
    exceldir = select_dir(r'C:\Users\Maaike\SURFdrive\UTdrive\LSPEAS\Analysis\Ring motion\ringdynamics', 
                    r'D:\Profiles\koenradesma\SURFdrive\UTdrive\LSPEAS\Analysis\Ring motion\ringdynamics')
    dirsaveIm =  select_dir(r'C:\Users\Maaike\Desktop','D:\Profiles\koenradesma\Desktop')
    
    
    def __init__(self):
        self.exceldir =  ExcelAnalysisRingDynamics.exceldir
        self.dirsaveIm = ExcelAnalysisRingDynamics.dirsaveIm
        self.patients =['LSPEAS_001', 
                        'LSPEAS_002',	
                        'LSPEAS_003', 
                        'LSPEAS_005',	
                        # 'LSPEAS_008', 
                        # 'LSPEAS_009',	
                        # 'LSPEAS_011', 
                        # 'LSPEAS_015',	
                        # 'LSPEAS_017',	
                        # 'LSPEAS_018',
                        # 'LSPEAS_019', 
                        # 'LSPEAS_020', 
                        # 'LSPEAS_021', 
                        # 'LSPEAS_022', 
                        # 'LSPEAS_025', 
                        ]
        
        
    def get_curvature_change(self, patients=None, analysis=['R1','Q1R1','Q2R1','Q3R1','Q4R1'], 
                curvetype='pointchange', storemat=False):
        """ Read curvature change for the ring or ring quadrant
        analysis: R1 and/or Q1R1 etc for ring or part of ring
        curvetype: 'pointchange'        --> curvature change per point
                   'peakcurvature'      --> max diff between peak angle over phases
                   'meancurvature'      --> max diff between mean curvature over phases
        """
        
        if patients == None:
            patients = self.patients
        
        self.curveChange = []
        self.angleMin = []
        self.angleMax = []
        self.locationOnRing = [] # location on ring as percentage distance from prox / length chimney
        self.locationChange = [] # location change of peak angle 
        
        ctcodes = ['discharge', '1month', '6months', '12months', '24months']
        
        # read workbooks per patient (this might take a while)
        for a in analysis:
            # init to collect over time all pts
            curvechangeOverTime_pts = [] # cm-1; 5 values for each patient
            curvechangePOverTime_pts = [] # %
            curvechangeRelLocOverTime_pts = [] # %
            avgcurvechangeOverTime_pts = [] # cm-1
            avgcurvechangePOverTime_pts = [] # %
            for patient in patients:
                # init to collect parameter for all timepoints for this patient
                curvechangeOverTime = [] # cm-1
                curvechangePOverTime = [] # %
                curvechangeRelLocOverTime = [] # %
                avgcurvechangeOverTime = []
                avgcurvechangePOverTime = [] 
                for i in range(len(ctcodes)):
                    filename = '{}_ringdynamics_{}.xlsx'.format(patient, ctcodes[i])
                    workbook_stent = os.path.join(self.exceldir, filename)
                    # read workbook
                    try:
                        wb = openpyxl.load_workbook(workbook_stent, data_only=True)
                    except FileNotFoundError: # handle missing scans
                        if curvetype == 'pointchange':
                            # collect
                            curvechangeOverTime.append(np.nan)
                            curvechangePOverTime.append(np.nan)
                            curvechangeRelLocOverTime.append(np.nan)
                            avgcurvechangeOverTime.append(np.nan)
                            avgcurvechangePOverTime.append(np.nan)
                        continue # no scan, next
                    
                    # get sheet
                    sheetname = 'Curvature{}'.format(a)
                    sheet = wb.get_sheet_by_name(sheetname)
                    # set row for type of curvature analysis
                    if curvetype == 'pointchange':
                        rowstart = 5 # as in excel; max change of a point
                        rowstart2 = 9 # as in excel; rel location
                        rowstart3 = 13 # as in excel; mean curvature change all points
                        colstart = 1 # 1 = B
                        # read change
                        curvechange = sheet.rows[rowstart-1][colstart].value
                        curvechangeP = sheet.rows[rowstart-1][colstart+1].value
                        curvechangeRelLoc = sheet.rows[rowstart2-1][colstart].value 
                        avgcurvechange = sheet.rows[rowstart3-1][colstart].value
                        avgcurvechangeP = sheet.rows[rowstart3-1][colstart+1].value
                        # collect
                        curvechangeOverTime.append(curvechange)
                        curvechangePOverTime.append(curvechangeP)
                        curvechangeRelLocOverTime.append(curvechangeRelLoc)
                        avgcurvechangeOverTime.append(avgcurvechange)
                        avgcurvechangePOverTime.append(avgcurvechangeP)
                        
                # collect for pts
                curvechangeOverTime_pts.append(np.asarray(curvechangeOverTime))
                curvechangePOverTime_pts.append(np.asarray(curvechangePOverTime))
                curvechangeRelLocOverTime_pts.append(np.asarray(curvechangeRelLocOverTime))        
                avgcurvechangeOverTime_pts.append(np.asarray(avgcurvechangeOverTime))
                avgcurvechangePOverTime_pts.append(np.asarray(avgcurvechangePOverTime))
            
            # Store to .mat per analysis
            if storemat:
                if curvetype == 'pointchange':
                    # cm-1
                    self.store_var_to_mat(np.asarray(curvechangeOverTime_pts), varname='Curvature{}_maxchange_pts'.format(a) )
                    self.store_var_to_mat(np.asarray(curvechangePOverTime_pts), varname='Curvature{}_maxchangeP_pts'.format(a) )
                    self.store_var_to_mat(np.asarray(curvechangeRelLocOverTime_pts), varname='Curvature{}_maxchangeRelLoc_pts'.format(a) )
                    self.store_var_to_mat(np.asarray(avgcurvechangeOverTime_pts), varname='Curvature{}_meanchange_pts'.format(a) )
                    self.store_var_to_mat(np.asarray(avgcurvechangePOverTime_pts), varname='Curvature{}_meanchangeP_pts'.format(a) )
                    
    
    def store_var_to_mat(self, variable, varname=None, storematdir=None):
        """ Save as .mat to easy copy to spss
        """
        if storematdir is None:
            storematdir = os.path.join(self.dirsaveIm, 'python_to_mat_output_ringdynamics')
        if varname is None:
            varname = 'variable_from_python'
        storemat = os.path.join(storematdir, varname+'.mat')
        
        storevar = dict()
        # check if variable has multiple vars in list
        if isinstance(variable, list):
            for i, var in enumerate(variable):
                name = varname+'{}'.format(i)
                storevar[name] = var
        else:
            storevar[varname] = variable
        
        storevar['workbooks_ringdynamics'] = '_ringdynamics_'
        storevar['patients'] = self.patients
        
        io.savemat(storemat,storevar)
        print('')
        print('variable {} was stored as.mat to {}'.format(varname, storemat))
                    


if __name__ == '__main__':
    
    foo = ExcelAnalysisRingDynamics()
    
    ## Ring motion manuscript 
    
    # get and store curvature change rings and quadrants post left ant right
    foo.get_curvature_change(patients=None, analysis=['R1','Q1R1','Q2R1','Q3R1','Q4R1'], 
            curvetype='pointchange', storemat=True)