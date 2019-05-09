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
        """ Read curvature change for the ring and/or ring quadrant
        analysis: R1 and/or Q1R1 etc for ring or part of ring
        curvetype: 'pointchange'        --> curvature change per point
                   'peakcurvature'      --> max diff between peak angle over phases
                   'meancurvature'      --> max diff between mean curvature over phases
        """
        
        if patients == None:
            patients = self.patients
        
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
                    self.store_var_to_mat(np.asarray(curvechangeOverTime_pts), varname='Curvature{}_maxchange_pts'.format(a) )
                    self.store_var_to_mat(np.asarray(curvechangePOverTime_pts), varname='Curvature{}_maxchangeP_pts'.format(a) )
                    self.store_var_to_mat(np.asarray(curvechangeRelLocOverTime_pts), varname='Curvature{}_maxchangeRelLoc_pts'.format(a) )
                    self.store_var_to_mat(np.asarray(avgcurvechangeOverTime_pts), varname='Curvature{}_meanchange_pts'.format(a) )
                    self.store_var_to_mat(np.asarray(avgcurvechangePOverTime_pts), varname='Curvature{}_meanchangeP_pts'.format(a) )
                    
    
    def get_displacement_cycle(self, patients=None, analysis=['R1','Q1R1','Q2R1','Q3R1','Q4R1'], 
                storemat=False):
        """ Read displacement during the cycle for the ring and/or ring quadrant
        analysis: R1 and/or Q1R1 etc for ring or part of ring
        Obtain mean displacement in 3d and x,y,z directions
        """
        
        if patients == None:
            patients = self.patients
        
        ctcodes = ['discharge', '1month', '6months', '12months', '24months']
        
        # read workbooks per patient (this might take a while)
        for a in analysis:
            # init to collect over time all pts
            displxyzOverTime_pts = [] #  xyz mm; 5 values for each patient
            curvechangePOverTime_pts = [] # x mm
            curvechangeRelLocOverTime_pts = [] # y mm
            avgcurvechangeOverTime_pts = [] # z mm
            for patient in patients:
                # init to collect parameter for all timepoints for this patient
                displxyzOverTime = [] 
                displxOverTime = [] 
                displyOverTime = [] 
                displzOverTime = []
                for i in range(len(ctcodes)):
                    filename = '{}_ringdynamics_{}.xlsx'.format(patient, ctcodes[i])
                    workbook_stent = os.path.join(self.exceldir, filename)
                    # read workbook
                    try:
                        wb = openpyxl.load_workbook(workbook_stent, data_only=True)
                    except FileNotFoundError: # handle missing scans
                        # collect
                        displxyzOverTime.append(np.nan)
                        displxOverTime.append(np.nan)
                        displyOverTime.append(np.nan)
                        displzOverTime.append(np.nan)
                        continue # no scan, next
                    
                    # get sheet
                    sheetname = 'Motion{}'.format(a)
                    sheet = wb.get_sheet_by_name(sheetname)
                    # set row for 3d,x,y,z
                    rowstart = 6 # as in excel; xyz/3d
                    rowstart2 = 7 # as in excel; x
                    rowstart3 = 8 # as in excel; y
                    rowstart4 = 9 # as in excel; z
                    
                    colstart = 1 # 1 = B
                    # read change
                    displxyz = sheet.rows[rowstart-1][colstart].value
                    displx = sheet.rows[rowstart2-1][colstart].value 
                    disply = sheet.rows[rowstart3-1][colstart].value
                    displz = sheet.rows[rowstart4-1][colstart].value 
                    # collect
                    displxyzOverTime.append(displxyz)
                    displxOverTime.append(displx)
                    displyOverTime.append(disply)
                    displzOverTime.append(displz)
                    
                # collect for pts
                displxyzOverTime_pts.append(np.asarray(displxyzOverTime))
                curvechangePOverTime_pts.append(np.asarray(displxOverTime))
                curvechangeRelLocOverTime_pts.append(np.asarray(displyOverTime))        
                avgcurvechangeOverTime_pts.append(np.asarray(displzOverTime))
                
            # Store to .mat per analysis
            if storemat:
                self.store_var_to_mat(np.asarray(displxyzOverTime_pts), varname='Displacement{}_3d_pts'.format(a) )
                self.store_var_to_mat(np.asarray(curvechangePOverTime_pts), varname='Displacement{}_x_pts'.format(a) )
                self.store_var_to_mat(np.asarray(curvechangeRelLocOverTime_pts), varname='Displacement{}_y_pts'.format(a) )
                self.store_var_to_mat(np.asarray(avgcurvechangeOverTime_pts), varname='Displacement{}_z_pts'.format(a) )
    
    
    def get_distance_change_between_rings(self, patients=None, analysis=['R1R2','Q1R1R2','Q2R1R2','Q3R1R2','Q4R1R2'], 
                storemat=False):
        """ Read distance change during the cycle for the ring and/or ring quadrant
        analysis: R1 and/or Q1R1 etc for ring or part of ring
        Obtain mean and max distance change
        """
        
        if patients == None:
            patients = self.patients
        
        ctcodes = ['discharge', '1month', '6months', '12months', '24months']
        
        # read workbooks per patient (this might take a while)
        for a in analysis:
            # init to collect over time all pts
            meanOverTime_pts = [] # mm; 5 values for each patient
            maxOverTime_pts = [] # mm
            for patient in patients:
                # init to collect parameter for all timepoints for this patient
                meanOverTime = [] 
                maxOverTime = [] 
                for i in range(len(ctcodes)):
                    filename = '{}_ringdynamics_{}.xlsx'.format(patient, ctcodes[i])
                    workbook_stent = os.path.join(self.exceldir, filename)
                    # read workbook
                    try:
                        wb = openpyxl.load_workbook(workbook_stent, data_only=True)
                    except FileNotFoundError: # handle missing scans
                        # collect
                        meanOverTime.append(np.nan)
                        maxOverTime.append(np.nan)
                        continue # no scan, next
                    
                    # get sheet
                    sheetname = 'Distances{}'.format(a)
                    sheet = wb.get_sheet_by_name(sheetname)
                    # set row for 3d,x,y,z
                    rowstart = 16 # as in excel; mean
                    rowstart2 = 6 # as in excel; max
                    colstart = 1 # 1 = B
                    # read change
                    distmean = sheet.rows[rowstart-1][colstart].value
                    distmax = sheet.rows[rowstart2-1][colstart].value 
                    # collect
                    meanOverTime.append(distmean)
                    maxOverTime.append(distmax)
                    
                # collect for pts
                meanOverTime_pts.append(np.asarray(meanOverTime))
                maxOverTime_pts.append(np.asarray(maxOverTime))
                
            # Store to .mat per analysis
            if storemat:
                self.store_var_to_mat(np.asarray(meanOverTime_pts), varname='Distances{}_mean_pts'.format(a) )
                self.store_var_to_mat(np.asarray(maxOverTime_pts), varname='Distances{}_max_pts'.format(a) )
                
    
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
    
    # Get and store curvature change rings and quadrants post left ant right
    # R1
    # foo.get_curvature_change(patients=None, analysis=['R1','Q1R1','Q2R1','Q3R1','Q4R1'], 
    #         curvetype='pointchange', storemat=True)
    # R2
    # foo.get_curvature_change(patients=None, analysis=['R2','Q1R2','Q2R2','Q3R2','Q4R2'], 
    #         curvetype='pointchange', storemat=True)
    
    # Get and store displacement rings and quadrants
    # R1
    # foo.get_displacement_cycle(patients=None, analysis=['R1','Q1R1','Q2R1','Q3R1','Q4R1'], 
    #         storemat=True)
    # R2
    # foo.get_displacement_cycle(patients=None, analysis=['R2','Q1R2','Q2R2','Q3R2','Q4R2'], 
    #         storemat=True)
    
    # Get and store displacement between rings
    foo.get_distance_change_between_rings(patients=None, analysis=['R1R2','Q1R1R2','Q2R1R2','Q3R1R2','Q4R1R2'], 
                storemat=True)
    
    
    
    