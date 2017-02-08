""" Functionality for stent analysis

"""

from stentseg.utils import PointSet
import openpyxl
from stentseg.utils.datahandling import select_dir
import sys, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np 
import itertools


def point_in_pointcloud_closest_to_p(pp, point):
    """ Find point in PointSet which is closest to a point
    Returns a PointSet with point found in PointSet and point 
    """
    vecs = pp-point
    dists_to_pp = ( (vecs[:,0]**2 + vecs[:,1]**2 + vecs[:,2]**2)**0.5 ).reshape(-1,1)
    pp_index =  list(dists_to_pp).index(dists_to_pp.min() ) # index on path
    pp_point = pp[pp_index]
    p_in_pp_and_point = PointSet(3, dtype='float32')
    [p_in_pp_and_point.append(*p) for p in (pp_point, point)]
    
    return p_in_pp_and_point


class ExcelAnalysis():
    
    exceldir = select_dir(r'C:\Users\Maaike\Dropbox\UTdrive\LSPEAS\Analysis', 
                    r'D:\Profiles\koenradesma\Dropbox\UTdrive\LSPEAS\Analysis')
    dirsaveIm =  select_dir(r'C:\Users\Maaike\Desktop','D:\Profiles\koenradesma\Desktop')
    
    def __init__(self):
        self.exceldir =  ExcelAnalysis.exceldir
        self.dirsaveIm = ExcelAnalysis.dirsaveIm
        self.workbook_stent = 'LSPEAS_pulsatility_expansion_avgreg_subp_v15.1.xlsx'
        self.workbook_renal = 'postop_measures_renal_aortic.xlsx'
        self.workbook_variables = 'LSPEAS_Variables.xlsx'
        self.sheet_peak_valley = 'peak valley locations'
        self.patients =['LSPEAS_001', 'LSPEAS_002',	'LSPEAS_003', 'LSPEAS_005',	
                        'LSPEAS_008', 'LSPEAS_009',	'LSPEAS_011', 'LSPEAS_015',	'LSPEAS_017',	
                        'LSPEAS_018', 'LSPEAS_019',	'LSPEAS_020', 'LSPEAS_021',	'LSPEAS_022',
                        'LSPEAS_025', 'LSPEAS_023', 'LSPEAS_024', 'LSPEAS_004']   


    def readRingExcel(self, ptcode, ctcode, ring='R1'):
        """ To read peak and valley locations, R1 or R2
        """
        
        exceldir = self.exceldir
        workbook = self.workbook_stent
        sheet = 'Output_'+ ptcode[-3:]
        
        wb = openpyxl.load_workbook(os.path.join(exceldir, workbook), data_only=True)
        sheet = wb.get_sheet_by_name(sheet)
        
        if ring == 'R1':
            colsStart = [1,5,9,13]
        elif ring == 'R2':
            colsStart = [17,21,25,29]
        if ctcode == 'discharge':
            R1_ant = sheet.rows[16][colsStart[0]:colsStart[0]+3]
            R1_ant = [obj.value for obj in R1_ant]
            R1_post = sheet.rows[29][colsStart[0]:colsStart[0]+3]
            R1_post = [obj.value for obj in R1_post] 
            R1_left = sheet.rows[53][colsStart[0]:colsStart[0]+3]
            R1_left = [obj.value for obj in R1_left]
            R1_right = sheet.rows[66][colsStart[0]:colsStart[0]+3]
            R1_right = [obj.value for obj in R1_right]
        elif ctcode == '1month':
            R1_ant = sheet.rows[16][colsStart[1]:colsStart[1]+3]
            R1_ant = [obj.value for obj in R1_ant]
            R1_post = sheet.rows[29][colsStart[1]:colsStart[1]+3]
            R1_post = [obj.value for obj in R1_post] 
            R1_left = sheet.rows[53][colsStart[1]:colsStart[1]+3]
            R1_left = [obj.value for obj in R1_left]
            R1_right = sheet.rows[66][colsStart[1]:colsStart[1]+3]
            R1_right = [obj.value for obj in R1_right]
        elif ctcode == '6months':
            R1_ant = sheet.rows[16][colsStart[2]:colsStart[2]+3]
            R1_ant = [obj.value for obj in R1_ant]
            R1_post = sheet.rows[29][colsStart[2]:colsStart[2]+3]
            R1_post = [obj.value for obj in R1_post] 
            R1_left = sheet.rows[53][colsStart[2]:colsStart[2]+3]
            R1_left = [obj.value for obj in R1_left]
            R1_right = sheet.rows[66][colsStart[2]:colsStart[2]+3]
            R1_right = [obj.value for obj in R1_right]
        elif ctcode == '12months':
            R1_ant = sheet.rows[16][colsStart[3]:colsStart[3]+3]
            R1_ant = [obj.value for obj in R1_ant]
            R1_post = sheet.rows[29][colsStart[3]:colsStart[3]+3]
            R1_post = [obj.value for obj in R1_post] 
            R1_left = sheet.rows[53][colsStart[3]:colsStart[3]+3]
            R1_left = [obj.value for obj in R1_left]
            R1_right = sheet.rows[66][colsStart[3]:colsStart[3]+3]
            R1_right = [obj.value for obj in R1_right]
        else:
            print('ctcode not known')
            ValueError
            
        return R1_ant, R1_post, R1_left, R1_right


    def writeRingPatientsExcel(self, ptcodes='all', ctcodes='all'):
        """ Write peak valley locations from multiple patients
        ptcodes/ctcodes can be 'all' or a list with ptcode/ctcode strings
        overwrites 'peak valleys locations
        """
        
        sheetwrite = 'peak valley locations'
        exceldir = self.exceldir
        workbook_stent = self.workbook_stent
        workbook_renal = self.workbook_renal
        sheetwrite = self.sheet_peak_valley
        
        wb = openpyxl.load_workbook(os.path.join(exceldir, workbook_renal), data_only=True)
        sheet_to_write = wb.get_sheet_by_name(sheetwrite)
        
        patients = ['LSPEAS_001', 'LSPEAS_002',	'LSPEAS_003', 'LSPEAS_005',	
                    'LSPEAS_008', 'LSPEAS_009',	'LSPEAS_011', 'LSPEAS_015',	'LSPEAS_017',	
                    'LSPEAS_018', 'LSPEAS_019',	'LSPEAS_020', 'LSPEAS_021',	'LSPEAS_022',
                    'LSPEAS_025']#, 'LSPEAS_023', 'LSPEAS_024', 'LSPEAS_004']
        codes = ['discharge', '1month', '6months', '12months']#, '24months'] first adjust excel
        if ptcodes == 'all':
            ptcodes = patients
        if ctcodes == 'all':
            ctcodes = codes
        rowStart = [8,31,54,77] # ant, post, left, right; for writing
        colStart = [4,8,12,16]#,20] ; for writing ctcodes
        for ctcode in ctcodes:
            col = colStart[ctcodes.index(ctcode)]
            for ptcode in ptcodes:
                R1_ant, R1_post, R1_left, R1_right = readRingExcel(workbook_stent, ptcode, ctcode, ring='R1')
                # write positions
                for i, position in enumerate([R1_ant, R1_post, R1_left, R1_right]):
                    row = rowStart[i] + patients.index(ptcode)
                    for j, coordinate in enumerate(position):
                        sheet_to_write.cell(row=row, column=col+j).value = coordinate
        wb.save(os.path.join(exceldir, workbook_renal))
        print('worksheet "%s" overwritten in workbook "%s"' % (sheetwrite,workbook_renal) )
    

    def readRenalsExcel(self, sheet, ptcode, ctcode, zPositive=True):
        """ To read renal locations. zPositive True will return z as non-negative
        use sheet to specify sheet in excel for observer
        """
        
        exceldir = self.exceldir
        workbook_renal = self.workbook_renal
        
        wb = openpyxl.load_workbook(os.path.join(exceldir, workbook_renal), data_only=True)
        sheet = wb.get_sheet_by_name(sheet)
        
        colsStart = [5,9,13,17,21]
        patients = self.patients
        
        if ctcode == 'discharge':
            col = colsStart[0]
        elif ctcode == '1month':
            col = colsStart[1]
        elif ctcode == '6months':
            col = colsStart[2]
        elif ctcode == '12months':
            col = colsStart[3]
        elif ctcode == '24months':
            col = colsStart[4]
        else:
            print('ctcode not known')
            ValueError
        
        rowstartleft = 7 # left renal
        rowstartright = 30 # right renal
        
        rowleft = rowstartleft + patients.index(ptcode)
        rowright = rowstartright + patients.index(ptcode)
        
        renal_left = sheet.rows[rowleft][col:col+3]
        renal_left = [obj.value for obj in renal_left]
        renal_right = sheet.rows[rowright][col:col+3]
        renal_right = [obj.value for obj in renal_right] 
        
        if zPositive == True:
            for renal in [renal_left, renal_right]:
                if renal[-1] < 0:
                    renal[-1] *= -1
        
        renal_left = PointSet(renal_left)
        renal_right = PointSet(renal_right)
        
        return renal_left, renal_right


    def plot_distance_to_renal(self, sheet, ptcode):
        """ Plot distance renal from peak valleys for 1 patient
        e.g. sheet = 'distances to renal obs1'
        """
        import prettyplotlib as ppl
        from prettyplotlib import brewer2mpl # colormaps, http://colorbrewer2.org/#type=sequential&scheme=YlGnBu&n=5
        from scipy.interpolate import interp1d #, spline, splrep, splev, UnivariateSpline
        
        colormap = brewer2mpl.get_map('YlGnBu', 'sequential', 5).mpl_colormap # ppl...(fig,ax,...,cmap = )
        exceldir = self.exceldir
        workbook_renal = self.workbook_renal
        
        wb = openpyxl.load_workbook(os.path.join(exceldir, workbook_renal), data_only=True)
        sheet = wb.get_sheet_by_name(sheet)
        
        patients = self.patients
        
        rowstartleft = 7 # left renal
        rowstartright = 30 # right renal
        
        colsStart = ['D', 'I', 'N', 'S']#, 'X'] # ctcodes
        colsStart = [(openpyxl.cell.column_index_from_string(char)-1) for char in colsStart]
        
        rowleft = rowstartleft + patients.index(ptcode)
        rowright = rowstartright + patients.index(ptcode)
        
        # read distances to renal for every ctcode and 4 peak valleys positions
        zDistances_ctcodes = [] # VR PA VL PP
        for col in colsStart:
            zDistance = sheet.rows[rowleft][col:col+4]
            zDistance = [obj.value for obj in zDistance]
            zDistances_ctcodes.append(zDistance) 
        
        # plot
        f1 = plt.figure(num=1, figsize=(7.6, 5))
        ax1 = f1.add_subplot(111)
        _initaxis([ax1])
        ax1.set_xlabel('location on ring-stent', fontsize=14)
        ax1.set_ylabel('distance to left renal (mm)', fontsize=14)
        plt.ylim(-10,12)
        xlabels = sheet.rows[rowstartleft-1][col:col+4] # VR PA VL PP
        xlabels = [obj.value for obj in xlabels]
        legend = ['discharge', '1month', '6months', '12months']#, '24months'] first adjust excel
        xrange = range(1,1+len(xlabels)) # not start from x=0 to play with margin xlim
        for i, zDistances in enumerate(zDistances_ctcodes):
            ax1.plot(xrange, zDistances, linestyle='--', marker='o', label=legend[i])
            # # plot smooth with spline interpolation
            # xnew = np.linspace(1,4,25)
            # f2 = interp1d(xrange, zDistances, kind='quadratic')
            # plt.plot(xnew,f2(xnew))
        plt.xticks(xrange, xlabels, fontsize = 14)
        plt.xlim(0.8,len(xlabels)+0.2) # xlim margins 0.2
        ax1.legend(loc='best')
        for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
            label.set_fontsize(14)


    def plot_pp_vv_deployment(self, ring=1, saveFig=True):
        """ Plot multipanel pp and vv deployment residu per patient; 
        show (a)symmetry; ring=1 or 2 for R1 or R2
        """
        
        exceldir = self.exceldir
        workbook_stent = self.workbook_stent
        workbook_vars = self.workbook_variables
        
        wb = openpyxl.load_workbook(os.path.join(exceldir, workbook_stent), data_only=True)
        wbvars = openpyxl.load_workbook(os.path.join(exceldir, workbook_vars), data_only=True) 
        
        # init figure
        f1 = plt.figure(num=2, figsize=(17, 12))
        # plt.xlim(18,34)
        # ax1.plot([0,30],[0,30], ls='--', color='dimgrey')
        xlabels = ['D', '1M', '6M', '12M']
        xrange = range(1,1+len(xlabels)) # not start from x=0 to play with margin xlim
        
        # read data
        colStart = [1, 6] # B, G
        rowStart = 83 # pp
        patients = self.patients
        
        for i, patient in enumerate(patients):
            if patient == 'LSPEAS_023':
                break
            # init axis
            ax1 = f1.add_subplot(5,3,i+1)
            _initaxis([ax1])
            # ax1.set_xlabel('PP distance (mm)', fontsize=14)
            ax1.set_ylabel('RDC ring (%)', fontsize=14) # ring deployment capacity
            plt.ylim(-10,47)
            
            sheet = wb.get_sheet_by_name(patient)
            
            if ring == 1 or ring == 12:
                # read R1
                ppR1 = sheet.rows[rowStart][colStart[0]:colStart[0]+4] # +4 is read until 12M
                ppR1 = [obj.value for obj in ppR1]
                vvR1 = sheet.rows[rowStart+1][colStart[0]:colStart[0]+4]
                vvR1 = [obj.value for obj in vvR1]
                # read R2
            if ring == 2 or ring == 12:
                ppR2 = sheet.rows[rowStart][colStart[1]:colStart[1]+4]
                ppR2 = [obj.value for obj in ppR2]
                vvR2 = sheet.rows[rowStart+1][colStart[1]:colStart[1]+4]
                vvR2 = [obj.value for obj in vvR2]
            
            # plot R1
            if ring == 1:
                ax1.plot(xrange, ppR1, ls='-', marker='o', color='#ef8a62', label='PP - R1')
                ax1.plot(xrange, vvR1, ls='-', marker='o', color='#67a9cf', label='VV - R1')
            if ring == 2:
                ax1.plot(xrange, ppR2, ls='-', marker='o', color='#ef8a62', label='PP - R2')
                ax1.plot(xrange, vvR2, ls='-', marker='o', color='#67a9cf', label='VV - R2')
            if ring == 12:
                ax1.plot(xrange, ppR1, ls='-', marker='o', color='#ef8a62', label='PP - R1')
                ax1.plot(xrange, vvR1, ls='-', marker='o', color='#67a9cf', label='VV - R1')
                ax1.plot(xrange, ppR2, ls='--', marker='^', color='#ef8a62', label='PP - R2')
                ax1.plot(xrange, vvR2, ls='--', marker='^', color='#67a9cf', label='VV - R2')
                
            
            plt.xticks(xrange, xlabels, fontsize = 14)
            plt.xlim(0.8,len(xlabels)+0.2) # xlim margins 0.2
            ax1.legend(loc='upper right', fontsize=8, title=('%i: ID %s' % (i+1, patient[-3:]) )  )
        
        if saveFig:
            plt.savefig(os.path.join(self.dirsaveIm, 
            'plot_pp_vv_deployment_R{}.png'.format(ring)), papertype='a0', dpi=300)

    
    def plot_ring_deployment(self, patients=None, saveFig=True):
        """ Plot residual deployment capacity ring, mean peak and valley diameters
        """
        
        exceldir = self.exceldir
        workbook_stent = self.workbook_stent
        workbook_vars = self.workbook_variables
        
        wb = openpyxl.load_workbook(os.path.join(exceldir, workbook_stent), data_only=True)
        wbvars = openpyxl.load_workbook(os.path.join(exceldir, workbook_vars), data_only=True)
        
        # init figure
        f1 = plt.figure(num=3, figsize=(11.6, 6.3))
        xlabels = ['Pre','D', '1M', '6M', '12M']
        xrange = range(1,1+len(xlabels)) # not start from x=0 to play with margin xlim
        
        # init axis
        ax1 = f1.add_subplot(1,2,1)
        plt.xticks(xrange, xlabels, fontsize = 14)
        ax2 = f1.add_subplot(1,2,2)
        plt.xticks(xrange, xlabels, fontsize = 14)
        
        ax1.set_ylabel('Residual deployment capacity R1 (%)', fontsize=15) # ring deployment capacity
        ax2.set_ylabel('Residual deployment capacity R2 (%)', fontsize=15) # ring deployment capacity
        ax1.set_ylim([0, 42])
        ax2.set_ylim([0, 42])
        ax1.set_xlim([0.8, len(xlabels)+0.2]) # xlim margins 0.2
        ax2.set_xlim([0.8, len(xlabels)+0.2])
        
        # lines and colors; 12-class Paired
        colors = itertools.cycle(['#a6cee3','#1f78b4','#b2df8a','#33a02c',
        '#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928'])
        # lStyles = ['-', '--']
        # mStyles = itertools.cycle(['^', '^'])#'D', 's', '+'])
        # marker = 'o'
        markers = ['D', 'o', '^', 's', '*']
        lw = 3
        
        # read data
        colStart = [1, 6] # B, G
        rowStart = 82 # mean pp vv
        if patients is None:
            patients = self.patients
        # loop through patient sheets
        for i, patient in enumerate(patients):
            sheet = wb.get_sheet_by_name(patient)
            # read R1/R2
            if patient == ('LSPEAS_023' or 'LSPEAS_024'):
                R1 = sheet.rows[20][1:5] # D to 12 M; prox part of freeflow ring
                R2 = sheet.rows[21][1:5] # D to 12 M; dist part of freeflow ring
            else:    
                R1 = sheet.rows[rowStart][colStart[0]:colStart[0]+4] # +4 is read until 12M
                R2 = sheet.rows[rowStart][colStart[1]:colStart[1]+4] # +4 is read until 12M
            R1 = [obj.value for obj in R1]
            R2 = [obj.value for obj in R2]
            # read preop applied oversize
            sheet_preop = wbvars.get_sheet_by_name('preCTA_bone_align')
            rowPre = 8 # row 9 in excel
            for j in range(18): # read sheet column with patients
                pt = sheet_preop.rows[rowPre+j][1]
                pt = pt.value
                if pt == patient[-3:]:
                    preR1 = sheet_preop.rows[rowPre+j][19]
                    preR1 = preR1.value
                    preR2 = sheet_preop.rows[rowPre+j][20]
                    preR2 = preR2.value
                    devicesize = sheet_preop.rows[rowPre+j][4].value # col E
                    break
            # plot
            ls = '-'
            color = next(colors)
            # if i > 11: # through 12 colors
            #     marker = next(mStyles)
            if devicesize == 25.5:
                marker = markers[0]
            elif devicesize == 28:
                marker = markers[1]
            elif devicesize == 30.5:
                marker = markers[2]
            elif devicesize == 32:
                marker = markers[3]
            else:
                marker = markers[4]
            if patient == 'LSPEAS_004': # FEVAR
                color = 'k'
                ls = ':'
            elif patient == 'LSPEAS_023': # endurant
                color = 'k'
                ls = '-.'
            
            # when scans are not scored in excel do not plot '#DIV/0!'
            R1 = [el if not isinstance(el, str) else None for el in R1]
            R2 = [el if not isinstance(el, str) else None for el in R2]
            ax1.plot(xrange[1:], R1, ls=ls, lw=lw, marker=marker, color=color, 
            label='%i: ID %s' % (i+1, patient[-3:]))
            ax2.plot(xrange[1:], R2, ls=ls, lw=lw, marker=marker, color=color, 
            label='%i: ID %s' % (i+1, patient[-3:]))
            
            if not isinstance(preR1, str): # not yet scored in excel so '#DIV/0!'
                ax1.plot(xrange[:2], [preR1,R1[0]], ls='--', marker=marker, color=color)
            if not isinstance(preR2, str):
                ax2.plot(xrange[:2], [preR2,R2[0]], ls='--', marker=marker, color=color)
            
        ax2.legend(loc='upper right', fontsize=8, numpoints=1, title='Patients'  )
        _initaxis([ax1, ax2])
        
        if saveFig:
            plt.savefig(os.path.join(self.dirsaveIm, 
            'plot_ring_deployment.png'), papertype='a0', dpi=300)
        
    
    def change_in_rdc_D_12(self, rowStart = 51, rowEnd = 66):
        """ Do peaks expand more than valleys?
        """
        exceldir = self.exceldir
        workbook_stent = self.workbook_stent
        
        wb = openpyxl.load_workbook(os.path.join(exceldir, workbook_stent), data_only=True)
        sheet = wb.get_sheet_by_name('Summary')
        
        # read data
        colStart = ['DY', 'DZ', 'EA', 'EB']#, 'X'] # R1 pp vv R2 pp vv
        colStart = [(openpyxl.cell.column_index_from_string(char)-1) for char in colStart]
        rowStart = rowStart # pt 001 summery sheet
        rowEnd = rowEnd
        nrows = rowEnd - rowStart + 1 
        
        # get arrays with change in rdc all patients
        R1pp = sheet.columns[colStart[0]][rowStart:rowStart+nrows] # 
        R1pp = [obj.value for obj in R1pp]
        R1pp004 = R1pp.pop(3) # pt 004 is in between, remove from list
        R1vv = sheet.columns[colStart[1]][rowStart:rowStart+nrows] # 
        R1vv = [obj.value for obj in R1vv]
        R1vv004 = R1vv.pop(3) # pt 004 is in between, remove from list
        R2pp = sheet.columns[colStart[2]][rowStart:rowStart+nrows] # 
        R2pp = [obj.value for obj in R2pp]
        R2pp004 = R2pp.pop(3) # pt 004 is in between, remove from list
        R2vv = sheet.columns[colStart[3]][rowStart:rowStart+nrows] # 
        R2vv = [obj.value for obj in R2vv]
        R2vv004 = R2vv.pop(3) # pt 004 is in between, remove from list
        
        # boxplot
        data = [R1pp, R1vv, R2pp, R2vv]
        labels = ['R1 peaks', 'R1 valleys', 'R2 peaks', 'R2 valleys']
        
        import plotly
        from plotly.offline import plot
        import plotly.graph_objs as go
        # https://plot.ly/python/box-plots/
        # https://plot.ly/python/axes/
        
        x = ['R1'] * len(R1pp) + ['R2'] * len(R2pp)
        
        trace0 = go.Box(
            y= R1pp + R2pp,
            x = x,
            name = 'peaks',
            boxmean=True,
            fillcolor= 'white', # default is half opacity of line/marker color
            line = dict(
                    color='#ef8a62' # orange/red
            ),
            marker=dict(
                    color='#ef8a62' # orange/red
            ) 
        )
        trace1 = go.Box(
            y= R1vv + R2vv,
            x = x,
            name = 'valleys',
            boxmean=True,
            fillcolor= 'white',
            line = dict(
                    color='#000000'
            ),
            marker=dict(
                    color='#000000'
            )
        )
        data = [trace0, trace1]
        layout = go.Layout(
            yaxis=dict(
                title='Decrease residual deployment capacity ring (%)',
                zeroline=False,
                dtick=5, # steps of 5 on y ax
                tickfont=dict(
                    # family='Old Standard TT, serif',
                    size=26, # size of y ax numbers
                ),
                titlefont = dict(
                        # family='Arial, sans-serif',
                        size=35
                    ),
                
            ),
            xaxis=dict(
                    # titlefont=dict(
                    #     # family='Arial, sans-serif',
                    #     size=25,
                    # ),
                    tickfont=dict(
                        # family='Old Standard TT, serif',
                        size=35, # size of x labels
                    ),
                ),
            legend=dict(
                    font=dict(
                        # family='sans-serif',
                        size=35,
                    ),
                    bordercolor='black',
                    borderwidth=1
                ),
            
            boxmode='group'
        )
        
        fig = go.Figure(data=data, layout=layout)
        plot(fig, image = 'png', image_filename = 'testbox', image_height=1100, image_width=1500) # in w/h pixels)
        
        # plot(data, image = 'png', image_filename = 'testbox', image_height=600, image_width=800) # in w/h pixels
        
        
    def plot_pp_vv_distance_ratio(self, saveFig=True):
        """ Plot pp and vv distance ratio
        show asymmetry
        """
        
        exceldir = self.exceldir
        workbook_stent = self.workbook_stent
        
        wb = openpyxl.load_workbook(os.path.join(exceldir, workbook_stent), data_only=True)
        
        # init figure
        f1 = plt.figure(num=3, figsize=(7.6, 5))
        ax1 = f1.add_subplot(111)
        _initaxis([ax1])
        ax1.set_xlabel('Patient number', fontsize=14)
        ax1.set_ylabel('Asymmetry ratio PP/VV distance', fontsize=14)
        plt.ylim(0.6,1.5)
        ax1.plot([0,15],[1,1], ls='--', color='dimgrey')
        fillstyles = ('full', 'left', 'bottom', 'none')
        colors = ('#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494') 
        # html hex codes from http://colorbrewer2.org/#type=sequential&scheme=YlGnBu&n=5
        legend = ['discharge', '1month', '6months', '12months']#, '24months'] first adjust excel
        
        # read data
        colStart = [1, 6] # B, G
        rowStart = 85 # pp/vv ratio
        patients = self.patients
        xrange = range(1,1+len(patients[:16])) # first 15 patients
        
        for i, patient in enumerate(patients):
            if patient == 'LSPEAS_023':
                break
            sheet = wb.get_sheet_by_name(patient)
            # read R1
            R1ratio = sheet.rows[rowStart][colStart[0]:colStart[0]+4] # +4 is read until 12M
            R1ratio = [obj.value for obj in R1ratio]
            # read R2
            R2ratio = sheet.rows[rowStart][colStart[1]:colStart[1]+4] # +4 is read until 12M
            R2ratio = [obj.value for obj in R2ratio]
            
            # plot R1
            for j in range(len(R1ratio)):
                if i == 0: # legend only once
                    plt.plot(xrange[i], R1ratio[j], marker='D', fillstyle=fillstyles[0], 
                         color=colors[j], ls='None', mec='k', mew=1, label=legend[j]) # each timepoint a specific fill
                else:
                    plt.plot(xrange[i], R1ratio[j], marker='D', fillstyle=fillstyles[0], 
                         color=colors[j], ls='None', mec='k', mew=1) # each timepoint a specific fill
        
            plt.xticks(xrange)
            ax1.legend(loc='upper right', numpoints=1, fontsize=12)
        
        if saveFig:
            plt.savefig(os.path.join(self.dirsaveIm, 'plot_pp_vv_distance_ratio.png'), 
            papertype='a0', dpi=300)

        
    def plot_ellipse_pp_vv():
        """
        """
        from stentseg.utils.fitting import sample_ellipse
        import numpy as np
        #todo: wip
        x0, y0 = 0, 0
        r1 = 20 # pp; x
        r2 = 18 # vv; y
        phi = 0
        e = [x0, y0, r1, r2, phi]
        # define line over axis
        dxax1 = np.cos(phi)*r1 
        dyax1 = np.sin(phi)*r1
        dxax2 = np.cos(phi+0.5*np.pi)*r2 
        dyax2 = np.sin(phi+0.5*np.pi)*r2
        # p1ax1, p2ax1 
        r1ax = np.array((x0+dxax1, y0+dyax1), (x0-dxax1, y0-dyax1)) # r1
        # p1ax2, p2ax2 
        r2ax = np.array((x0+dxax2, y0+dyax2), (x0-dxax2, y0-dyax2)) # r2
        
        
        ppax = [[x0-r1*np.cos(phi), x0+r1*np.cos(phi) ], [y0-r1*np.sin(phi), y0+r1*np.sin(phi)]  ] # [x1, x2]  [y1, y2]
        # vvax =  
        
        pp = sample_ellipse(e, N=32) # sample ellipse to get a PointSet r1,r2
        
        f1 = plt.figure(num=4, figsize=(7.6, 5))
        plt.cla()
        ax1 = f1.add_subplot(111)
        _initaxis([ax1])
        ax1.set_xlabel('PP', fontsize=14)
        ax1.set_ylabel('VV', fontsize=14)
        
        ax1.plot(pp[:,0], pp[:,1])
        ax1.axis('equal')
        ax1.plot(r1ax)
        # vv.plot(np.array([p1ax1, p2ax1]), lc='w', lw=2) # major axis
        # vv.plot(np.array([p1ax2, p2ax2]), lc='w', lw=2) # minor axis
        
        ax1.plot(ppax)
        ax1.plot([0, 4], [0, 4], linestyle='-', color='g')
        
    
def _initaxis(axis, legend=None, xlabel=None, ylabel=None, labelsize=16, 
              axsize=15, legendtitle=None):
    """ Set axis for nice visualization
    axis is list such as [ax] or [ax1, ax2]
    legend = None or provide location 'upper right'
    xlabel = 'time (s)'
    """
    for ax in axis:
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
        if not legend is None:
            if not legendtitle is None:
                ax.legend(loc=legend, title=legendtitle)
            else:
                ax.legend(loc=legend)
        if not xlabel is None:
            ax.set_xlabel(xlabel, fontsize=labelsize)
        if not ylabel is None:
            ax.set_ylabel(ylabel, fontsize=labelsize)
        # set fontsize axis numbers
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(axsize)
    plt.tight_layout() # so that labels are not cut off


if __name__ == '__main__':
    
    patients =['LSPEAS_001', 'LSPEAS_002',	'LSPEAS_003', 'LSPEAS_005',	
                'LSPEAS_008', 'LSPEAS_009',	'LSPEAS_011', 'LSPEAS_015',	'LSPEAS_017',	
                'LSPEAS_018', 'LSPEAS_019',	'LSPEAS_020', 'LSPEAS_021',	'LSPEAS_022',
                'LSPEAS_025', 'LSPEAS_004', 'LSPEAS_023']#, 'LSPEAS_024'] 
    
    # create class object for excel analysis
    foo = ExcelAnalysis() # excel locations initialized in class
    # foo.plot_pp_vv_distance_ratio(saveFig=False)
    # foo.plot_pp_vv_deployment(ring=12, saveFig=False)
    foo.plot_ring_deployment(patients=patients, saveFig=False)
    # foo.change_in_rdc_D_12()
    