""" Functionality for stent analysis

"""

from stentseg.utils import PointSet
import openpyxl
from stentseg.utils.datahandling import select_dir
import sys, os


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
    
    def __init__(self):
        self.exceldir =  ExcelAnalysis.exceldir
        self.workbook_stent = 'LSPEAS_pulsatility_expansion_avgreg_subp_v15.xlsx'
        self.workbook_renal = 'postop_measures_renal_aortic.xlsx'
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
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import prettyplotlib as ppl
        from prettyplotlib import brewer2mpl # colormaps, http://colorbrewer2.org/#type=sequential&scheme=YlGnBu&n=5
        
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
        ax1.spines["top"].set_visible(False)  
        ax1.spines["right"].set_visible(False)
        ax1.get_xaxis().tick_bottom()  
        ax1.get_yaxis().tick_left()
        ax1.set_xlabel('location on ring-stent', fontsize=14)
        ax1.set_ylabel('distance to left renal (mm)', fontsize=14)
        plt.ylim(-10,12)
        xlabels = sheet.rows[rowstartleft-1][col:col+4] # VR PA VL PP
        xlabels = [obj.value for obj in xlabels]
        legend = ['discharge', '1month', '6months', '12months']#, '24months'] first adjust excel
        for i, zDistances in enumerate(zDistances_ctcodes):
            ax1.plot(range(len(xlabels)), zDistances, linestyle='--', marker='o', label=legend[i]) 
        plt.xticks(range(len(xlabels)), xlabels, fontsize = 14)
        ax1.legend(loc='best')
        for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
            label.set_fontsize(14)
                    
        # from scipy.interpolate import spline
        # 
        # xnew = np.linspace(T.min(),T.max(),300)
        # 
        # power_smooth = spline(T,power,xnew)
        # 
        # plt.plot(xnew,power_smooth)
        
        
        
    
    
    
