""" Precision of camera reference signal 

"""
# add lspeas folder to pythonpath via shell
from phantom.motion_pattern_error import readCameraExcel
from phantom.peakdetection import peakdet
from analysis.utils_analysis import _initaxis
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    from stentseg.utils.datahandling import select_dir
    
    # preset dirs
    dirsave =  select_dir(r'C:\Users\Maaike\Desktop','D:\Profiles\koenradesma\Desktop')
    exceldir = select_dir(r'C:\Users\Maaike\Dropbox\UTdrive\LSPEAS\Analysis\Validation robot', 
                  r'D:\Profiles\koenradesma\Dropbox\UTdrive\LSPEAS\Analysis\Validation robot')
    workbookCam1 = 'Grafieken camera matlab meting 21012016.xlsx' # 21/1/2016
    workbookCam2 = '20160215 GRAFIEKEN van camera systeem uit matlab in excel.xlsx' # 22/1/2016
    workbookCam3 = 'Grafieken camera matlab meting 25012016.xlsx' # 25/1/2016
    
    sheetProfile = 'ZB1'
    colSt = 'P'
    
    # read cam1 data
    time_all_cam1, pos_all_cam1 = readCameraExcel(exceldir, workbookCam1, sheetProfile, colSt)
    
    # get local minima as starting points for camera periods
    peakmax1, peakmin1 = peakdet(pos_all_cam1, 0.05)
    if sheetProfile == 'ZB5' or sheetProfile == 'ZB6':
        peakmin1 = peakmin1[::2] # only for B5 and B6 with extra gauss peak
    tt = [time_all_cam1[int(peak)] for peak in peakmin1[:,0]] # get time in s
    T = (tt[-1]-tt[0])/(len(peakmin1)-1) # period of signal
    
    f1 = plt.figure(figsize=(18,5.25))
    ax0 = f1.add_subplot(111)
    ax0.plot(time_all_cam1, pos_all_cam1, 'r.-', alpha=0.5, label='camera reference')