""" Precision of camera reference signal 

"""
# add lspeas folder to pythonpath via shell
from phantom.motion_pattern_error import readCameraExcel
from phantom.peakdetection import peakdet
from analysis.utils_analysis import _initaxis
import matplotlib.pyplot as plt
import numpy as np

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
    colSt1 = 'P'
    colSt2 = 'P'
    colSt3 = 'P'
    
    f1 = plt.figure(figsize=(18,5.5))
    ax0 = f1.add_subplot(111)
    
    # read cam data
    time_all_cam1, pos_all_cam1 = readCameraExcel(exceldir, workbookCam1, sheetProfile, colSt1)
    time_all_cam2, pos_all_cam2 = readCameraExcel(exceldir, workbookCam2, sheetProfile, colSt2)
    time_all_cam3, pos_all_cam3 = readCameraExcel(exceldir, workbookCam3, sheetProfile, colSt3)
    
    # get local minima as starting points for camera periods
    peakmax1, peakmin1 = peakdet(pos_all_cam1, 0.05)
    if sheetProfile == 'ZB5' or sheetProfile == 'ZB6':
        peakmin1 = peakmin1[::2] # only for B5 and B6 with extra gauss peak
    ttPeriodStarts1 = [time_all_cam1[int(peak)] for peak in peakmin1[:,0]] # get time in s
    Tcam1 = (ttPeriodStarts1[-1]-ttPeriodStarts1[0])/(len(peakmin1)-1) # period of signal
    
    peakmax2, peakmin2 = peakdet(pos_all_cam2, 0.05)
    if sheetProfile == 'ZB5' or sheetProfile == 'ZB6':
        peakmin2 = peakmin2[::2] # only for B5 and B6 with extra gauss peak
    ttPeriodStarts2 = [time_all_cam2[int(peak)] for peak in peakmin2[:,0]] # get time in s
    Tcam2 = (ttPeriodStarts2[-1]-ttPeriodStarts2[0])/(len(peakmin2)-1) # period of signal
    
    peakmax3, peakmin3 = peakdet(pos_all_cam3, 0.05)
    if sheetProfile == 'ZB5' or sheetProfile == 'ZB6':
        peakmin3 = peakmin3[::2] # only for B5 and B6 with extra gauss peak
    ttPeriodStarts3 = [time_all_cam3[int(peak)] for peak in peakmin3[:,0]] # get time in s
    Tcam3 = (ttPeriodStarts3[-1]-ttPeriodStarts3[0])/(len(peakmin3)-1) # period of signal
    
    # plot cam signals and start points periods
    ax0.plot(time_all_cam1, pos_all_cam1, 'r.-', alpha=0.5, label='camera reference 1')
    ax0.scatter(ttPeriodStarts1, np.array(peakmin1)[:,1], color='green')
    ax0.plot(time_all_cam2, pos_all_cam2, 'g.-', alpha=0.5, label='camera reference 2')
    ax0.scatter(ttPeriodStarts2, np.array(peakmin2)[:,1], color='green')
    ax0.plot(time_all_cam3, pos_all_cam3, 'b.-', alpha=0.5, label='camera reference 3')
    ax0.scatter(ttPeriodStarts3, np.array(peakmin3)[:,1], color='green')
    
    _initaxis([ax0], legend='upper right', xlabel='time (s)', ylabel='position (mm)')
    
    