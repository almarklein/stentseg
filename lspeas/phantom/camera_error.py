""" Precision of camera reference signal 

"""
# add lspeas folder to pythonpath via shell
from phantom.motion_pattern_error import readCameraExcel, rmse, getFreqCamera
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
    ylim = 0.3
    xlim = (-1,6)
    colSt1 = 'P'
    colSt2 = 'P'
    colSt3 = 'P'
    
    f1 = plt.figure(figsize=(18,11), num=1); plt.clf()
    ax0 = f1.add_subplot(211)
    
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
    
    # offset from t=0
    offsett2 = ttPeriodStarts2[0] 
    offsett1 = ttPeriodStarts1[0]
    offsett3 = ttPeriodStarts3[0] 
    
    time_all_cam1 = np.asarray(time_all_cam1)
    time_all_cam2 = np.asarray(time_all_cam2)
    time_all_cam3 = np.asarray(time_all_cam3)
    
    ttPeriodStarts1 = np.asarray(ttPeriodStarts1)
    ttPeriodStarts2 = np.asarray(ttPeriodStarts2)
    ttPeriodStarts3 = np.asarray(ttPeriodStarts3)
    
    time_all_cam1t0 = time_all_cam1-offsett1
    time_all_cam2t0 = time_all_cam2-offsett2
    time_all_cam3t0 = time_all_cam3-offsett3
    # plot cam signals and start points periods
    ax0.plot(time_all_cam1t0, pos_all_cam1, 'r.-', alpha=0.5, label='camera reference 1')
    ax0.scatter(ttPeriodStarts1-offsett1, np.array(peakmin1)[:,1], color='green')
    ax0.plot(time_all_cam2t0, pos_all_cam2, 'g.-', alpha=0.5, label='camera reference 2')
    ax0.scatter(ttPeriodStarts2-offsett2, np.array(peakmin2)[:,1], color='green')
    # ax0.plot(time_all_cam3t0, pos_all_cam3, 'b.-', alpha=0.5, label='camera reference 3')
    # ax0.scatter(ttPeriodStarts3-offsett3, np.array(peakmin3)[:,1], color='green')
    
    _initaxis([ax0], legend='upper right', xlabel='time (s)', ylabel='position (mm)')
    ax0.set_ylim((-0.02, ylim))
    ax0.set_xlim(xlim)
    
    
##  overlay signals, smallest rmse

ax1 = f1.add_subplot(212)    

peakstart = int(peakmin2[0,0]) # cam2 as ref
peakend = int(peakmin2[-1,0])
tc2, pc2 = time_all_cam2t0[peakstart:peakend], pos_all_cam2[peakstart:peakend]
pc2 = np.asarray(pc2) - min(pc2) # not always zero as min value
amplitudeC2 = max(pc2)
# freqC2,bpmC2,Tc2 = getFreqCamera(tc2,pc2) # see Tcam2

# overlay cam1 on cam2
rmse_val = 10000
for i in range(-3,4): # analyse for 6 cam start points from peak
    istart = int(peakmin1[0,0])+i
    iend = int(peakmin1[-1,0])+i
    
    tc1, pc1 = time_all_cam1t0[istart:iend], pos_all_cam1[istart:iend]
    pc1 = np.asarray(pc1) - min(pc1) # not always zero as min value
    
    tc1shift = tc1 - (tc1[0]-tc2[0]) # for visualisation shift tc1 to start of tc2
    
    # calc errors
    errors = pc2 - pc1
    
    # root mean squared error
    rmse_val_new = rmse(pc1, pc2)
    # print('rms error for lag', i, 'is:', str(rmse_val_new))
    rmse_val = min(rmse_val, rmse_val_new) # keep smallest, better overlay with algorithm
    if rmse_val == rmse_val_new:
        pc1best = pc1
        tc1best = tc1shift
        errors_best = errors
        i_best = i # best lag / overlay

amplitudeC1 = max(pc1best) # same when pc1
print('period was read from index: ', i_best)

# calc differences
rmse_cam1 = rmse_val
amplitudeDiff = amplitudeC2-amplitudeC1 # r2 - r1
Tdiff = Tcam2 - Tcam1

abs_errors_cam1 = [(abs(e)) for e in errors_best]
mean_abs_error_cam1 = np.mean(abs_errors_cam1)
print('rmse of profile=', rmse_cam1)
print('mean abs error of cam1 vs cam 2=', mean_abs_error_cam1)



# vis best overlay    
ax1.plot(tc1best, pc1best, 'r.-', alpha=0.5, label='camera reference 1')
ax1.plot(tc2, pc2, 'g.-', alpha=0.5, label='camera reference 2')

_initaxis([ax1], legend='upper right', xlabel='time (s)', ylabel='position (mm)')
ax1.set_ylim((-0.02, ylim))
ax1.set_xlim(xlim)
