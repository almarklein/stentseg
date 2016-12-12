""" Precision of camera reference signal 

"""
# add lspeas folder to pythonpath via shell
from phantom.motion_pattern_error import readCameraExcel, rmse, getFreqCamera, resample
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
    
    sheetProfile = 'ZA1'
    ylim = 0.8
    xlim = (-1.5,7)
    colSt1 = 'D'
    colSt2 = 'D'
    colSt3 = 'D'
    
    # read the cam signal with consecutive periods
    f1 = plt.figure(figsize=(18,11), num=1); plt.clf()
    ax0 = f1.add_subplot(211)
    
    # read cam data
    time_all_cam1, pos_all_cam1 = readCameraExcel(exceldir, workbookCam1, sheetProfile, colSt1)
    time_all_cam2, pos_all_cam2 = readCameraExcel(exceldir, workbookCam2, sheetProfile, colSt2)
    time_all_cam3, pos_all_cam3 = readCameraExcel(exceldir, workbookCam3, sheetProfile, colSt3)
    
    # make sure position min value is 0
    pos_all_cam1 = np.asarray(pos_all_cam1) - min(pos_all_cam1)
    pos_all_cam2 = np.asarray(pos_all_cam2) - min(pos_all_cam2)
    pos_all_cam3 = np.asarray(pos_all_cam3) - min(pos_all_cam3)
    
    # get local minima as starting points for camera periods
    peakmax1, peakmin1 = peakdet(pos_all_cam1, 0.05)
    if sheetProfile == 'ZB5' or sheetProfile == 'ZB6':
        peakmin1 = peakmin1[::2] # only for B5 and B6 with extra gauss peak
    ttPeriodStarts1 = [time_all_cam1[int(peak)] for peak in peakmin1[:,0]] # get time in s
    Tcam1 = (ttPeriodStarts1[-1]-ttPeriodStarts1[0])/(len(peakmin1)-1) # period of signal
    ttPeriodPeaks1 = [time_all_cam1[int(peak)] for peak in peakmax1[:,0]] # get time in s
    
    peakmax2, peakmin2 = peakdet(pos_all_cam2, 0.05)
    if sheetProfile == 'ZB5' or sheetProfile == 'ZB6':
        peakmin2 = peakmin2[::2] # only for B5 and B6 with extra gauss peak
    ttPeriodStarts2 = [time_all_cam2[int(peak)] for peak in peakmin2[:,0]] # get time in s
    Tcam2 = (ttPeriodStarts2[-1]-ttPeriodStarts2[0])/(len(peakmin2)-1) # period of signal
    ttPeriodPeaks2 = [time_all_cam2[int(peak)] for peak in peakmax2[:,0]] # get time in s
    
    peakmax3, peakmin3 = peakdet(pos_all_cam3, 0.05)
    if sheetProfile == 'ZB5' or sheetProfile == 'ZB6':
        peakmin3 = peakmin3[::2] # only for B5 and B6 with extra gauss peak
    ttPeriodStarts3 = [time_all_cam3[int(peak)] for peak in peakmin3[:,0]] # get time in s
    Tcam3 = (ttPeriodStarts3[-1]-ttPeriodStarts3[0])/(len(peakmin3)-1) # period of signal
    ttPeriodPeaks3 = [time_all_cam3[int(peak)] for peak in peakmax3[:,0]] # get time in s
    
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
    ttPeriodPeaks1 = np.asarray(ttPeriodPeaks1)
    ttPeriodPeaks2 = np.asarray(ttPeriodPeaks2)
    ttPeriodPeaks3 = np.asarray(ttPeriodPeaks3)
    
    time_all_cam1t0 = time_all_cam1-offsett1
    time_all_cam2t0 = time_all_cam2-offsett2
    time_all_cam3t0 = time_all_cam3-offsett3
    # plot cam signals and start points periods
    ax0.plot(time_all_cam1t0, pos_all_cam1, 'r.-', alpha=0.5, label='camera reference 1')
    ax0.scatter(ttPeriodStarts1-offsett1, np.array(peakmin1)[:,1], color='green')
    ax0.scatter(ttPeriodPeaks1-offsett1, np.array(peakmax1)[:,1], color='k')
    ax0.plot(time_all_cam2t0, pos_all_cam2, 'g.-', alpha=0.5, label='camera reference 2')
    ax0.scatter(ttPeriodStarts2-offsett2, np.array(peakmin2)[:,1], color='green')
    ax0.scatter(ttPeriodPeaks2-offsett2, np.array(peakmax2)[:,1], color='k')
    ax0.plot(time_all_cam3t0, pos_all_cam3, 'b.-', alpha=0.5, label='camera reference 3')
    ax0.scatter(ttPeriodStarts3-offsett3, np.array(peakmin3)[:,1], color='green')
    ax0.scatter(ttPeriodPeaks3-offsett3, np.array(peakmax3)[:,1], color='k')
    
    _initaxis([ax0], legend='upper right', xlabel='time (s)', ylabel='position (mm)')
    ax0.set_ylim((-0.02, ylim))
    ax0.set_xlim(xlim)
    

##  overlay signals with consecutive periods, smallest rmse

ax1 = f1.add_subplot(212)    

numpeaks = min(len(peakmin1), len(peakmin2), len(peakmin3)) # num peaks to analyse

peakstart = int(peakmin2[0,0]) # cam2 as ref
peakend = int(peakmin2[numpeaks-1,0])
tc2, pc2 = time_all_cam2t0[peakstart:peakend+1], pos_all_cam2[peakstart:peakend+1]

# overlay cam1 on cam2
rmse_val1 = 10000
for i in range(-3,5): # analyse for 8 cam start points from peak
    istart = int(peakmin1[0,0])+i
    iend = istart+len(tc2)
    
    tc1, pc1 = time_all_cam1t0[istart:iend], pos_all_cam1[istart:iend]
    
    tc1shift = tc1 - (tc1[0]-tc2[0]) # for visualisation shift tc1 to start of tc2
    
    # calc errors
    errors = pc2 - pc1
    
    # root mean squared error
    rmse_val_new = rmse(pc1, pc2)
    # print('rms error for lag', i, 'is:', str(rmse_val_new))
    rmse_val1 = min(rmse_val1, rmse_val_new) # keep smallest, better overlay with algorithm
    if rmse_val1 == rmse_val_new:
        pc1best = pc1
        tc1best = tc1shift
        errors_best1 = errors
        i_best1 = i # best lag / overlay

# overlay cam3 on cam2
rmse_val3 = 10000
for i in range(-3,5): # analyse for 8 cam start points from peak
    istart = int(peakmin3[0,0])+i
    iend = istart+len(tc2)
    
    tc3, pc3 = time_all_cam3t0[istart:iend], pos_all_cam3[istart:iend]
    
    tc3shift = tc3 - (tc3[0]-tc2[0]) # for visualisation shift tc3 to start of tc2
    
    # calc errors
    errors = pc2 - pc3
    
    # root mean squared error
    rmse_val_new = rmse(pc3, pc2)
    # print('rms error for lag', i, 'is:', str(rmse_val_new))
    rmse_val3 = min(rmse_val3, rmse_val_new) # keep smallest, better overlay with algorithm
    if rmse_val3 == rmse_val_new:
        pc3best = pc3
        tc3best = tc3shift
        errors_best3 = errors
        i_best3 = i # best lag / overlay

print('period cam1 was read from index: ', i_best1)
print('period cam3 was read from index: ', i_best3)

# vis best overlay    
ax1.plot(tc1best, pc1best, 'r.-', alpha=0.5, label='camera reference 1')
ax1.plot(tc2, pc2, 'g.-', alpha=0.5, label='camera reference 2')
ax1.plot(tc3best, pc3best, 'b.-', alpha=0.5, label='camera reference 3')


## amplitude, freq and differences cams with consecutive periods
# amplitude cam signal
amplitudeC1 = max(pc1best) # same as pc1
amplitudeC3 = max(pc3best) # same as pc3
amplitudeC2 = max(pc2)

freqC2,bpmC2,Tc2 = getFreqCamera(tc2,pc2) # see Tcam2
freqC1,bpmC1,Tc1 = getFreqCamera(tc1best,pc1best) # see Tcam1 
freqC3,bpmC3,Tc3 = getFreqCamera(tc3best,pc3best) # see Tcam3 

# calc differences
rmse_cam1 = rmse_val1
amplitudeDiff21 = amplitudeC2-amplitudeC1 # r2 - r1
Tdiff21 = Tcam2 - Tcam1

abs_errors_cam1 = [(abs(e)) for e in errors_best1]
mean_abs_error_cam1 = np.mean(abs_errors_cam1)
print('rmse of signal cam1=', rmse_cam1)
print('mean abs error of cam1 vs cam 2=', mean_abs_error_cam1)

rmse_cam3 = rmse_val3
amplitudeDiff23 = amplitudeC2-amplitudeC3 # r2 - r3
Tdiff23 = Tcam2 - Tcam3

abs_errors_cam3 = [(abs(e)) for e in errors_best3]
mean_abs_error_cam3 = np.mean(abs_errors_cam3)
print('rmse of signal cam3=', rmse_cam3)
print('mean abs error of cam3 vs cam 2=', mean_abs_error_cam3)

## average signal of 3 cams with each still consecutive periods

tcmean = tc2
pcmean = (pc1best+pc3best+pc2)/3

ax1.plot(tcmean, pcmean, 'ko:', alpha=0.5, label='camera reference mean')

_initaxis([ax1], legend='upper right', xlabel='time (s)', ylabel='position (mm)')
ax1.set_ylim((-0.02, ylim))
ax1.set_xlim(xlim)



## get single periods, using detected minima to split periods

n_samplepoints = 30 # 30fps*T

def getSinglePeriods(peakmin, pos_all_cam,n_samplepoints):
    """ based on detected minima get periods in signal
    return tt, pp, amplitude and T for each period
    """
    ttperiodsC = []
    pperiodsC = []
    AperiodsC = []
    TperiodsC = []
    for i in range(len(peakmin)-1):
        peakstart = int(peakmin[i,0]) # first peak start period
        peakend = int(peakmin[i+1,0])+1 # start of second period
        ttperiodC1 = time_all_cam1[peakstart:peakend]
        pperiodC1 = pos_all_cam[peakstart:peakend]
        # resample for equal number of points in each period
        ttperiodC1s, pperiodC1s = resample(ttperiodC1,pperiodC1, num=n_samplepoints)
        # make sure position min value is 0
        pperiodC1s = pperiodC1s - min(pperiodC1s)
        # get amplitudes
        Aperiod = max(pperiodC1s)
        Tperiod = ttperiodC1[-1]-ttperiodC1[0]
        # collect all
        ttperiodsC.append(ttperiodC1s)
        pperiodsC.append(pperiodC1s)
        AperiodsC.append(Aperiod)
        TperiodsC.append(Tperiod)
        
    return ttperiodsC, pperiodsC, AperiodsC, TperiodsC


ttperiodsC1, pperiodsC1, AperiodsC1, TperiodsC1 = getSinglePeriods(peakmin1, pos_all_cam1,n_samplepoints)
ttperiodsC2, pperiodsC2, AperiodsC2, TperiodsC2 = getSinglePeriods(peakmin2, pos_all_cam2,n_samplepoints)
ttperiodsC3, pperiodsC3, AperiodsC3, TperiodsC3 = getSinglePeriods(peakmin3, pos_all_cam3,n_samplepoints)

# get signal ampl and freq std
AperiodsC1mean, AperiodsC1std = np.mean(AperiodsC1), np.std(AperiodsC1) # n = 7 bv
AperiodsC2mean, AperiodsC2std = np.mean(AperiodsC2), np.std(AperiodsC2)
AperiodsC3mean, AperiodsC3std = np.mean(AperiodsC3), np.std(AperiodsC3)

TperiodsC1mean, TperiodsC1std = np.mean(TperiodsC1), np.std(TperiodsC1)
TperiodsC2mean, TperiodsC2std = np.mean(TperiodsC2), np.std(TperiodsC2)
TperiodsC3mean, TperiodsC3std = np.mean(TperiodsC3), np.std(TperiodsC3)

# plot periods
f2 = plt.figure(figsize=(18,5.5), num=2); plt.clf()
ax1 = f2.add_subplot(111)

colors = ['#d7191c','#fdae61','#2c7bb6'] # http://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=5
for p, ttperiod in enumerate(ttperiodsC1):
    offsett = ttperiod[0] # start at t=0
    if p == 0:
        ax1.plot(ttperiod-offsett, pperiodsC1[p], '.-', color=colors[0], alpha=0.5, label='camera reference 1')
    else:
        ax1.plot(ttperiod-offsett, pperiodsC1[p], '.-', color=colors[0], alpha=0.5)
for p, ttperiod in enumerate(ttperiodsC2):
    offsett = ttperiod[0] # start at t=0
    if p == 0:
        ax1.plot(ttperiod-offsett, pperiodsC2[p], '.-', color=colors[1], alpha=0.5, label='camera reference 2')
    else:
        ax1.plot(ttperiod-offsett, pperiodsC2[p], '.-', color=colors[1], alpha=0.5)
for p, ttperiod in enumerate(ttperiodsC3):
    offsett = ttperiod[0] # start at t=0
    if p == 0:
        ax1.plot(ttperiod-offsett, pperiodsC3[p], '.-', alpha=0.5, color=colors[2], label='camera reference 3')
    else:
        ax1.plot(ttperiod-offsett, pperiodsC3[p], '.-', alpha=0.5, color=colors[2])

_initaxis([ax1], legend='upper right', xlabel='time (s)', ylabel='position (mm)')
ax1.set_ylim((-0.02, ylim))
ax1.set_xlim(-0.02,max((ttperiod-offsett))+0.1)

## plot average of each cam with bounds, start periods is detected peak, no lag

pperiodsC1mean, pperiodsC1std = np.mean(pperiodsC1, axis=0), np.std(pperiodsC1, axis=0)
pperiodsC2mean, pperiodsC2std = np.mean(pperiodsC2, axis=0), np.std(pperiodsC2, axis=0)
pperiodsC3mean, pperiodsC3std = np.mean(pperiodsC3, axis=0), np.std(pperiodsC3, axis=0)

pperiodsC2q25 = np.percentile(pperiodsC2, 25, axis=0)
pperiodsC2q75 = np.percentile(pperiodsC2, 75, axis=0)

ttperiodsC1t0 = [tt - tt[0] for tt in ttperiodsC1]
ttperiodsC1mean = np.mean(ttperiodsC1t0, axis=0)
ttperiodsC2t0 = [tt - tt[0] for tt in ttperiodsC2]
ttperiodsC2mean = np.mean(ttperiodsC2t0, axis=0)
ttperiodsC3t0 = [tt - tt[0] for tt in ttperiodsC3]
ttperiodsC3mean = np.mean(ttperiodsC3t0, axis=0)

# get average of all periods of the 3 cams
ttperiodsC123t0 = np.concatenate((ttperiodsC1t0,ttperiodsC2t0,ttperiodsC3t0))
ttperiodsC123mean = np.mean(ttperiodsC123t0, axis=0)
pperiodsC123 = np.concatenate((pperiodsC1,pperiodsC2,pperiodsC3))
pperiodsC123mean, pperiodsC123std = np.mean(pperiodsC123, axis=0), np.std(pperiodsC123, axis=0)

# plot
f3 = plt.figure(figsize=(18,5.5), num=3); plt.clf()
ax2 = f3.add_subplot(121)

colors = ['#d7191c','#fdae61','#2c7bb6'] # http://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=5
ax2.plot(ttperiodsC1mean, pperiodsC1mean, '.-', color=colors[0], label='camera reference 1')
ax2.fill_between(ttperiodsC1mean, pperiodsC1mean-pperiodsC1std, pperiodsC1mean+pperiodsC1std, 
                color=colors[0], alpha=0.2)
ax2.plot(ttperiodsC2mean, pperiodsC2mean, '.-', color=colors[1], label='camera reference 2')
ax2.fill_between(ttperiodsC2mean, pperiodsC2mean-pperiodsC2std, pperiodsC2mean+pperiodsC2std, 
                color=colors[1], alpha=0.3)
# ax2.fill_between(ttperiodsC2mean, pperiodsC2q25, pperiodsC2q75, 
#                 color=colors[1], alpha=0.3)
ax2.plot(ttperiodsC3mean, pperiodsC3mean, '.-', color=colors[2], label='camera reference 3')
ax2.fill_between(ttperiodsC3mean, pperiodsC3mean-pperiodsC3std, pperiodsC3mean+pperiodsC3std, 
                color=colors[2], alpha=0.2)

_initaxis([ax2], legend='upper right', xlabel='time (s)', ylabel='position (mm)')
ax2.set_ylim((0, ylim))
ax2.set_xlim(-0.02,max(ttperiodsC3mean)+0.1)

# add plot of average with bounds
ax3 = f3.add_subplot(122)
ax3.plot(ttperiodsC123mean, pperiodsC123mean, '.-', color='k', label='camera reference mean')
ax3.fill_between(ttperiodsC123mean, pperiodsC123mean-pperiodsC123std, pperiodsC123mean+pperiodsC123std, 
                color='k', alpha=0.2)

_initaxis([ax3], legend='upper right', xlabel='time (s)', ylabel='position (mm)')
ax3.set_ylim((0, ylim))
ax3.set_xlim(-0.02,max(ttperiodsC123mean)+0.1)



## overlay all periods individually, best lag

def bestFitPeriods(ttperiodRef, pperiodRef, ttperiodsC, pperiodsC):
    """ given an array with periods (tt and pp) and a ref period,
    overlay each period best on ref with lag + and - 1. 
    for lag to the right, 0 is added at start and signal is resampled
    for lag to the left, 0 is added at end and cut at the start
    """
    ttRef = ttperiodRef - ttperiodRef[0] # shift to t0=0
    ppbest_periods = []
    ttbest_periods = []
    rmse_val_periods = []
    errors_best_periods = []
    for j, period in enumerate(pperiodsC):
        tt = ttperiodsC[j]
        tt = tt - tt[0] # shift to t0=0
        tstep = (tt[-1]-tt[0])/(len(tt)-1)
        rmse_val = 10000
        for i in range(-1,2): # analyse for lag 1, pos and neg from start of period
            if i < 0:
                pp = period.copy()
                for neg in range(i,0):
                    np.insert(pp, neg, 0) # add zero to start to shift right
                    np.append(tt, tt[-1]+tstep)
                ttS, pperiodS = resample(tt,pp, num=n_samplepoints)
                # calc errors
                errors = pperiodRef - pperiodS
                rmse_val_new = rmse(pperiodS, pperiodRef) # root mean squared error
                rmse_val = min(rmse_val, rmse_val_new) # keep smallest, better overlay with algorithm
                if rmse_val == rmse_val_new:
                    ppbest = pperiodS
                    ttbest = ttS # with t0=0
                    errors_best = errors
                    i_best = i # best lag / overlay
            elif i > 0:
                pp = period.copy()
                for pos in range(1,i+1):
                    pp = np.append(pp, 0) # add zero to end to shift left
                pp = pp[i:] # tt does not change
                # calc errors
                errors = pperiodRef - pp
                rmse_val_new = rmse(pp, pperiodRef) # root mean squared error
                rmse_val = min(rmse_val, rmse_val_new) # keep smallest, better overlay with algorithm
                if rmse_val == rmse_val_new:
                    ppbest = pp
                    ttbest = tt # with t0=0
                    errors_best = errors
                    i_best = i # best lag / overlay
            else:
                errors = pperiodRef - period
                rmse_val_new = rmse(period, pperiodRef) # root mean squared error
                rmse_val = min(rmse_val, rmse_val_new) # keep smallest, better overlay with algorithm
                if rmse_val == rmse_val_new:
                    ppbest = period
                    ttbest = tt # with t0=0
                    errors_best = errors
                    i_best = i # best lag / overlay
        print(i_best)
        # store signal with smallest rmse for each peak/period
        ppbest_periods.append(ppbest) # num of periods equal to j+1
        ttbest_periods.append(ttbest)
        rmse_val_periods.append(rmse_val)
        errors_best_periods.append(errors_best)
    
    return ttbest_periods, ppbest_periods, rmse_val_periods, errors_best_periods

# define reference period
ttperiodRef, pperiodRef = ttperiodsC1[2], pperiodsC1[2] # take 3rd=middle period as ref

# cam 1, fit each period on ref
ttperiodsC1best, pperiodsC1best, rmse_val_periodsC1, errors_best_periodsC1 = bestFitPeriods(
        ttperiodRef, pperiodRef, ttperiodsC1, pperiodsC1)
# cam 2, fit each period on ref
ttperiodsC2best, pperiodsC2best, rmse_val_periodsC2, errors_best_periodsC2 = bestFitPeriods(
        ttperiodRef, pperiodRef, ttperiodsC2, pperiodsC2)
# cam 3, fit each period on ref
ttperiodsC3best, pperiodsC3best, rmse_val_periodsC3, errors_best_periodsC3 = bestFitPeriods(
        ttperiodRef, pperiodRef, ttperiodsC3, pperiodsC3)

## plot average of each cam with bounds, start periods is best lag from detected peak





