""" Precision of camera reference signal 

"""
# add lspeas folder to pythonpath via shell
from lspeas.phantom.motion_pattern_error import readCameraExcel, rmse, getFreqCamera, resample
from lspeas.phantom.peakdetection import peakdet
from lspeas.analysis.utils_analysis import _initaxis
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
    
    sheetProfile = 'ZB6'
    ylim = 2.95 # input + 0.45 marge legend
    xlim = (-1.5,7)
    colSt1 = 'S' # see workbookCams
    colSt2 = 'W'
    colSt3 = 'M'
    
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
    # ax0.scatter(ttPeriodPeaks1-offsett1, np.array(peakmax1)[:,1], color='k')
    ax0.plot(time_all_cam2t0, pos_all_cam2, 'g.-', alpha=0.5, label='camera reference 2')
    ax0.scatter(ttPeriodStarts2-offsett2, np.array(peakmin2)[:,1], color='green')
    # ax0.scatter(ttPeriodPeaks2-offsett2, np.array(peakmax2)[:,1], color='k')
    ax0.plot(time_all_cam3t0, pos_all_cam3, 'b.-', alpha=0.5, label='camera reference 3')
    ax0.scatter(ttPeriodStarts3-offsett3, np.array(peakmin3)[:,1], color='green')
    # ax0.scatter(ttPeriodPeaks3-offsett3, np.array(peakmax3)[:,1], color='k')
    
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
for i in range(-2,3): # analyse for 4 cam start points from peak
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

def getSinglePeriods(peakmin, pos_all_cam,n_samplepoints=None):
    """ based on detected minima get periods in signal
    if n_samplepoints is given, resample points
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
        if n_samplepoints is not None:
            # resample for equal number of points in each period
            ttperiodC1s, pperiodC1s = resample(ttperiodC1,pperiodC1, num=n_samplepoints)
        else: # do not resample
            ttperiodC1s, pperiodC1s = ttperiodC1.copy(), pperiodC1.copy()
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

# plot periods
def show_periods_cam123(ttperiodsC1, ttperiodsC2, ttperiodsC3, pperiodsC1, 
                        pperiodsC2, pperiodsC3, shiftt0=True, ax=ax1):
    
    colors = ['#d7191c','#fdae61','#2c7bb6'] # http://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=5
    for p, ttperiod in enumerate(ttperiodsC1):
        if shiftt0==True:
            offsett = ttperiod[0] # start at t=0
        else:
            offsett = 0 # no shift
        if p == 0:
            ax.plot(ttperiod-offsett, pperiodsC1[p], '.-', color=colors[0], alpha=0.5, label='camera reference 1')
        else:
            ax.plot(ttperiod-offsett, pperiodsC1[p], '.-', color=colors[0], alpha=0.5)
    for p, ttperiod in enumerate(ttperiodsC2):
        if shiftt0==True:
            offsett = ttperiod[0] # start at t=0
        else:
            offsett = 0 # no shift
        if p == 0:
            ax.plot(ttperiod-offsett, pperiodsC2[p], '.-', color=colors[1], alpha=0.5, label='camera reference 2')
        else:
            ax.plot(ttperiod-offsett, pperiodsC2[p], '.-', color=colors[1], alpha=0.5)
    for p, ttperiod in enumerate(ttperiodsC3):
        if shiftt0==True:
            offsett = ttperiod[0] # start at t=0
        else:
            offsett = 0 # no shift
        if p == 0:
            ax.plot(ttperiod-offsett, pperiodsC3[p], '.-', alpha=0.5, color=colors[2], label='camera reference 3')
        else:
            ax.plot(ttperiod-offsett, pperiodsC3[p], '.-', alpha=0.5, color=colors[2])
    
    ax.set_ylim((-0.02, ylim))
    ax.set_xlim(-0.1,max((ttperiod-offsett))+0.1)
    _initaxis([ax], legend='upper right', xlabel='time (s)', ylabel='position (mm)')


f2 = plt.figure(figsize=(19,11), num=2); plt.clf()
ax1 = f2.add_subplot(311)
ax1.set_title('getSinglePeriods-resampled')

show_periods_cam123(ttperiodsC1, ttperiodsC2, ttperiodsC3, pperiodsC1, 
                        pperiodsC2, pperiodsC3, ax=ax1)

## plot average of each cam with bounds, start periods is detected peak, no lag

def show_period_cam123_with_bounds(ttperiodsC1, ttperiodsC2, ttperiodsC3, 
        pperiodsC1, pperiodsC2, pperiodsC3, shiftt0=True, fignum=3):
    
    pperiodsC1mean, pperiodsC1std = np.mean(pperiodsC1, axis=0), np.std(pperiodsC1, axis=0)
    pperiodsC2mean, pperiodsC2std = np.mean(pperiodsC2, axis=0), np.std(pperiodsC2, axis=0)
    pperiodsC3mean, pperiodsC3std = np.mean(pperiodsC3, axis=0), np.std(pperiodsC3, axis=0)
    
    # pperiodsC2q25 = np.percentile(pperiodsC2, 25, axis=0)
    # pperiodsC2q75 = np.percentile(pperiodsC2, 75, axis=0)
    
    # shift to t0 = 0
    if shiftt0 == True:
        ttperiodsC1 = [tt - tt[0] for tt in ttperiodsC1]
        ttperiodsC2 = [tt - tt[0] for tt in ttperiodsC2]
        ttperiodsC3 = [tt - tt[0] for tt in ttperiodsC3]
    # get mean tt per cam
    ttperiodsC1mean = np.mean(ttperiodsC1, axis=0)
    ttperiodsC2mean = np.mean(ttperiodsC2, axis=0)
    ttperiodsC3mean = np.mean(ttperiodsC3, axis=0)
    # get average of all tt periods of the 3 cams
    ttperiodsC123t0 = np.concatenate((ttperiodsC1,ttperiodsC2,ttperiodsC3))
    ttperiodsC123mean = np.mean(ttperiodsC123t0, axis=0)
    
    # get average of all pp periods of the 3 cams
    pperiodsC123 = np.concatenate((pperiodsC1,pperiodsC2,pperiodsC3))
    pperiodsC123mean, pperiodsC123std = np.mean(pperiodsC123, axis=0), np.std(pperiodsC123, axis=0)
    
    # plot
    f3 = plt.figure(figsize=(18,5.5), num=fignum); plt.clf()
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
    ax2.set_xlim(-0.1,max(ttperiodsC3mean)+0.1)
    
    # add plot of average with bounds
    ax3 = f3.add_subplot(122)
    ax3.plot(ttperiodsC123mean, pperiodsC123mean, '.-', color='k', label='camera reference mean')
    ax3.fill_between(ttperiodsC123mean, pperiodsC123mean-pperiodsC123std, pperiodsC123mean+pperiodsC123std, 
                    color='k', alpha=0.2)
    
    _initaxis([ax3], legend='upper right', xlabel='time (s)', ylabel='position (mm)')
    ax3.set_ylim((0, ylim))
    ax3.set_xlim(-0.1,max(ttperiodsC123mean)+0.1)
    
    return ttperiodsC123mean, pperiodsC123mean, pperiodsC123std
           

# show
ttperiodsC123mean, pperiodsC123mean, pperiodsC123std = show_period_cam123_with_bounds(
                                    ttperiodsC1, ttperiodsC2, ttperiodsC3, 
                                    pperiodsC1, pperiodsC2, pperiodsC3, fignum=3)


## overlay all periods individually, best lag

def bestFitPeriods(ttperiodRef, pperiodRef, ttperiodsC, pperiodsC):
    """ given an array with periods (tt and pp) and a ref period,
    overlay each period best on ref (rmse for overlapping parts)
    return shifted ttperiodsC
    """
    ttRef = ttperiodRef - ttperiodRef[0] # shift to t0=0
    ppbest_periods = []
    ttbest_periods = []
    rmse_val_periods = []
    errors_best_periods = []
    for j, period in enumerate(pperiodsC):
        ttperiod = ttperiodsC[j]
        ttperiod = ttperiod - ttperiod[0] # shift to t0=0
        tstep = ttperiod[1]
        rmse_val = 10000
        for i in range(-2,2): # analyse for lag -2, +1 from start of period
            pp = period.copy()
            tt = ttperiod.copy()
            ppRef = pperiodRef.copy()
            if i < 0: # shift pp right
                ppRef = ppRef[-i:] # cut start from ppRef, pp shifted i to right
            elif i > 0: # shift pp left
                pp = pp[i:] # cut start from pp, pp shifted i to left
            # arrays equal size, cut end from longest
            lendif = len(ppRef) - len(pp)
            if lendif > 0:
                ppRef = ppRef[:-lendif]
            elif lendif < 0:
                pp = pp[:lendif]
            # calc errors
            errors = ppRef - pp
            rmse_val_new = rmse(pp, ppRef) # root mean squared error
            rmse_val = min(rmse_val, rmse_val_new) # keep smallest, better overlay with algorithm
            if rmse_val == rmse_val_new: # found better overlay
                errors_best = errors
                i_best = i # best lag / overlay
        
        print(i_best)
        # shift tt for best lag i
        ttbest = ttperiod + tstep*-i_best # if i_best < 0 add time and > 0 visa versa
        # store tt with smallest rmse for each peak/period
        ttbest_periods.append(ttbest)
        rmse_val_periods.append(rmse_val)
        errors_best_periods.append(errors_best)
        
    # return ttbest_periods, ppbest_periods, rmse_val_periods, errors_best_periods
    return ttbest_periods, rmse_val_periods, errors_best_periods

# get periods for each cam without resampling the number of points
ttperiodsC1, pperiodsC1, AperiodsC1, TperiodsC1 = getSinglePeriods(peakmin1, pos_all_cam1)
ttperiodsC2, pperiodsC2, AperiodsC2, TperiodsC2 = getSinglePeriods(peakmin2, pos_all_cam2)
ttperiodsC3, pperiodsC3, AperiodsC3, TperiodsC3 = getSinglePeriods(peakmin3, pos_all_cam3)

# get signal ampl and freq std
AperiodsC1mean, AperiodsC1std = np.mean(AperiodsC1), np.std(AperiodsC1) # n = 7 bv
AperiodsC2mean, AperiodsC2std = np.mean(AperiodsC2), np.std(AperiodsC2)
AperiodsC3mean, AperiodsC3std = np.mean(AperiodsC3), np.std(AperiodsC3)

TperiodsC1mean, TperiodsC1std = np.mean(TperiodsC1), np.std(TperiodsC1)
TperiodsC2mean, TperiodsC2std = np.mean(TperiodsC2), np.std(TperiodsC2)
TperiodsC3mean, TperiodsC3std = np.mean(TperiodsC3), np.std(TperiodsC3)

Aperiods123 = np.concatenate((AperiodsC1, AperiodsC2, AperiodsC3))
AperiodsC123mean = np.mean(Aperiods123)
AperiodsC123std = np.std(Aperiods123)

Tperiods123 = np.concatenate((TperiodsC1, TperiodsC2, TperiodsC3))
TperiodsC123mean = np.mean(Tperiods123)
TperiodsC123std = np.std(Tperiods123)

BPMperiodsC123 = 60/Tperiods123
BPMperiodsC123mean, BPMperiodsC123std = np.mean(BPMperiodsC123), np.std(BPMperiodsC123)

# print output A en T
print('AperiodsC1mean, AperiodsC1std = {}, {}'.format(AperiodsC1mean, AperiodsC1std))
print('AperiodsC2mean, AperiodsC2std = {}, {}'.format(AperiodsC2mean, AperiodsC2std))
print('AperiodsC3mean, AperiodsC3std = {}, {}'.format(AperiodsC3mean, AperiodsC3std))
print('AperiodsC123mean, AperiodsC123std = {}, {}'.format(AperiodsC123mean, AperiodsC123std))
print()
print('TperiodsC1mean, TperiodsC1std = {}, {}'.format(TperiodsC1mean, TperiodsC1std))
print('TperiodsC2mean, TperiodsC2std = {}, {}'.format(TperiodsC2mean, TperiodsC2std))
print('TperiodsC3mean, TperiodsC3std = {}, {}'.format(TperiodsC3mean, TperiodsC3std))
print('TperiodsC123mean, TperiodsC123std = {}, {}'.format(TperiodsC123mean, TperiodsC123std))
print('BPMperiodsC123mean, BPMperiodsC123std = {}, {}'.format(BPMperiodsC123mean, BPMperiodsC123std))
print()

# define reference period
ttperiodRef, pperiodRef = ttperiodsC3[1], pperiodsC3[1] # take mid period cam 1 as ref

# plot periods not resampled with ref pattern to use for lag reference 
ax5 = f2.add_subplot(312) # num = 2
ax5.set_title('getSinglePeriods')
ax5.plot(ttperiodRef-ttperiodRef[0], pperiodRef, '*', color='k', label='camera ref period for lag')
show_periods_cam123(ttperiodsC1, ttperiodsC2, ttperiodsC3, pperiodsC1, 
                        pperiodsC2, pperiodsC3, ax=ax5)

# for each cam overlay periods best
# cam 1, fit each period on ref
ttperiodsC1best, rmse_val_periodsC1, errors_best_periodsC1 = bestFitPeriods(
        ttperiodRef, pperiodRef, ttperiodsC1, pperiodsC1)
# cam 2, fit each period on ref
ttperiodsC2best, rmse_val_periodsC2, errors_best_periodsC2 = bestFitPeriods(
        ttperiodRef, pperiodRef, ttperiodsC2, pperiodsC2)
# cam 3, fit each period on ref
ttperiodsC3best, rmse_val_periodsC3, errors_best_periodsC3 = bestFitPeriods(
        ttperiodRef, pperiodRef, ttperiodsC3, pperiodsC3)


# plot periods cam123, with lag

ax6 = f2.add_subplot(313) # num = 2
ax6.set_title('bestFitPeriods')
show_periods_cam123(ttperiodsC1best, ttperiodsC2best, ttperiodsC3best, pperiodsC1, 
                        pperiodsC2, pperiodsC3, shiftt0=False, ax=ax6)
plt.tight_layout() # so that labels are not cut off

## Get mean of Cam recordings

def cutBefore0equalSizePeriods(ttperiodsCbest,pperiodsC):
    """ cut pp before t=0. make periods equal length by adding NaN to shortest
    periods
    """
    # cut before t=0
    pperiodsCcut = []
    maxlength = 0
    for i, tt in enumerate(ttperiodsCbest):
        pp = pperiodsC[i]
        icut = np.array(np.where(tt==0)[0]).tolist()
        if not icut == [0]: # tpoints before t=0
            if icut: # only pass if list not empty, tt=0 does occur
                pp = pp[icut[0]:]
        #todo: also cut end after TperiodsCmean??
        lengthpp = len(pp)
        maxlength = max(maxlength, lengthpp)
        pperiodsCcut.append(pp)
    
    # add nan's to make equal size for average
    for j, pp in enumerate(pperiodsCcut):
        lendif = maxlength - len(pp)
        if lendif > 0:
            for l in range(lendif):
                pp = np.append(pp, np.nan)
            pperiodsCcut[j] = pp
    
    return pperiodsCcut

# combine cams
ttperiodsC123best = np.concatenate((ttperiodsC1best,ttperiodsC2best,ttperiodsC3best)).tolist()
pperiodsC123 = np.concatenate((pperiodsC1,pperiodsC2,pperiodsC3)).tolist()

pperiodsC1bestCut = cutBefore0equalSizePeriods(ttperiodsC1best,pperiodsC1)
pperiodsC2bestCut = cutBefore0equalSizePeriods(ttperiodsC2best,pperiodsC2)
pperiodsC3bestCut = cutBefore0equalSizePeriods(ttperiodsC3best,pperiodsC3)

pperiodsC123bestCut = cutBefore0equalSizePeriods(ttperiodsC123best,pperiodsC123)


def geTttCamMean(TperiodsC1mean, pperiodsC1bestCut):
    """ with Tmean and number of points in periods (equal length) get ttperiodmean
    """
    num = np.shape(pperiodsC1bestCut)[1] # (nperiods, npoints)[1]
    tt = np.linspace(0, TperiodsC1mean, num) # sampled equidistant
    return tt

ttperiodmeanC1 = geTttCamMean(TperiodsC1mean, pperiodsC1bestCut)
ttperiodmeanC2 = geTttCamMean(TperiodsC2mean, pperiodsC2bestCut)
ttperiodmeanC3 = geTttCamMean(TperiodsC3mean, pperiodsC3bestCut)
ttperiodmeanC123 = geTttCamMean(TperiodsC123mean, pperiodsC123bestCut)


## plot average of each cam with bounds, start periods is best lag position from detected peak

def show_period_cam123bestCut_with_bounds(ttperiodmeanC1, ttperiodmeanC2, 
        ttperiodmeanC3, ttperiodmeanC123, pperiodsC1bestCut, pperiodsC2bestCut, 
        pperiodsC3bestCut, pperiodsC123bestCut, fignum=4, save=True):
    
    from stentseg.utils.aortamotionpattern import get_motion_pattern, plot_pattern
    
    def repeatCamPeriod(tt,aa, aastd, mark=False, **kwargs):
        # make signal 3 periods
        T = tt[-1]
        amax = max(aa)
        
        # Repeats, so that you can see whether the signal is continuous
        aa[0] = 0 # correct for small offset error
        aa3 = np.asarray(list(aa) * 3)
        aa3std = np.asarray(list(aastd) * 3)
        tt3 = []
        for i in range(0, 3):
            tt3 += [t + i*T for t in tt]
        
        # Plot vertical line to mark a single period
        if mark == True:
            ax = plt.gca()
            ax.plot([0, 0], [0, amax], 'k', ls='-', marker = '_')
            ax.plot([T,T], [0, amax], 'k', ls='-', marker = '_')
            ax.plot([2*T, 2*T], [0, amax], 'k', ls='-', marker = '_')
        
        return tt3, aa3, aa3std
    
    # mean and std per cam 
    pperiodsC1mean, pperiodsC1std = np.nanmean(pperiodsC1bestCut, axis=0), np.nanstd(pperiodsC1bestCut, axis=0)
    pperiodsC2mean, pperiodsC2std = np.nanmean(pperiodsC2bestCut, axis=0), np.nanstd(pperiodsC2bestCut, axis=0)
    pperiodsC3mean, pperiodsC3std = np.nanmean(pperiodsC3bestCut, axis=0), np.nanstd(pperiodsC3bestCut, axis=0)
    
    # get average of all pp periods of the 3 cams
    pperiodsC123mean =  np.nanmean(pperiodsC123bestCut, axis=0)
    pperiodsC123std = np.nanstd(pperiodsC123bestCut, axis=0)
    
    # plot
    f3 = plt.figure(figsize=(18,5.5), num=fignum); plt.clf()
    ax2 = f3.add_subplot(121)
    
    # repeat period
    ttperiodmeanC1rep, pperiodsC1mean, pperiodsC1std = repeatCamPeriod(ttperiodmeanC1,
                                        pperiodsC1mean,pperiodsC1std)
    ttperiodmeanC2rep, pperiodsC2mean, pperiodsC2std = repeatCamPeriod(ttperiodmeanC2,
                                        pperiodsC2mean,pperiodsC2std)
    ttperiodmeanC3rep, pperiodsC3mean, pperiodsC3std = repeatCamPeriod(ttperiodmeanC3,
                                        pperiodsC3mean,pperiodsC3std)
    
    # add input function simulator; first run gauspatterns.py
    plot_pattern_plt(*(tt6,aa6),label='Input simulator B6', mark=False)
    
    colors = ['#d7191c','#fdae61','#2c7bb6'] # http://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=5
    ax2.plot(ttperiodmeanC1rep, pperiodsC1mean, '.-', color=colors[0], label='Output simulator day 1 (camera)')
    ax2.fill_between(ttperiodmeanC1rep, pperiodsC1mean-pperiodsC1std, pperiodsC1mean+pperiodsC1std, 
                    color=colors[0], alpha=0.2)
    ax2.plot(ttperiodmeanC2rep, pperiodsC2mean, '.-', color=colors[1], label='Output simulator day 2 (camera)')
    ax2.fill_between(ttperiodmeanC2rep, pperiodsC2mean-pperiodsC2std, pperiodsC2mean+pperiodsC2std, 
                    color=colors[1], alpha=0.3)
    # ax2.fill_between(ttperiodmeanC2rep, pperiodsC2q25, pperiodsC2q75, 
    #                 color=colors[1], alpha=0.3)
    ax2.plot(ttperiodmeanC3rep, pperiodsC3mean, '.-', color=colors[2], label='Output simulator day 3 (camera)')
    ax2.fill_between(ttperiodmeanC3rep, pperiodsC3mean-pperiodsC3std, pperiodsC3mean+pperiodsC3std, 
                    color=colors[2], alpha=0.2)
    
    _initaxis([ax2], legend='upper right', xlabel='time (s)', ylabel='position (mm)')
    ax2.set_ylim((0, ylim))
    # ax2.set_xlim(-0.1,max(ttperiodmeanC3rep)+0.1)
    xlim = 1.5
    major_ticks = np.arange(0, xlim, 0.2)  
    ax2.set_xlim(-0.02,xlim)
    ax2.set_xticks(major_ticks)
    
    # add plot of average with bounds
    ax3 = f3.add_subplot(122)
    # add input function simulator
    plot_pattern_plt(*(tt6,aa6),label='Input simulator B6') # (A1: A=1.0, T=1.2)')
    
    ttperiodmeanC123rep, pperiodsC123meanRep, pperiodsC123stdRep = repeatCamPeriod(ttperiodmeanC123,
                                    pperiodsC123mean,pperiodsC123std, mark=True)
    ax3.plot(ttperiodmeanC123rep, pperiodsC123meanRep, '.-', color='k', label='Output simulator mean (camera)')
    ax3.fill_between(ttperiodmeanC123rep, pperiodsC123meanRep-pperiodsC123stdRep,     
                pperiodsC123meanRep+pperiodsC123stdRep, color='k', alpha=0.2)
    
    _initaxis([ax3], legend='upper right', xlabel='time (s)', ylabel='position (mm)')
    ax3.set_ylim((0, ylim))
    # ax3.set_xlim(-0.1,max(ttperiodmeanC123rep)+0.1)
    ax3.set_xlim(-0.02,xlim)
    ax3.set_xticks(major_ticks)
    
    if save:
        f3.savefig(os.path.join(dirsave, 'simInputOutput.pdf'), papertype='a0', dpi=300)
         
    return pperiodsC123mean, pperiodsC123std


# plot bounds for bestCut periods
pperiodsC123bestCutMean, pperiodsC123bestCutStd = show_period_cam123bestCut_with_bounds(
        ttperiodmeanC1, ttperiodmeanC2, 
        ttperiodmeanC3, ttperiodmeanC123, pperiodsC1bestCut, pperiodsC2bestCut, 
        pperiodsC3bestCut, pperiodsC123bestCut, fignum=4)

