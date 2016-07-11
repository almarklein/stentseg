# Code to analyze error between motion pattern found by algortihm and camera ground truth
 
# Created 4/7/2016
# Maaike Koenrades

import os
import openpyxl # http://openpyxl.readthedocs.org/
import matplotlib.pyplot as plt
import matplotlib as mpl
import prettyplotlib as ppl
from prettyplotlib import brewer2mpl # colormaps
import numpy as np
import scipy
import string

# https://automatetheboringstuff.com/chapter12/
# https://github.com/olgabot/prettyplotlib/wiki/Examples-with-code#fill_between-area-between-two-lines%22
# http://stackoverflow.com/questions/24396589/python-interpretation-on-xcorr


def readCameraExcel(exceldir, workbookCam, sheetProfile, colSt, bpm):
    wb = openpyxl.load_workbook(os.path.join(exceldir, workbookCam))
    sheet = wb.get_sheet_by_name(sheetProfile)
    # r1 = tuple(sheet[colSt+'1':colEnd+'1']) #todo: get period automatically
    # r2 = tuple(sheet[colSt+'2':colEnd+'2'])
    start = col2num(colSt)-1
    r1 = sheet.rows[0][start:]
    r2 = sheet.rows[1][start:]
    time = [obj.value for obj in r1] 
    t0 = time[0]
    tend = t0 + 60/bpm
    tdif = [abs(t-tend) for t in time]
    end = tdif.index(min(tdif)) # first index of time closest to tend
    time = time[:end]
    # time = [obj.value for obj in r1[0]]   
    positions = [obj.value for obj in r2][:end]
    positions = np.asarray(positions)-min(positions) # so that positions have value 0
    
    return time, positions

def readAnalysisExcel(exceldir, workbookAlg, sheetProfile, cols=['G','H','I'], startRows=[18,31,55,68,92,105,129,142]):
    
    import numpy as np
    
    wb = openpyxl.load_workbook(os.path.join(exceldir, workbookAlg), data_only=True)
    sheet = wb.get_sheet_by_name(sheetProfile)
    for startRow in startRows:
        for col in cols: # x,y,z
            p1 = tuple(sheet[col+str(startRow):col+str(startRow+9) ])
            p1 = [obj[0].value for obj in p1]
            p1 = np.vstack(p1)
            if col == cols[0]:
                p1_d = p1
            else:
                p1_d = np.hstack( [ p1_d, p1] )
        if startRow == startRows[0]:
            pp = p1_d
        else: 
            pp = np.append(pp, p1_d, axis=0) 
        
    return pp.reshape(len(startRows),10,len(cols))


def cor_timeshift(cor_seq, y, x):
    """ obtain physical offsets per lag position
    """
    #Generate an x axis
    xcorr = np.arange(cor_seq.size)
    #Convert this into lag units, not physical yet
    lags = xcorr - (y.size-1)
    distancePerLag = (x[-1] - x[0])/float(x.size-1)  # timestep in data
    #Convert lags into physical units
    offsets = -lags*distancePerLag
    
    return offsets, lags

def resample(x,y, num=50, Tnew=None):
    """ Use the univariate interpolators in scipy.interpolate to resample x,y
    Tnew must be =< x[-1]
    """
    from scipy import interpolate
    f = interpolate.interp1d(x, y) # default kind=‘linear’
    xx = np.linspace(x[0], x[-1], num) # sampled equidistant
    if Tnew:
        xx = np.linspace(x[0], Tnew, num) # sampled equidistant & to new T
    yy = f(xx)
    
    return xx, yy

def col2num(col):
    """ convert column in excel to number.
    Input = 'K'; output = 11
    """
    num = 0
    for c in col:
        if c in string.ascii_letters:
            num = num * 26 + (ord(c.upper()) - ord('A')) + 1
    return num

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


if __name__ == '__main__':
    
    from stentseg.utils.datahandling import select_dir
    
    # load excels camera and algorithm motion pattern
    exceldir = select_dir(r'C:\Users\Maaike\Dropbox\UTdrive\LSPEAS\Analysis\Validation robot', 
                  r'D:\Profiles\koenradesma\Dropbox\UTdrive\LSPEAS\Analysis\Validation robot')
    workbookCam = '20160215 GRAFIEKEN van camera systeem uit matlab in excel.xlsx'
    workbookAlg = '20160624 DATA Toshiba.xlsx'
    sheetProfile = 'ZA0'
    # colSt, colEnd = 'CF', 'DK' # 'AP', 'CU' #  'BI', 'CH'
    colSt = 'J' # 2nd first zero/min position
    bpm = 70
    
    time_cam, posCam = readCameraExcel(exceldir, workbookCam, sheetProfile, colSt, bpm)
    pp = readAnalysisExcel(exceldir, workbookAlg, sheetProfile)
    pz = pp[0][:,2]
    pz = np.append(pz, pz[0]) # now point 1, z-axis
    time_pp = np.linspace(0,60/bpm,11) # scale phases to time domain
    time_cam = np.array(time_cam)-time_cam[0] # start at To=0
    # upsample posCam
    time_cam_s, posCam_s = resample(time_cam,posCam, num=len(time_cam)*4) #todo: is factor 4 reasonable?
    
    # downsample camera to 10 positions
    time_cam_s, posCam_s = resample(time_cam,posCam, num=11)
    # %down sample y2 to be same length as y1
    time_cam_sT, posCam_sT = resample(time_cam,posCam, num=11, Tnew=60/bpm)
    # y3 = interp1(x2,y2,x1);%y3 length is 32
    
    # #Test easy 1
    # time_cam_s, posCam_s = time_pp+0.9, pz 
    # Test easy 2
    # time_cam_s, posCam_s = resample((time_pp+0.4),pz, num=len(time_pp)*2)
    
    
    # normalize input arrays to balance amplitudes in array
    pz2 = pz / np.linalg.norm(pz)
    posCam_s2 = posCam_s / np.linalg.norm(posCam_s)
    #todo: subtract mean, divide by std?
    
    # calculcate cross correlation and lag
    cor_seq = np.correlate(pz,posCam_s, mode='full') # second array is shifted; first is the largest array(?)
    #todo: right array order?
    #todo: full or valid or same mode best?
    maxseqI = np.argmax(cor_seq)
    lagdistances, lags = cor_timeshift(cor_seq, posCam_s, time_cam_s)
    shift = lags[maxseqI] # nr of points lag between signals
    print('shift (lag number)=', shift, 'of', lagdistances[maxseqI], 'mm lag')
    #todo: get optimal shift and vertically align data
    # time_cam_shift = time_cam - lagdistances[maxseqI]
    # time_pp_shift = time_pp - (time_pp[-1]-time_cam[0]) # shift Alg to most left
    # time_pp_shift = time_pp - lagdistances[maxseqI]
    # time_pp_shift = time_pp - maxseqI*0.0080128205128204809 #distanceperlag
    distancePerLag = (time_cam_s[-1] - time_cam_s[0])/float(time_cam_s.size-1)  # timestep in data
    time_pp_shift = time_pp + shift*distancePerLag + (time_cam_s[0]-time_pp[0])
    
    
    colormap = brewer2mpl.get_map('YlGnBu', 'sequential', 5).mpl_colormap
    
    f = plt.figure()
    ax1 = f.add_subplot(311)
    ax1.plot(time_cam,posCam, 'ro-', label='camera')
    ax1.plot(time_pp,pz, 'bo-', label='algorithm')
    ax1.plot(time_cam_s, posCam_s, 'ks-', label='camera sampled')
    # ax1.plot(time_pp,pz2, 'bo--', label='algorithm normalized')
    # ax1.plot(time_cam_s, posCam_s2, 'r.--', label='camera sampled normalized')
    plt.legend()
    plt.xlabel('time (s)')
    plt.ylabel('position (mm)')
    ax2 = f.add_subplot(312)
    ax2.plot(lags,cor_seq, 'go-', label='cross corr sequence')
    plt.xlabel('lag position')
    plt.ylabel('correlation measure')
    plt.legend()
    ax3 = f.add_subplot(313, sharex=ax1, sharey=ax1)
    ax3.plot(time_pp_shift,pz, 'yo--', label='algorithm shifted')
    ax3.plot(time_cam_s,posCam_s, 'ks-', label='camera sampled')
    plt.legend()
    plt.xlabel('time (s)')
    plt.ylabel('position (mm)')
    
    #todo: subtract camera displacement from algortithm displacement/ root mean square error?
    # http://dsp.stackexchange.com/questions/14306/percentage-difference-between-two-signals
    # http://stackoverflow.com/questions/17197492/root-mean-square-error-in-python
    
    rmse_val = rmse(posCam_s, pz)
    print("rms error is: " + str(rmse_val))
    
    for pos in pz:
        v = posCam-pos
        d = [np.linalg.norm(p) for p in v]

    # # subtract area, get overlap?
    # from scipy import integrate
    # posCam_int = integrate.cumtrapz(posCam, time_cam, initial=0)
    # pz_int = integrate.cumtrapz(pz, time_pp, initial=0)
    # Id = posCam_int - pz_int #todo: must be same length
    # 
    # f2 = plt.figure()
    # a1 = f2.add_axes()
    # a1.plot(time_cam,Id)
    
    # calc mean, min, max, std
    pz_mean = np.mean(pz)
    pz_std = np.std(pz)
    posCam_mean = np.mean(posCam)
    posCam_std = np.std(posCam)
    posCam_s_mean = np.mean(posCam_s)
    posCam_s_std = np.std(posCam_s)
    
    # t-test to compare arrays
    from scipy import stats
    tt, pval = stats.ttest_ind(pz, posCam_s) # ind = unpaired t-test; tt=t-statistic for mean
    #todo: _ind? for paired = _rel sample size needs the be equal
    print('t-statistic =', tt)
    print('pvalue =     ', pval)

    
    
    
    
    
    # http://matplotlib.org/examples/pylab_examples/xcorr_demo.html
    # cor = plt.xcorr(posCam, pz, mode='full')
    # 
    # 
    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax1.xcorr(posCam, pz, usevlines=True, maxlags=50, normed=True, lw=2)
    # ax1.grid(True)
    # ax1.axhline(0, color='black', lw=2)
    # 
    # ax2 = fig.add_subplot(212, sharex=ax1)
    # ax2.acorr(posCam, usevlines=True, normed=True, maxlags=50, lw=2)
    # ax2.grid(True)
    # ax2.axhline(0, color='black', lw=2)
    # 
    # plt.show()
    #     