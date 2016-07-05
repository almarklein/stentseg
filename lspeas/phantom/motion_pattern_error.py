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

# https://automatetheboringstuff.com/chapter12/
# https://github.com/olgabot/prettyplotlib/wiki/Examples-with-code#fill_between-area-between-two-lines%22
# http://stackoverflow.com/questions/24396589/python-interpretation-on-xcorr


def readCameraExcel(exceldir, workbookCam, sheetProfile, colSt, colEnd):
    wb = openpyxl.load_workbook(os.path.join(exceldir, workbookCam))
    sheet = wb.get_sheet_by_name(sheetProfile)
    r1 = tuple(sheet[colSt+'1':colEnd+'1']) #todo: get period automatically
    r2 = tuple(sheet[colSt+'2':colEnd+'2'])
    # r1 = sheet.rows[0][57:84] 
    # r2 = sheet.rows[1][57:84]
    time = [obj.value for obj in r1[0]]  
    positions = [obj.value for obj in r2[0]]
    positions = np.asarray(positions)-min(positions) # positions to have 0 value
    
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
    #Generate an x axis
    xcorr = np.arange(cor_seq.size)
    #Convert this into lag units, not physical yet
    lags = xcorr - (y.size-1)
    distancePerLag = (x[-1] - x[0])/float(x.size-1)  # timestep in data
    #Convert lags into physical units
    offsets = -lags*distancePerLag
    
    return offsets, lags

def resample(x,y, num=50):
    """ Use the univariate interpolators in scipy.interpolate to resample x,y
    """
    from scipy import interpolate
    f = interpolate.interp1d(x, y) # default kind=‘linear’
    xx = np.linspace(x[0], x[-1], num) # sampled equidistant
    yy = f(xx)
    
    return xx, yy

def normalize_array(a, v):
    """ N
    http://stackoverflow.com/questions/5639280/why-numpy-correlate-and-corrcoef-return-different-values-and-how-to-normalize
    """
    a = (a - np.mean(a)) /   np.std(a)
    v = (v - np.mean(v)) /  (np.std(v)* len(v))
    return a, v


if __name__ == '__main__':
    
    # load excels camera and algorithm motion pattern
    exceldir = r'D:\Profiles\koenradesma\Dropbox\UTdrive\LSPEAS\Analysis\Validation robot'
    workbookCam = '20160215 GRAFIEKEN van camera systeem uit matlab in excel.xlsx'
    workbookAlg = '20160624 DATA Toshiba.xlsx'
    sheetProfile = 'ZA0'
    colSt, colEnd = 'AP', 'CU' #  'BI', 'CH'
    BPM = 70
    
    timeCam, posCam = readCameraExcel(exceldir, workbookCam, sheetProfile, colSt, colEnd)
    # timeCam = np.array(timeCam)-min(timeCam) # start at To=0
    timeCam = np.array(timeCam)-(min(timeCam))
    pp = readAnalysisExcel(exceldir, workbookAlg, sheetProfile)
    time_pp = np.linspace(0,60/BPM,11) # scale phases to time domain
    pz = pp[0][:,2]
    pz = np.append(pz, pz[0]) # now for point 1, z-axis
    # upsample posCam
    # http://stackoverflow.com/questions/20889501/resampled-time-using-scipy-signal-resample
    # http://stackoverflow.com/questions/29085268/resample-a-numpy-array
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html
    timeCam_s, posCam_s = resample(timeCam,posCam, num=len(timeCam)*4) #todo: is factor 4 reasonable?
    
    
    # #Test easy 1
    # time_pp += -0.3
    # timeCam_s, posCam_s = time_pp+0.9, pz 
    # Test easy 2
    # timeCam_s, posCam_s = resample((time_pp+0.4),pz, num=len(time_pp)*2)
    
    # normalize input arrays to balance amplitudes in array
    pz2 = pz / np.linalg.norm(pz)
    posCam_s2 = posCam_s / np.linalg.norm(posCam_s)
    
    # calculcate cross correlation and lag
    cor_seq = np.correlate(pz2,posCam_s2, mode='full') # second array is shifted; first is the largest array(?)
    #todo: right array order?
    maxseqI = np.argmax(cor_seq)
    lagdistances, lags = cor_timeshift(cor_seq, posCam_s2, timeCam_s)
    shift = lags[maxseqI] # nr of points lag between signals
    print('shift (lag number)=', shift, 'of', lagdistances[maxseqI], 'mm lag')
    #todo: get optimal shift and vertically align data
    # timeCam_shift = timeCam - lagdistances[maxseqI]
    # time_pp_shift = time_pp - (time_pp[-1]-timeCam[0]) # shift Alg to most left
    # time_pp_shift = time_pp - lagdistances[maxseqI]
    # time_pp_shift = time_pp - maxseqI*0.0080128205128204809 #distanceperlag
    distancePerLag = (timeCam_s[-1] - timeCam_s[0])/float(timeCam_s.size-1)  # timestep in data
    time_pp_shift = time_pp + shift*distancePerLag + (timeCam_s[0]-time_pp[0])
    
    
    colormap = brewer2mpl.get_map('YlGnBu', 'sequential', 5).mpl_colormap
    
    f = plt.figure()
    ax1 = f.add_subplot(311)
    # ax1.plot(timeCam,posCam, 'ro-', label='camera')
    ax1.plot(time_pp,pz, 'bo-', label='algorithm')
    ax1.plot(timeCam_s, posCam_s, 'r.-', label='camera sampled')
    ax1.plot(time_pp,pz2, 'bo--', label='algorithm normalized')
    ax1.plot(timeCam_s, posCam_s2, 'r.--', label='camera sampled normalized')
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
    ax3.plot(timeCam_s,posCam_s, 'r.-', label='camera sampled')
    plt.legend()
    plt.xlabel('time (s)')
    plt.ylabel('position (mm)')
    
    # subtract camera displacement from algortithm displacement/mean root squared
    
    # calc mean, min, max, std
    
    
    
    
    
    
    
    
    
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