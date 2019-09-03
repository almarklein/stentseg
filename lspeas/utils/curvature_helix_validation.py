""" Curvature calculation validation helix phantom

2019, Maaike A. Koenrades

A CT scanned helix-shaped phantom with a known theoretically calculated curvature was used:
Schuurmann RCL, Kuster L, Slump CH, Vahl A, Van Den Heuvel DAF, Ouriel K, et al. Aortic curvature instead of angulation allows improved estimation of the true aorto-iliac trajectory. Eur J Vasc Endovasc Surg 2016;51(2):216–24. Doi: 10.1016/j.ejvs.2015.09.008.

"""
from stentseg.utils.datahandling import select_dir
import sys, os
import scipy.io
from lspeas.utils.curvature import get_curvatures
import numpy as np
import visvis as vv
from stentseg.utils.centerline import smooth_centerline

filedir = select_dir(r'C:\Users\Maaike\SURFdrive\UTdrive\LSPEAS\Analysis\Ring motion\curvature check helix fantoom', 
                    r'D:\Profiles\koenradesma\SURFdrive\UTdrive\LSPEAS\Analysis\Ring motion\curvature check helix fantoom')

filename = 'helix'
matdict = scipy.io.loadmat(os.path.join(filedir, filename+'.mat'))
var = matdict['HELIX']
# show phantom
vv.plot(var)
pp = np.asarray(var[99:1399]) # 100:1400 was used in Matlab implementation to exclude ends (Jaimy)
vv.plot(pp, lc='r', lw=3)

# smooth pp (as in implementation)
smooth_pp = smooth_centerline(pp, 15) # smooth the 'interpolated polygon' to helix shape
vv.figure()
vv.plot(smooth_pp, lc='r', lw=1)

# calc curvature
cv = get_curvatures(smooth_pp)
# convert mm-1 to m-1
cv *= 1000

# skip set of start and end points where curvature cannot be calculated properly
# print(cv)
# print(cv[:30])
# print(cv[-30:])
n_to_skip = 10
cv = cv[n_to_skip:-n_to_skip]

mean_curvature = np.mean(cv)
std_curvature = np.std(cv)
min_curvature = np.min(cv)
max_curvature = np.max(cv)

vv.figure()
ax = vv.gca()
vv.plot(np.arange(len(cv)), cv)
curv_theory = 28.62 # m-1
vv.plot(np.arange(len(cv)), np.ones_like(cv)*curv_theory, lc='r')
ax.SetLimits(rangeY=(0,100))

# error based on theoretical value of curvature
perc_mean_error = (mean_curvature-curv_theory)/curv_theory*100
perc_std_error = std_curvature/curv_theory*100

errors = cv - curv_theory
perc_errors = (cv - curv_theory) / curv_theory *100

mean_error = np.mean(errors)
mean_error_perc = np.mean(perc_errors)
std_error = np.std(errors)
std_error_perc = np.std(perc_errors)

print('Mean error = {} +/- {} m-1'.format(mean_error,std_error))
print('Mean percentage error = {} +/- {} %'.format(mean_error_perc,std_error_perc))

# 1.7 +-/ 2.6 m-1
# 6.0 +/- 9.0 %
