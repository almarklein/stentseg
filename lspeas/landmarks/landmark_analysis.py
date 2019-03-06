"""Functionality to calculate errors and visualize the landmark validation data
Uses as input .mat files with struct containing 10x3x12 data for each observer and 
patient case. See matcode internship Freija.
"""
#...\LSPEAS\Studenten\M2.2_20161205 Freija Geldof\Freija overdracht\LSPEAS_landmarks

import os
import scipy.io
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol
import numpy as np
from stentseg.utils import PointSet, _utils_GUI, visualization
from stentseg.stentdirect import stentgraph
from visvis import ssdf
from lspeas.utils import loadmatstruct
from lspeas.analysis.utils_analysis import _initaxis
import matplotlib.pyplot as plt


def get_error_case_obs_lm(pAlg, pExperts, case=1, obs=1, lnr=1):
    """ Error for a certain case, observer and landmarks
    case = 1; lnr (landmarknr) = 1; obs = 1
    """
    obs = 'o'+str(obs)
    case = 'p'+str(case)
    ppA = pAlg[obs][case][:,:,lnr-1] # 10x3 from 10x3x12
    ppE = pExperts[obs][case][:,:,lnr-1]
    error_x = abs(ppA[:,0] - ppE[:,0]) # 10x1; error per phase for x
    error_y = abs(ppA[:,1] - ppE[:,1]) # 10x1
    error_z = abs(ppA[:,2] - ppE[:,2]) 
    error_xyz = np.hstack((np.vstack(error_x), np.vstack(error_y), np.vstack(error_z))) # stack column wise
    error_3d = np.vstack(np.linalg.norm(ppA-ppE, axis=1)) # 10x1 - euclidian distance
    
    return error_xyz, error_3d

def get_error_case_obs(pAlg, pExperts, case=1, obs=1, Lnrs=[1]):
    """ Error for a certain case and observer and all given landmarks
    Lnrs = [1,2,3..]
    """
    # errors_xyz = []
    # errors_3d = []
    for i, lnr in enumerate(Lnrs):
        error_xyz, error_3d = get_error_case_obs_lm(pAlg, pExperts, case, obs, lnr)
        # errors_xyz.append(error_xyz)
        # spread = np.concatenate([a for a in abs_errors_profile], axis=0)
        
        if i == 0:
            # errors_xyz = [error_xyz] # for np.append
            errors_xyz = error_xyz # for vstack
            errors_3d = error_3d
        else:
            # errors_xyz = np.append(errors_xyz, [error_xyz], axis=0)
            errors_xyz = np.vstack((errors_xyz, error_xyz)) # stack landmarks row wise
            #(to 20x10x3 if 2 landmarks were given)
            errors_3d = np.vstack((errors_3d, error_3d))
    
    return errors_xyz, errors_3d # ndarrays (nx10)x3 and (nx10)x1 with n nr of landmarks 
    
def get_error_case(pAlg, pExperts, case=1, Obs=[1], Lnrs=[1]):
    """ Error for a certain case, all given landmarks and all given observers
    Obs = [1,2..]; Lnrs = [1,2,3..]
    """
    errors_xyz_all = []
    errors_3d_all = []
    for i, obs in enumerate(Obs):
        errors_xyz, errors_3d = get_error_case_obs(pAlg, pExperts, case, obs, Lnrs)
        errors_xyz_all.append(errors_xyz)
        errors_3d_all.append(errors_3d)
        
    return errors_xyz_all, errors_3d_all


## Select dataset landmarks
ptcode = 'LSPEAS_002' # 002_6months, 008_12months, 011_discharge, 17_1month, 20_discharge, 25_12months
ctcode = '6months'
cropname = 'stent' # use stent crops
what = 'landmarksphases' # 'landmarksphases' or 'landmarksavgreg'

dirlandmarks = r'D:\Profiles\koenradesma\SURFdrive\UTdrive\LSPEAS\Analysis\Landmark Validation\LSPEAS'
# where to save figures
dirsave =  select_dir(r'C:\Users\Maaike\Desktop','D:\Profiles\koenradesma\Desktop')
savefig = False

# Load the mat file dictionaries
mat1 = loadmatstruct.loadmat(os.path.join(dirlandmarks, 'landmarkpointsavgregalgorithm'+'.mat'))
mat2 = loadmatstruct.loadmat(os.path.join(dirlandmarks, 'landmarkpointsphases'+'.mat'))

# Get the variables
pAlg = mat1['algorithm'] # dict with obersers o1,o2..
pExperts = mat2['landmarks'] # reference, dict with obersers o1,o2..


## Example for one case for plotting
if True:
    pAlg_o1_p1 = pAlg['o1']['p1'] # 10x3x12 get array all landmarks from dict o1.p1
    pExperts_o1_p1 = pExperts['o1']['p1']
    
    landmarknr = 12
    ppA = pAlg['o1']['p1'][:,:,landmarknr-1] # 10x3
    ppE = pExperts['o1']['p1'][:,:,landmarknr-1]
    
    # make relative to position at 0%
    ppAr = ppA - ppA[0]
    ppEr = ppE - ppE[0]
    ppAdisR = np.linalg.norm(ppA-ppA[0], axis=1)
    ppEdisR = np.linalg.norm(ppE-ppE[0], axis=1)
    
    # error landmark level
    error_x = abs(ppA[:,0] - ppE[:,0]) # 10x1; error per phase for x
    error_y = abs(ppA[:,1] - ppE[:,1]) 
    error_z = abs(ppA[:,2] - ppE[:,2]) 
    error_3d = np.linalg.norm(ppA-ppE, axis=1)
    
    # plot relative position for one landmark x-y-z
    fignum = 1
    f1 = plt.figure(figsize=(9,5.5), num=fignum); plt.clf()
    ax1 = f1.add_subplot(111)
    
    colors = ['#d7191c','#fdae61','#2c7bb6']
    marker = 's'
    xrange = range(0,100,10)
    xlim = 95
    ylim = 1.5
    
    ax1.plot(xrange, ppAr[:,0], color=colors[0], marker=marker, label='algorithm - x')
    ax1.plot(xrange, ppAr[:,1], color=colors[1], marker=marker, label='algorithm - y')
    ax1.plot(xrange, ppAr[:,2], color=colors[2], marker=marker, label='algorithm - z')
    
    ax1.plot(xrange, ppEr[:,0], '--', color=colors[0], marker=marker, label='reference - x')
    ax1.plot(xrange, ppEr[:,1], '--', color=colors[1], marker=marker, label='reference - y')
    ax1.plot(xrange, ppEr[:,2], '--', color=colors[2], marker=marker, label='reference - z')
    
    _initaxis([ax1], legend='upper right', xlabel='phases of cardiac cycle', ylabel='relative position (mm)',
            legendtitle='Landmark {}'.format(landmarknr))
    major_ticksx = np.arange(0, xlim, 10)
    major_ticksy = np.arange(-1, ylim, 0.2)
    #ax4.set_ylim((0, ylim))
    ax1.set_xlim(-0.5,xlim)
    ax1.set_xticks(major_ticksx)
    ax1.set_yticks(major_ticksy)
    
    
    # plot relative 3D displacement for one landmark
    fignum = 2
    f2 = plt.figure(figsize=(9,5.5), num=fignum); plt.clf()
    ax1 = f2.add_subplot(111)
    
    marker = 's'
    xlim = 95
    ylim = 1.5
    
    ax1.plot(xrange, ppAdisR, color=colors[0], marker=marker, label='algorithm')
    ax1.plot(xrange, ppEdisR, '--', color=colors[2], marker=marker, label='reference')
    
    _initaxis([ax1], legend='upper right', xlabel='phases of cardiac cycle', ylabel='displacement 3D (mm)',
            legendtitle='Landmark {}'.format(landmarknr))
    major_ticksx = np.arange(0, xlim, 10)
    major_ticksy = np.arange(-1, ylim, 0.2)
    ax1.set_xlim(-0.5,xlim)
    ax1.set_xticks(major_ticksx)
    ax1.set_yticks(major_ticksy)


## Get errors for a case for given observers and landmarks 
# Lnrs = [1,2,3,4,5,6,7,8,9,10,11,12]
Lnrs = [1]
# Lnrs = [4,5,6]
# Lnrs = [7,8,9]
# Lnrs = [10,11,12]
# Obs = [1,2,3,4,5,6,7,8,9,10,11]
Obs = [1,2,3]

case = 1
cases = [1]


for case in cases:
    # error_xyz, error_3d = get_error_case_obs_lm(pAlg, pExperts, case=5, obs=1, lnr=1 )
    errors_xyz_all, errors_3d_all = get_error_case(pAlg, pExperts, case=case, Obs=Obs, Lnrs=Lnrs)
    
    # Calculate mean, std, bounds of errors_3d_all over all landmarks per observer
    meanerror3d = [np.mean(x) for x in errors_3d_all] # in list the means for n observers
    stdabserror3d = [np.std(x) for x in errors_3d_all]
    maxabserror3d = [np.max(x) for x in errors_3d_all]
    medianabserror3d = [np.percentile(x,50) for x in errors_3d_all]  
    
    # Calculate mean, std, bounds of errors_xyz_all over all landmarks per observer
    meanerrorxyz = [np.mean(x, axis=0) for x in errors_xyz_all] # in list means for x,y,z for n observers
    stderrorxyz = [np.std(x, axis=0) for x in errors_xyz_all]
    maxerrorxyz = [np.max(x, axis=0) for x in errors_xyz_all]
    medianerrorxyz = [np.percentile(x, 50, axis=0) for x in errors_xyz_all]
    
    # Get mean over observers, than calculate mean, std, bounds for errors
    # for errors_3d_all
    error3dAvg = np.mean(np.hstack(errors_3d_all), axis=1)  # mean errors over observers
    meanerror3dAvg = np.mean(error3dAvg)
    stdabserror3dAvg = np.std(error3dAvg)
    maxabserror3dAvg = np.max(error3dAvg)
    medianabserror3dAvg = np.percentile(error3dAvg,50)  
    
    # for errors_xyz_all: get directions separately
    errors_x_all = [obs_array[:,0] for obs_array in errors_xyz_all] # list array
    errors_y_all = [obs_array[:,1] for obs_array in errors_xyz_all]
    errors_z_all = [obs_array[:,2] for obs_array in errors_xyz_all]
    
    # for x
    errorxAvg = np.mean(errors_x_all, axis=0) # get (nlandmarksx10)x1 array
    meanerrorxAvg = np.mean(errorxAvg)
    stdabserrorxAvg = np.std(errorxAvg)
    maxabserrorxAvg = np.max(errorxAvg)
    medianabserrorxAvg = np.percentile(errorxAvg,50)  
    
    # for y
    
    
    
    # for z
    
    # collect in list
    
    


## plot mean errors per case
fignum = 3
f3 = plt.figure(figsize=(9,5.5), num=fignum); plt.clf()
ax1 = f3.add_subplot(111)

xrange = range(1,len(Obs)+1,1)
xlim = 12
ylim = 1.0
boxwidth = 0.5

# plot means
ax1.plot(xrange, error3dAvg, linestyle='', marker='.', color='b') 
# boxes
bp = ax1.boxplot(error3dAvg, 
                positions=xrange, 
                widths=boxwidth,
                patch_artist=True,  # fill with color) 
                ) # dict

for patch in zip(bp['boxes']):
    # patch.set_facecolor(color)
    patch.set(hatch = '/')

_initaxis([ax1], legend='upper right', xlabel='Case', 
        ylabel='registration error (mm)')

major_ticks = np.arange(0, xlim, 1)
ax1.set_ylim((0, ylim))
ax1.set_xlim(-0.04,xlim)
ax1.set_xticks(major_ticks)
# plt.setp(bp['boxes'], color='k')
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], color='k', marker='+')



