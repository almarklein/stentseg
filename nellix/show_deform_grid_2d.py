""" Show deform grid and checkerboard of 2d registration Nellix data

"""

import os, time 

import numpy as np
import visvis as vv
import pirt.reg # Python Image Registration Toolkit
from stentseg.utils.datahandling import select_dir, loadvol
import scipy
from scipy import ndimage
from lspeas.utils.vis_grid_deform import imagePooper

# Select the ssdf basedir
basedir = select_dir(r'E:\Nellix_chevas\CT_SSDF\SSDF_automated',
            r'D:\Nellix_chevas_BACKUP\CT_SSDF\SSDF_automated')

# Select dataset to register
ptcode = 'chevas_10'
ctcode, nr = '12months', 1
cropname = 'prox'


# Select dataset to register
what = 'phases'
phase1 = 3 # 30% 
phase2 = 9 # 90%
slice = None #70
slicey = 107

# Load volumes
s = loadvol(basedir, ptcode, ctcode, cropname, what)

vols = []
phases = []
for key in dir(s):
    if key.startswith('vol'):
        print(key)
        # create vol with zoom in z-direction
        zscale = (s[key].sampling[0] / s[key].sampling[1]) # z / y
        # resample vol using spline interpolation, 3rd order piecewise polynomial
        vol_zoom = scipy.ndimage.interpolation.zoom(s[key],[zscale,1,1],'float32')
        s[key].sampling = [s[key].sampling[1],s[key].sampling[1],s[key].sampling[2]]
        
        # set scale and origin
        vol_zoom_type = vv.Aarray(vol_zoom, s[key].sampling, s[key].origin)
        
        # get slice 2d
        if slice:
            vol = vol_zoom_type[slice,:,:] # z y x
        elif slicey:
            vol = vol_zoom_type[:,slicey,:] # z y x
        else: # use mid slice z
            vol = vol_zoom_type[int(0.5*volOr.shape[0]),:,:]
        
        phases.append(key)
        vols.append(vol)
        
t0 = time.time()

# Initialize registration object
reg = pirt.reg.GravityRegistration(*vols)

reg.params.mass_transforms = 2  # 2nd order (Laplacian) triggers more at lines
reg.params.speed_factor = 1.0
reg.params.deform_wise = 'groupwise' # groupwise!
reg.params.mapping = 'backward'
reg.params.deform_limit = 1.0
reg.params.final_scale = 1.0  # We might set this a wee bit lower than 1 (but slower!)
reg.params.scale_sampling = 16
reg.params.final_grid_sampling = 20
reg.params.grid_sampling_factor = 0.5 

# Go!
reg.register(verbose=1)

t1 = time.time()
print('Registration completed, which took %1.2f min.' % ((t1-t0)/60))


imagePooper(reg, fname='chevas_10_slicey107_grid15', gridsize=15, phase1=phase1, phase2=phase2)