""" 
Script to do the registation. This loads in N ecg-gated volumes and
calculates N deformation fields from it, which are then stored to disk.
We can also create an average image from all volumes by first registering
the volumes toward each-other.
"""

## Perform image registration

import os, time

import numpy as np
import visvis as vv
import pirt.reg # Python Image Registration Toolkit
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel

# Select the ssdf basedir
basedir = select_dir(r'D:\LSPEAS\Nellix_chevas\CHEVAS_SSDF',
                    r'G:\Nellix_chevas\CHEVAS_SSDF_BACKUP')

# Select dataset to register
cropname = 'prox'
ptcode = 'chevas_01'
ctcode = '12months'
what = '10phases'


# Load volumes
s = loadvol(basedir, ptcode, ctcode, cropname, what)
vols = []
phases = []
for key in dir(s):
    if key.startswith('vol'):
        print(key)
        phases.append(key)
        vols.append(s[key])

t0 = time.time()

# Initialize registration object
reg = pirt.reg.GravityRegistration(*vols)
#
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

# t1 = time.time()
# print('Registration completed, which took %1.2f min.' % ((t1-t0)/60))


# Store registration result

from visvis import ssdf

# Create struct
s2 = vv.ssdf.new()
N = len(vols)
for key in dir(s):
    if key.startswith('meta'):
        s2[key] = s[key]
s2.origin = s.origin
s2.stenttype = s.stenttype
s2.croprange = s.croprange
# Obtain deform fields
for i in range(N):
    fields = [field for field in reg.get_deform(i).as_backward()]
    phase = phases[i][3:]
    s2['deform%s'%phase] = fields
s2.sampling = s2['deform%s'%phase][0].sampling  # Sampling of deform is different!
# s2.origin = s2.deform0[0].origin  # But origin is zero
s2.params = reg.params

# Save
filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'deforms')
ssdf.save(os.path.join(basedir, ptcode, filename), s2)
print("deforms saved to disk.")

# Store averaged volume, where the volumes are registered

from visvis import ssdf

# Create average volume from *all* volumes deformed to the "center"
N = len(reg._ims)
mean_vol = np.zeros(reg._ims[0].shape, 'float64')
for i in range(N):
    vol, deform = reg._ims[i], reg.get_deform(i)
    mean_vol += deform.as_backward().apply_deformation(vol)
mean_vol *= 1.0/N

# Create struct
s_avg = ssdf.new()
for key in dir(s):
    if key.startswith('meta'):
        s_avg[key] = s[key]
s_avg.sampling = s.sampling  # z, y, x voxel size in mm
s_avg.origin = s.origin
s_avg.stenttype = s.stenttype
s_avg.croprange = s.croprange
s_avg.vol = mean_vol.astype('float32')
s_avg.params = s2.params

# Save
avg = 'avgreg'
filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, avg)
ssdf.save(os.path.join(basedir, ptcode, filename), s_avg)
print("avgreg saved to disk.")

t1 = time.time()
print('Registration completed, which took %1.2f min.' % ((t1-t0)/60))
