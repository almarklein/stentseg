""" 
Script to do the registation. This loads in the 10 volumes and
calculates 10 deformation fields from it, which are then stored to disk.
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
# basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
#                      r'D:\LSPEAS\LSPEAS_ssdf',
#                      r'G:\LSPEAS_ssdf_backup')

basedir = select_dir(r'D:\LSPEAS_F\LSPEASF_ssdf')

# Select dataset to register
# ptcode = 'QRM_FANTOOM_20160121'
# ctcode = '12months'
cropnames = ['ringFOV128', 'ring', 'stent']
ptcodes = ['LSPEASF_C_01']
ctcodes = ['D', 'pre']  

for ptcode in ptcodes:
    for ctcode in ctcodes:
        for cropname in cropnames:
            # Load volumes
            try:
                s = loadvol(basedir, ptcode, ctcode, cropname, 'phases')
            except FileNotFoundError:
                continue
            vols = [s['vol%i'%(i*10)] for i in range(10)]
            
            t0 = time.time()
            
            # Initialize registration object
            reg = pirt.reg.GravityRegistration(*vols)
            # Set params
            reg.params.mass_transforms = 2  # 2nd order (Laplacian) triggers more at lines
            reg.params.deform_wise = 'groupwise' # groupwise!
            reg.params.mapping = 'backward'
            reg.params.deform_limit = 1.0
            reg.params.final_scale = 1.0  # We might set this a wee bit lower than 1 (but slower!)
            reg.params.grid_sampling_factor = 0.5 # !! important especially for Laplace !!
            # most important params
            reg.params.speed_factor = 1.0
            reg.params.scale_sampling = 16
            reg.params.final_grid_sampling = 20
            
            # Go!
            reg.register(verbose=1)
            
            # t1 = time.time()
            # print('Registration completed, which took %1.2f min.' % ((t1-t0)/60))
            
            
            # Store registration result
            
            from visvis import ssdf
            
            # Create struct
            s2 = vv.ssdf.new()
            N = len(vols)
            for i in range(N):
                phase = i*10
                s2['meta%i'%phase] = s['meta%i'%phase]
            s2.origin = s.origin
            s2.stenttype = s.stenttype
            s2.croprange = s.croprange
            # Obtain deform fields
            for i in range(N):
                phase = i*10
                fields = [field for field in reg.get_deform(i).as_backward()]
                s2['deform%i'%phase] = fields
            s2.sampling = s2.deform0[0].sampling  # Sampling of deform is different!
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
            for i in range(10):
                phase = i*10
                s_avg['meta%i'%phase] = s['meta%i'%phase]
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
            t_in_min = (t1-t0)/60
            print('Registration completed for %s - %s, which took %1.2f min.' % (ptcode,ctcode, t_in_min))
