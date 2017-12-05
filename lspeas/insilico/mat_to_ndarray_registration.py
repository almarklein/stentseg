import scipy.io 
import os
import os, time

import numpy as np
import visvis as vv
import pirt.reg # Python Image Registration Toolkit
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel

# Select the ssdf basedir for original data
basedir = select_dir(r'D:\LSPEAS\LSPEAS_ssdf',
                     r'F:\LSPEAS_ssdf_backup')

# Select original dataset for sampling
ptcode = 'LSPEAS_008'
ctcode = 'discharge'
cropname = 'ring'
what = 'phases' 

# Load volumes
orivol = '0'
s = loadvol(basedir, ptcode, ctcode, cropname, what)
vol0 = s['vol'+orivol]
sampling = s['vol'+orivol].sampling # z,y,x
origin = s.origin
sampling2 = (1,1,1) # ground truth Bernard in pixel/voxel coordinates?
origin2 = (0,0,0)

# Load the mat files with original and deformed vol
dirinsilico = select_dir(r'D:\Profiles\koenradesma\Dropbox\UTdrive\LSPEAS\In silico validation\translation_test_vol\translation',
  r'C:\Users\Maaike\Dropbox\UTdrive\LSPEAS\In silico validation\translation_test_vol\translation')

dirsaveinsilico = select_dir(r'D:\LSPEAS\LSPEAS_ssdf_insilico\reg1',
                    r'F:\LSPEAS_ssdf_insilico\reg1')

fileOr = 'datadressed_0p1_noclip'
fileTr = 'datadressedinterpolated_0p1_noclip'
mat = scipy.io.loadmat(os.path.join(dirinsilico, fileOr+'.mat')) # vol 0% was used

mat2 = scipy.io.loadmat(os.path.join(dirinsilico, fileTr+'.mat'))

volOr = mat['DataDressed']
volTr = mat2['DataDressedInterpolated']
volOr = np.swapaxes(volOr, 0,2) # from y,x,z to z,x,y
volTr = np.swapaxes(volTr, 0,2)
volOr = np.swapaxes(volOr, 1,2) # from z,x,y to z,y,x
volTr = np.swapaxes(volTr, 1,2)

# 'Dressed' part gives errors due to intensities -2000 -->
if True: # crop xy
    volOr = volOr[:, 200:-200, 300:-150]
    volTr = volTr[:, 200:-200, 300:-150]
if True: # clip intensities below -1000
    volOr[volOr<-1000] = 0
    volTr[volTr<-1000] = 0
if False: # crop z
    volOr = volOr[2:-2]
    volTr = volTr[2:-2]

# create Aarray volume    
volOr = vv.Aarray(volOr, sampling2, origin2)
volTr = vv.Aarray(volTr, sampling2, origin2)
vols = [volOr, volTr]

if False: # get a slice not vol
    imOr = volOr[int(0.5*volOr.shape[0]),:,:] # pick a z slice in mid volume
    imTr = volTr[int(0.5*volOr.shape[0]),:,:]
    sampling2 = sampling2[1:]
    origin2 = origin2[1:]
    vols = [imOr, imTr]

f = vv.figure(1); vv.clf()
f.position = 0.00, 22.00,  1914.00, 1018.00
clim = (0,1000)
a1 = vv.subplot(131)
vv.volshow(volOr,clim=clim)
vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
vv.title('Im1-Original')

a2 = vv.subplot(132)
vv.volshow(volTr,clim=clim)
vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
vv.title('Im2-Original transformed')

a3 = vv.subplot(133)
a3.daspect = 1, 1, -1
vv.volshow(volOr,clim=clim)
vv.volshow(volTr,clim=clim)
vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
vv.title('Overlay')

a1.camera = a2.camera = a3.camera

## Do registration
t0 = time.time()

# Initialize registration object
reg = pirt.reg.GravityRegistration(*vols)
# Set params
reg.params.mass_transforms = 2  # 2;2nd order (Laplacian) triggers more at lines
reg.params.deform_wise = 'groupwise' # groupwise!
reg.params.mapping = 'backward'
reg.params.deform_limit = 1.0
reg.params.final_scale = 1.0  #1.0 We might set this a wee bit lower than 1 (but slower!)
reg.params.grid_sampling_factor = 0.5 # 0.5 !! important especially for Laplace !!
# most important params
reg.params.speed_factor = 1.0 # ch9 0.5-2.0; ch10 2.0
reg.params.scale_sampling = 16 # 16; nr of iterations per scale level (ch10 16-200)
reg.params.final_grid_sampling = 20 #20; grid sampling ch9/10: 5-15-30/20 
                                    #This parameter is usually best coupled to final_scale

# Go!
reg.register(verbose=1)

# t1 = time.time()
# print('Registration completed, which took %1.2f min.' % ((t1-t0)/60))


# Store registration result

from visvis import ssdf

# Create struct

s2 = vv.ssdf.new()
N = len(vols)
s2.meta = s['meta'+orivol] # from orig vol
s2.origin = s.origin
s2.stenttype = s.stenttype
s2.croprange = s.croprange

# Obtain deform fields
for i in range(N):
    fields = [field for field in reg.get_deform(i).as_backward()] # fields of z,y,x
    s2['deform%i'%i] = fields
s2.sampling = s2.deform0[0].sampling  # Sampling of deform is different!
s2.originDeforms = s2.deform0[0].origin  # But origin is zero
s2.params = reg.params

# Save
name = cropname+fileOr
filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, name, 'deforms')
ssdf.save(os.path.join(dirsaveinsilico, ptcode, filename), s2)
print("deforms saved to disk.")


# Store averaged volume, where the volumes are registered

# Create average volume from *all* volumes deformed to the "center"
N = len(reg._ims)
mean_vol = np.zeros(reg._ims[0].shape, 'float64')
for i in range(N):
    vol, deform = reg._ims[i], reg.get_deform(i) # reg._ims[0]==volOr
    mean_vol += deform.as_backward().apply_deformation(vol)
mean_vol *= 1.0/N

# Create struct
s_avg = ssdf.new()
s_avg.meta = s['meta'+orivol]
s_avg.sampling = sampling2 # z, y, x 
s_avg.origin = origin2
s_avg.samplingOriginal = sampling
s_avg.originOriginal = origin
s_avg.stenttype = s.stenttype
s_avg.croprange = s.croprange
s_avg.vol = mean_vol.astype('float32')
s_avg.params = s2.params

# Save
avg = 'avgreg'
filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, name, avg)
ssdf.save(os.path.join(dirsaveinsilico, ptcode, filename), s_avg)
print("avgreg saved to disk.")

t1 = time.time()
t_in_min = (t1-t0)/60
print('Registration completed for %s - %s, which took %1.2f min.' % (ptcode,ctcode, t_in_min))


## Load deforms and apply deform to registered phases/vols
#load avgreg
name = cropname+fileOr
savg = loadvol(dirsaveinsilico, ptcode, ctcode, name, 'avgreg')
vol = savg.vol

f = vv.figure(2); vv.clf()
f.position = 968.00, 30.00,  944.00, 1002.00
a = vv.subplot(111)
a.daspect = 1, 1, -1
if False:
    t = vv.imshow(vol, clim=(0,2000))
else:
    t = vv.volshow(vol, clim=(0,1000), renderStyle='mip')
    #t.isoThreshold = 600
vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
vv.title('Averaged volume')

# load deforms
s3 = loadvol(dirsaveinsilico, ptcode, ctcode, name, 'deforms')
deformkeys = []
for key in dir(s3):
    if key.startswith('deform'):
        deformkeys.append(key)
deforms = [s3[key] for key in deformkeys] # list with deforms for n phases

# fields = deforms[0] # list 3xAarray zyx
deformfield0 = pirt.DeformationFieldBackward(deforms[0])
deformfield1 = pirt.DeformationFieldBackward(deforms[1])
# from im 2 transform to im 1
deform_tr_to_or = deformfield1.compose(deformfield0.inverse()).as_backward()
volTrBack = deform_tr_to_or.apply_deformation(volTr) # uses pirt deformation.py
# displacement
volTrBack_d_z = deform_tr_to_or.get_field(0)
volTrBack_d_y = deform_tr_to_or.get_field(1)
volTrBack_d_x = deform_tr_to_or.get_field(2)
# from im 1 transform to im 2
deform_or_to_tr = deformfield0.compose(deformfield1.inverse()).as_backward()
volOrBack = deform_or_to_tr.apply_deformation(volOr)
# displacement
volOrBack_d_z = deform_or_to_tr.get_field(0)
volOrBack_d_y = deform_or_to_tr.get_field(1)
volOrBack_d_x = deform_or_to_tr.get_field(2)

# volOrBack = deformfield.inverse().as_backward().apply_deformation(vol) # uses pirt deformation.py
#todo: valueError sampling vs shape?
#todo: how to go from avg to transformed?
# volOrBack = deformfield.inverse().apply_deformation(vol)
# volOrBack = deformfield.apply_deformation(vol)
# volavgFor = deformfield.as_forward().apply_deformation(vol)
# volavgOrBack = deformfield.inverse().as_backward().apply_deformation(volOr)

# todo: original vol --> overlay on transformed image (from avgreg to transformed orig vol)
# vv.figure(3); vv.clf()
# a1 = vv.subplot(131); t1 = vv.volshow(volOrBack)
# a1.daspect = (1, 1, -1)
# vv.title('vol original using deformation')
# # vv.figure(3); vv.clf()
# a2 = vv.subplot(132); t2 = vv.volshow(volOr)
# a2.daspect = (1, 1, -1)
# vv.title('volume original')
# # vv.figure(3); vv.clf()
# a3 = vv.subplot(133); t3 = vv.volshow2((volOr-volOrBack), clim=(-500,500))
# a3.daspect = (1, 1, -1)
# vv.title('difference')
# 
# a1.camera = a2.camera = a3.camera
# t1.clim = t2.clim = 0, 2000
# t3.clim = -500, 500
# 

vv.figure(4); vv.clf()
a1 = vv.gca()
vv.volshow(volOr)
vv.title('original im 1 (Or)')
vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
a1.daspect = 1, 1, -1

vv.figure(5); vv.clf()
a2 = vv.gca()
vv.volshow(volTrBack)
vv.title('image 2 deformed to im 1')
vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')

vv.figure(6); vv.clf()
a3 = vv.gca()
vv.volshow(volOr-volTrBack)
vv.title('difference')
vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')

vv.figure(7); vv.clf()
a4 = vv.gca()
vv.volshow(volTr)
vv.title('original im 2 (Tr)')
vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')

a1.camera = a2.camera = a3.camera = a4.camera



