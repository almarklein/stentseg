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
cropname = 'ring' # just for fast reading vol for sampling
what = 'phases' 

# Load volumes
oriPhase = '0'
s = loadvol(basedir, ptcode, ctcode, cropname, what)
sampling = s['vol'+oriPhase].sampling # z,y,x
origin = s.origin
sampling2 = (1,1,1) # ground truth Bernard in voxels (with width_pixel=0.1!)
origin2 = (0,0,0)

# Load the mat files with original and deformed volumes
dirinsilico = select_dir(r'D:\LSPEAS\LSPEAS_insilico_mat')

dirsaveinsilico = select_dir(r'D:\LSPEAS\LSPEAS_insilico_ssdf\reg1',
                    r'F:\LSPEAS_ssdf_insilico\reg1')

fileOr = 'original_DataDressed_008_d0'
fileTr = 'tr_DataDressed_008_d0_1_-2.00pi'

mat = scipy.io.loadmat(os.path.join(dirinsilico, ptcode, fileOr+'.mat'))
mat2 = scipy.io.loadmat(os.path.join(dirinsilico, ptcode, fileTr+'.mat'))

volOr = mat['DataDressed']
volTr = mat2['DataDressedInterpolated']
volOr = np.swapaxes(volOr, 0,2) # from y,x,z to z,x,y
volTr = np.swapaxes(volTr, 0,2)
volOr = np.swapaxes(volOr, 1,2) # from z,x,y to z,y,x
volTr = np.swapaxes(volTr, 1,2)

# 'Dressed' part gives errors due to extreme negative intensities (-2000) -->
if False: # crop xy
    ymin, ymax = 200, -200
    xmin, xmax = 300, -150
    croprange = [[0,volOr.shape[0]], [ymin,ymax], [xmin,xmax]] # zyx
    croprangeMat = [[ymin+1,volOr.shape[1]+ymax],[xmin+1,volOr.shape[2]+xmax], [1,volOr.shape[0]]] # yxz
    volOr = volOr[:, ymin:ymax, xmin:xmax]
    volTr = volTr[:, ymin:ymax, xmin:xmax]
else:
    croprange = False
    croprangeMat = 'false'
if False: # crop z
    croprange[0] = [1,-2]
    volOr = volOr[1:-2]
    volTr = volTr[1:-2]
# clip intensities below -1000, set to -200 (HU between fat-lung)
if True: 
    volOr[volOr<-1000] = -200
    volTr[volTr<-1000] = -200

# create Aarray volume    
volOr = vv.Aarray(volOr, sampling2, origin2) #todo: must be sampled with dicom sampling for correct registration?
volTr = vv.Aarray(volTr, sampling2, origin2)
vols = [volOr, volTr]

reg2d = False
if reg2d: # Get a slice for registration not the volume
    imOr = volOr[int(0.5*volOr.shape[0]),:,:] # pick a z slice in mid volume
    imTr = volTr[int(0.5*volOr.shape[0]),:,:]
    sampling2 = sampling2[1:] # keep y,x
    origin2 = origin2[1:]
    vols = [imOr, imTr]

visualize = False
if visualize:
    f = vv.figure(1); vv.clf()
    f.position = 0.00, 22.00,  1914.00, 1018.00
    clim = (-400,1500)
    a1 = vv.subplot(221)
    vv.volshow(volOr,clim=clim)
    vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
    vv.title('Im1:Original')
    
    a2 = vv.subplot(222)
    vv.volshow(volTr,clim=clim)
    vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
    vv.title('Im2:Original transformed')
    
    a3 = vv.subplot(223)
    a3.daspect = 1, 1, -1
    vv.volshow(volOr,clim=clim)
    vv.volshow(volTr,clim=clim)
    vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
    vv.title('Overlay')
    
    a4 = vv.subplot(224)
    
    c = vv.ClimEditor(vv.gcf())
    
    a1.camera = a2.camera = a3.camera = a4.camera

## Do registration

if True:
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
    
    
    # Store registration result
    
    from visvis import ssdf
    
    # Create struct
    
    s2 = vv.ssdf.new()
    N = len(vols)
    s2.meta = s['meta'+oriPhase] # from orig vol
    s2.origin = s.origin #todo: is used in loadvol when deforms are loaded. Correct or deform origin?
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
    filename = '%s_%s.ssdf' % (fileTr, 'deforms')
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
    s_avg.meta = s['meta'+oriPhase]
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
    filename = '%s_%s.ssdf' % (fileTr, avg)
    ssdf.save(os.path.join(dirsaveinsilico, ptcode, filename), s_avg)
    print("avgreg saved to disk.")
    
    t1 = time.time()
    t_in_min = (t1-t0)/60
    print('Registration completed for %s - %s, which took %1.2f min.' % (ptcode,ctcode, t_in_min))


## Load deforms and apply deform to registered phases/vols

if visualize:
    #load avgreg
    filename = fileTr+'_'+avg
    savg = loadvol(dirsaveinsilico, ptcode, ctcode, None, 'avgreg', fname=filename) # sets savg.sampling and savg.origin
    vol = savg.vol
    # show
    #f = vv.figure(2); vv.clf()
    #f.position = 968.00, 30.00,  944.00, 1002.00
    #a = vv.subplot(111)
    #a.daspect = 1, 1, -1
    a4.MakeCurrent()
    if reg2d:
        t = vv.imshow(vol, clim=clim)
    else:
        t = vv.volshow(vol, clim=clim, renderStyle='mip')
        #t.isoThreshold = 600
    vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
    vv.title('Averaged volume')

# load deforms
filename = fileTr+'_'+'deforms'
s3 = loadvol(dirsaveinsilico, ptcode, ctcode, None, 'deforms', fname=filename) # sets s3.sampling and s3.origin
deformkeys = []
for key in dir(s3):
    if key.startswith('deform'):
        deformkeys.append(key)
deforms = [s3[key] for key in deformkeys] # list with deforms for n phases
                                          # deforms[0] # list 3xAarray zyx

# Get deformfields to deform the volumes
deformfield0 = pirt.DeformationFieldBackward(deforms[0])
deformfield1 = pirt.DeformationFieldBackward(deforms[1])
# from im2 (transformed) transform to im1 (original)
deform_tr_to_or = deformfield1.compose(deformfield0.inverse()).as_backward()
volTrBack = deform_tr_to_or.apply_deformation(volTr) # uses pirt deformation.py
# displacement
volTrBack_d_z = deform_tr_to_or.get_field(0)
volTrBack_d_y = deform_tr_to_or.get_field(1)
volTrBack_d_x = deform_tr_to_or.get_field(2)
# from im1 (ori) transform to im2 (transformed)
deform_or_to_tr = deformfield0.compose(deformfield1.inverse()).as_backward()
volOrBack = deform_or_to_tr.apply_deformation(volOr)
# displacement
volOrBack_d_z = deform_or_to_tr.get_field(0)
volOrBack_d_y = deform_or_to_tr.get_field(1)
volOrBack_d_x = deform_or_to_tr.get_field(2)

# store displacement fields
displacementOrToTr = mat2.copy()
del displacementOrToTr['DataDressedInterpolated']
displacementOrToTr['displacementOrToTr_z'] = volOrBack_d_z
displacementOrToTr['displacementOrToTr_y'] = volOrBack_d_y
displacementOrToTr['displacementOrToTr_x'] = volOrBack_d_x
displacementOrToTr['croprange_zyx'] = croprange
displacementOrToTr['croprangeMat_yxz'] = croprangeMat

# save mat file displacement
if True:
    storemat = os.path.join(dirsaveinsilico, ptcode, 'displacementOrToTr_'+fileOr+'.mat')
    scipy.io.savemat(storemat,displacementOrToTr)
    print('displacementOrToTr was stored to {}'.format(storemat))

# visualize
if visualize:
    vv.figure(4); vv.clf()
    a1 = vv.gca()
    t1 = vv.volshow(volOr)
    vv.title('original im 1 (Or)')
    vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
    a1.daspect = 1, 1, -1
    
    vv.figure(5); vv.clf()
    a2 = vv.gca()
    t2 = vv.volshow(volTrBack)
    vv.title('im2 (tr) deformed to im1 (or)')
    vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
    
    vv.figure(6); vv.clf()
    a3 = vv.gca()
    t3 = vv.volshow(volOr-volTrBack)
    vv.title('difference im1 (or) and im1 by registration')
    vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
    
    vv.figure(7); vv.clf()
    a4 = vv.gca()
    t4 = vv.volshow(volTr)
    vv.title('original im2 (Tr)')
    vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
    
    a1.camera = a2.camera = a3.camera = a4.camera
    t1.clim = t2.clim = t3.clim = t4.clim = clim


