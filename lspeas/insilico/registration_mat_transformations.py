""" Script to register mat file image volumes that were obtained from 
get_and_deform_CTdata_loop.m

2018-2019 Maaike Koenrades
"""
import scipy.io 
import os
import os, time

import numpy as np
import visvis as vv
import pirt.reg # Python Image Registration Toolkit
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel

# Select the ssdf basedir for original data
# basedir = select_dir(r'D:\LSPEAS\LSPEAS_ssdf',
#                      r'F:\LSPEAS_ssdf_backup')

# Set original dataset for sampling
ptcode = 'LSPEAS_001'
ctcode = '1month'
cropname = 'ring' # just for fast reading vol for sampling
what = 'phases' 

# Load volumes
oriPhase = '50'
#s = loadvol(basedir, ptcode, ctcode, cropname, what)
sampling = (0.5, 0.383, 0.383) #s['vol'+oriPhase].sampling # z,y,x
origin = [0, 34.087, 71.238] #s.origin
# sampling2 = (1,1,1) # ground truth sampling from Matlab (width_pixel=1)
#todo: in matlab set pixel_width DataDressed/DataDressedInterpolated with dicom sampling for realistic registration?!
origin2 = (0,0,0)

# ======================================================
savemat = True
visualize = True
reg2d = False

# Load the mat files with original and deformed volumes
dirinsilico = select_dir(
        #r'D:\LSPEAS\LSPEAS_insilico_mat',
        #r'F:\LSPEAS_insilico_mat',
        (r'D:\Profiles\koenradesma\SURFdrive\UTdrive\LSPEAS\In silico validation'
            r'\translation_Geurts_2019-02-08\translation\LSPEAS_insilico_mat'),
        (r'C:\Users\Maaike\SURFdrive\UTdrive\LSPEAS\In silico validation'
            r'\translation_Geurts_2019-02-08\translation\LSPEAS_insilico_mat'),
        #r'/Users/geurtsbj/ownCloud/research/articles/maaike/registration_sensitivity/registration_assessment/LSPEAS_insilico_mat',
        r'/Users/geurtsbj/ownCloud/research/articles/maaike/registration_sensitivity_cases/translation/LSPEAS_insilico_mat')

dirsaveinsilico = select_dir(
        #r'D:\LSPEAS\LSPEAS_insilico_ssdf\reg1',
        #r'F:\LSPEAS_insilico_ssdf\reg1',
        (r'D:\Profiles\koenradesma\SURFdrive\UTdrive\LSPEAS\In silico validation'
            r'\translation_Geurts_2019-02-08\translation\LSPEAS_insilico_ssdf\reg1'),
        (r'C:\Users\Maaike\SURFdrive\UTdrive\LSPEAS\In silico validation'
            r'\translation_Geurts_2019-02-08\translation\LSPEAS_insilico_ssdf\reg1'),
        #r'/Users/geurtsbj/ownCloud/research/articles/maaike/registration_sensitivity/registration_assessment/LSPEAS_insilico_ssdf/reg1',
        r'/Users/geurtsbj/ownCloud/research/articles/maaike/registration_sensitivity_cases/translation/LSPEAS_insilico_ssdf/reg1')

# Get .mat volumes DataDressed and all DataDressedInterpolated automatically from folder dirinsilico\ptcode
fileTrs = []
files_in_folder = os.listdir(os.path.join(dirinsilico, ptcode))
for file_name in files_in_folder:
    if file_name.startswith('original'):
        fileOr = file_name
    elif file_name.startswith('tr_DD'): # collect all DataDressedInterpolated
        fileTrs.append(file_name)

# Or set manual filenames for volumes to register
# fileOr = 'original_DataDressed_001_d50_1-146'
# 
# fileTrs =  [ 'tr_DD_001_d50_1-146_1_0.01_t0.25pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_0.02_t0.25pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_0.05_t0.25pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_0.10_t0.25pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_0.20_t0.25pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_0.50_t0.25pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_1.00_t0.25pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_2.00_t0.25pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_5.00_t0.25pi_p0.5pi', 
#              'tr_DD_001_d50_1-146_1_10.00_t0.25pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_20.00_t0.25pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_50.00_t0.25pi_p0.5pi'                  
#              ]
             
# fileTrs =  [ 'tr_DD_001_d50_1-146_1_0.01_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_0.02_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_0.05_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_0.10_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_0.20_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_0.50_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_1.00_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_2.00_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_5.00_t0pi_p0.5pi', 
#              'tr_DD_001_d50_1-146_1_10.00_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_20.00_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_50.00_t0pi_p0.5pi'                  
#              ]
#              
# fileTrs =  [ 'tr_DD_001_d50_1-146_1_-0.01_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_-0.02_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_-0.05_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_-0.10_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_-0.20_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_-0.50_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_-1.00_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_-2.00_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_-5.00_t0pi_p0.5pi', 
#              'tr_DD_001_d50_1-146_1_-10.00_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_-20.00_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_-50.00_t0pi_p0.5pi', 
#              'tr_DD_001_d50_1-146_1_0.01_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_0.02_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_0.05_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_0.10_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_0.20_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_0.50_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_1.00_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_2.00_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_5.00_t0pi_p0.5pi', 
#              'tr_DD_001_d50_1-146_1_10.00_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_20.00_t0pi_p0.5pi',
#              'tr_DD_001_d50_1-146_1_50.00_t0pi_p0.5pi'                  
#              ]
# ======================================================

# load DataDressed and DataDressedInterpolated for all VelocityFieldTransformations
for fileTr in fileTrs:
    mat = scipy.io.loadmat(os.path.join(dirinsilico, ptcode, fileOr))
    mat2 = scipy.io.loadmat(os.path.join(dirinsilico, ptcode, fileTr))
    
    volOr = mat['DataDressed']
    volTr = mat2['DataDressedInterpolated']
    volOr = np.swapaxes(volOr, 0,2) # from y,x,z to z,x,y
    volTr = np.swapaxes(volTr, 0,2)
    volOr = np.swapaxes(volOr, 1,2) # from z,x,y to z,y,x
    volTr = np.swapaxes(volTr, 1,2)
    
    # to use only part of the volume enable crop and set range
    if False: # crop xy
        # ======================
        ymin, ymax = 200, -200
        xmin, xmax = 300, -150
        # ======================
        croprange = [[0,volOr.shape[0]], [ymin,ymax], [xmin,xmax]] # zyx
        croprangeMat = [[ymin+1,volOr.shape[1]+ymax],[xmin+1,volOr.shape[2]+xmax], [1,volOr.shape[0]]] # yxz (x and y different direction)
        volOr = volOr[:, ymin:ymax, xmin:xmax]
        volTr = volTr[:, ymin:ymax, xmin:xmax]
    elif False: # crop z
        # ======================
        croprange[0] = [1,-2]
        croprangeMat[2] = [2, volOr.shape[0]-2]
        # ======================
        volOr = volOr[1:-2]
        volTr = volTr[1:-2]
    else:
        croprange = False
        croprangeMat = False
    # 'Dressed' part gives errors due to extreme negative intensities (-2000) -->
    # clip intensities below -1000 for registration, --> set to -200 (HU between fat-lung)
    if True: 
        volOr[volOr<-1000] = -200
        volTr[volTr<-1000] = -200
    
    # Get the sampling of DataDressed and DataDressedInterpolated
    sampling2_x = mat['DressedVars']['width_pixel_y'][0][0][0][0] # y in matlab = x in python (left-right)
    sampling2_y = mat['DressedVars']['width_pixel_x'][0][0][0][0] # x in matlab  = y in python (ant-post)
    sampling2_z = mat['DressedVars']['width_pixel_z'][0][0][0][0]
    sampling2 = tuple([sampling2_z,sampling2_y,sampling2_x]) # zyx
    
    # create Aarray volume    
    volOr = vv.Aarray(volOr, sampling2, origin2)
    volTr = vv.Aarray(volTr, sampling2, origin2)
    vols = [volOr, volTr]
    
    if reg2d: # Get a slice for registration not the volume
        imOr = volOr[int(0.5*volOr.shape[0]),:,:] # pick a z slice in mid volume
        imTr = volTr[int(0.5*volOr.shape[0]),:,:]
        sampling2 = sampling2[1:] # keep y,x
        origin2 = origin2[1:]
        vols = [imOr, imTr]
    
    if visualize:
        f = vv.figure(1); vv.clf()
        f.position = 0.00, 22.00,  1914.00, 1018.00
        clim = (-400,1500)
        a1 = vv.subplot(221)
        vv.volshow(volOr,clim=clim)
        vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
        vv.title('Im1:Original (DataDressed)')
        a1.daspect = 1,1,-1
        
        a2 = vv.subplot(222)
        vv.volshow(volOr,clim=clim)
        vv.volshow(volTr,clim=clim)
        vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
        vv.title('Overlay')
        a2.daspect = 1,1,-1
        
        a3 = vv.subplot(223)
        vv.volshow(volTr,clim=clim)
        vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
        vv.title('Im2:Original transformed (DataDressedInterpolated u_\magnitude {})'.format(mat2['u_magnitude'][0][0]))
        a3.daspect = 1,1,-1
        
        a4 = vv.subplot(224)
        a4.daspect = 1,1,-1
        
        c = vv.ClimEditor(vv.gcf())
        
        a1.camera = a2.camera = a3.camera = a4.camera
    
    # Remove .mat for savename and loading below
    if fileTr.endswith('.mat'):
        fileTr = fileTr.replace('.mat', '')
    
    ## Do registration
    
    if True:
        t0 = time.time()
        
        # set final scale based on pixelwidth (scale should be =<0.5*pixelwidth)
        if max(sampling2) > 2:
            final_scale = max(sampling2)/2
        else:
            final_scale = 1.0
        
        # Initialize registration object
        reg = pirt.reg.GravityRegistration(*vols)
        # ======================
        # Set params
        reg.params.mass_transforms = 2  # 2;2nd order (Laplacian) triggers more at lines
        reg.params.deform_wise = 'groupwise' # groupwise!
        reg.params.mapping = 'backward'
        reg.params.deform_limit = 1.0
        reg.params.final_scale = final_scale  #1.0 We might set this a wee bit lower than 1 (but slower!)
        reg.params.grid_sampling_factor = 0.5 # 0.5 !! important especially for Laplace !!
        # most important params
        reg.params.speed_factor = 1.0 # ch9 0.5-2.0; ch10 2.0
        reg.params.scale_sampling = 16 # 16; nr of iterations per scale level (ch10 16-200)
        reg.params.final_grid_sampling = 20 #20; grid sampling ch9/10: 5-15-30/20 
                                            #This parameter is usually best coupled to final_scale
        
        # ======================
        # Go!
        reg.register(verbose=1)
        
        
        # Store registration result
        from visvis import ssdf
        
        # Create struct
        s2 = vv.ssdf.new()
        N = len(vols)
        # s2.meta = s['meta'+oriPhase] # from orig vol
        s2.origin = origin 
        # s2.stenttype = s.stenttype
        # s2.croprange = s.croprange
        
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
        # create average volume from *all* volumes deformed to the "center"
        N = len(reg._ims)
        mean_vol = np.zeros(reg._ims[0].shape, 'float64')
        for i in range(N):
            vol, deform = reg._ims[i], reg.get_deform(i) # reg._ims[0]==volOr
            mean_vol += deform.as_backward().apply_deformation(vol)
        mean_vol *= 1.0/N
        
        # Create struct
        s_avg = ssdf.new()
        #s_avg.meta = s['meta'+oriPhase]
        s_avg.sampling = sampling2 # z, y, x 
        s_avg.origin = origin2
        s_avg.samplingOriginal = sampling
        s_avg.originOriginal = origin
        #s_avg.stenttype = s.stenttype
        #s_avg.croprange = s.croprange
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
        avg = 'avgreg'
        filename = '%s_%s.ssdf' % (fileTr, avg)
        savg = loadvol(dirsaveinsilico, ptcode, ctcode, None, 'avgreg', fname=filename) 
        # loadvol sets savg.sampling and savg.origin
        vol = savg.vol
        # show
        #f = vv.figure(2); vv.clf()
        #f.position = 968.00, 30.00,  944.00, 1002.00
        #a = vv.subplot(111)
        #a.daspect = 1, 1, -1
        if reg2d:
            t = vv.imshow(vol, clim=clim, axes=a4)
        else:
            t = vv.volshow(vol, clim=clim, renderStyle='mip', axes=a4)
            #t.isoThreshold = 600
        vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
        vv.title('Averaged volume')
    
    # load deforms
    filename = '%s_%s.ssdf' % (fileTr, 'deforms')
    s3 = loadvol(dirsaveinsilico, ptcode, ctcode, None, 'deforms', fname=filename) 
    # loadvol sets s3.sampling and s3.origin
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
    fieldTrBack_d_z = deform_tr_to_or.get_field(0)
    fieldTrBack_d_y = deform_tr_to_or.get_field(1)
    fieldTrBack_d_x = deform_tr_to_or.get_field(2)
    # from im1 (ori) transform to im2 (transformed)
    deform_or_to_tr = deformfield0.compose(deformfield1.inverse()).as_backward()
    volOrBack = deform_or_to_tr.apply_deformation(volOr)
    # displacement
    fieldOrBack_d_z = deform_or_to_tr.get_field(0)
    fieldOrBack_d_y = deform_or_to_tr.get_field(1)
    fieldOrBack_d_x = deform_or_to_tr.get_field(2)
    
    # store displacement fields
    displacementOrToTr = mat2.copy()
    del displacementOrToTr['DataDressedInterpolated'] # remove Image data
    displacementOrToTr['displacementOrToTr_z'] = fieldOrBack_d_z
    displacementOrToTr['displacementOrToTr_x'] = fieldOrBack_d_y # x and y direction different in matlab
    displacementOrToTr['displacementOrToTr_y'] = fieldOrBack_d_x # x and y direction different in matlab
    displacementOrToTr['croprange_zyx'] = croprange
    displacementOrToTr['croprangeMat_yxz'] = croprangeMat
    displacementOrToTr['samplingOriginal'] = sampling
    displacementOrToTr['originOriginal'] = origin
    displacementOrToTr['sampling'] = sampling2
    displacementOrToTr['origin'] = origin2
    
    
    # save mat file displacement
    if savemat:
        storemat = os.path.join(dirsaveinsilico, ptcode, 'dispOrToTr_'+fileTr)
        scipy.io.savemat(storemat,displacementOrToTr)
        print('displacementOrToTr was stored to {}'.format(storemat))
    
    # visualize
    if visualize:
        vv.figure(2); vv.clf()
        a1b = vv.subplot(111)
        t1 = vv.volshow(volOr)
        vv.title('original im 1 (DataDressed)')
        vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
        a1b.daspect = 1, 1, -1
        
        vv.figure(3); vv.clf()
        a2b = vv.subplot(111)
        t2 = vv.volshow(volTrBack) # im1 as was calculated by the registration
        vv.title('im2 (DataDressedInterpolated) transformed by registration to im1 (DataDressed)')
        vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
        a2b.daspect = 1, 1, -1
        
        vv.figure(4); vv.clf()
        a3b = vv.subplot(111)
        t3 = vv.volshow(volTr)
        vv.title('original im2 (DataDressedInterpolated)')
        vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
        a3b.daspect = 1, 1, -1
        
        vv.figure(5); vv.clf()
        a4b = vv.subplot(111)
        t4 = vv.volshow(volOr-volTrBack)
        vv.title('difference im1 (DataDressed) and im1 by registration')
        vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
        a4b.daspect = 1, 1, -1
        
        a1b.camera = a2b.camera = a3b.camera = a4b.camera
        t1.clim = t2.clim = t3.clim = t4.clim = clim
    

