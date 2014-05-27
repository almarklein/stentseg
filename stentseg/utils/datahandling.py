"""
Module to import ECG-gated DICOM data and save as ssdf.
Input: patient code, CTcode, basedir load DICOM, basedir_s save ssdf, stenttype, cropname

"""

import imageio
import visvis as vv
from visvis import ssdf
from visvis.utils import cropper
import os
import numpy as np


def select_dir(*dirs):
    """ Given a series of directories, select the first one that exists.
    This helps to write code that works for multiple users.
    """
    for dir in dirs:
        if os.path.isdir(dir):
            return dir
    else:
        raise RuntimeError('None of the given dirs exists.')


def loadvol(basedir, ptcode, ctcode, cropname, what='phases'):
    """ Load volume data. An ssdf struct is returned. The volumes
    are made into Aarray's with their sampling and origin set.
    """
    fname = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, what)
    
    s = ssdf.load(os.path.join(basedir, ptcode, fname))
    for key in dir(s):
        if key.startswith('vol'):
            suffix = key[3:]
            vol = vv.Aarray(s[key], s.sampling, s.origin)
            s.meta = s['meta'+suffix]
            s[key] = vol
    return s


def loadmodel(basedir, ptcode, ctcode, cropname):
    """ Load stent model. An ssdf struct is returned.
    """
    fname = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'model')
    return ssdf.load(os.path.join(basedir, ptcode, fname))


def savecropvols(vols, basedir, ptcode, CTcode, stenttype, cropname):
    """ Step B: Crop and Save SSDF
    Input: vols from Step A
    Save 10 volumes (cardiac cycle phases) in one ssdf file
    Cropper tool opens to manually set the appropriate cropping range in x,y,z
    Click 'finish' to continue saving
    Meta information (dicom), origin, stenttype and cropping range are saved
    """

    vol0 = vv.Aarray(vols[0], vols[0].meta.sampling)  # vv.Aarray defines origin: 0,0,0
    
    # Open cropper tool
    print('Set the appropriate cropping range for cropname "%s" '
            'and click finish to continue' % cropname)
    fig = vv.figure()
    fig.title = 'Cropping for cropname "%s"' % cropname
    vol_crop = cropper.crop3d(vol0, fig)
    fig.Destroy()
    
    if vol_crop.shape == vol0.shape:
        raise RuntimeError('User cancelled (no crop)')
    
    # Calculate crop range from origin
    rz = int(vol_crop.origin[0] / vol_crop.sampling[0] + 0.5)
    rz = rz, rz + vol_crop.shape[0]
    ry = int(vol_crop.origin[1] / vol_crop.sampling[1] + 0.5)
    ry = ry, ry + vol_crop.shape[1]
    rx = int(vol_crop.origin[2] / vol_crop.sampling[2] + 0.5)
    rx = rx, rx + vol_crop.shape[2]
    
    # Initialize struct
    s = ssdf.new()
    s.sampling = vol_crop.sampling  # z, y, x voxel size in mm
    s.origin = vol_crop.origin
    s.croprange = rz, ry, rx
    s.stenttype = stenttype
    
    # Export and save
    for volnr in range(0,len(vols)):
        phase = volnr*10
        s['vol%i'% phase]  = vols[volnr][rz[0]:rz[1], ry[0]:ry[1], rx[0]:rx[1]]
        s['meta%i'% phase] = vols[volnr].meta
        vols[volnr].meta.PixelData = None  # avoid ssdf warning
        
    
    filename = '%s_%s_%s_phases.ssdf' % (ptcode, CTcode, cropname)
    file_out = os.path.join(basedir,ptcode, filename )
    ssdf.save(file_out, s)


def saveaveraged(basedir, ptcode, CTcode, cropname, phases):
    """ Step C: Save average of a number of volumes (phases in cardiac cycle)
    Load ssdf containing all phases and save averaged volume as new ssdf
    """

    filename = '%s_%s_%s_phases.ssdf' % (ptcode, CTcode, cropname)
    file_in = os.path.join(basedir,ptcode, filename)
    if not os.path.exists(file_in):
        raise RuntimeError('Could not find ssdf for given input %s' % ptcode, CTcode, cropname)
    s = ssdf.load(file_in)
    s_avg = ssdf.new()
    averaged = np.zeros(s.vol0.shape, np.float64)
    phaserange = range(phases[0],phases[1]+10,10)
    for phase in phaserange:
        averaged += s['vol%i'% phase]
        s_avg['meta%i'% phase] = s['meta%i'% phase]
    averaged *= 1.0 / len(phaserange)
    averaged = averaged.astype('float32')
    
    s_avg.vol = averaged
    s_avg.sampling = s.sampling  # z, y, x voxel size in mm
    s_avg.origin = s.origin
    s_avg.stenttype = s.stenttype
    s_avg.croprange = s.croprange
    s_avg.meta = s.meta0  # sort of default meta, for loadvol

    avg = 'avg'+ str(phases[0])+str(phases[1])
    filename = '%s_%s_%s_%s.ssdf' % (ptcode, CTcode, cropname, avg)
    file_out = os.path.join(basedir,ptcode, filename)
    ssdf.save(file_out, s_avg)


def cropaveraged(cropnames):
    """ Crop averaged volume of stent manually
    Load ssdf containing stent_avg and saves new ssdf with overwritten croprange
    """
    
    filename = ptcode+'_'+CTcode+'_'+'stent'+'_'+avg+'.ssdf'
    file_in = os.path.join(basedir_s,ptcode, filename)
    if not os.path.exists(file_in):
            raise RuntimeError('Could not find ssdf stent_avg for given input %s' % ptcode, CTcode)
    for cropname in cropnames:
        s = ssdf.load(file_in)
        
        vol0 = vv.Aarray(s.vol, s.sampling, s.origin)
        
        # Open cropper tool
        vol_crop,rz,ry,rx = cropper.crop3d(vol0) # Output of crop3d in cropper.py modified: return vol2, rz,ry,rx
        offset = [i[0] for i in s.croprange]
        s.croprange = ([rz.min+offset[0], rz.max+offset[0]], 
                       [ry.min+offset[1], ry.max+offset[1]], 
                       [rx.min+offset[2], rx.max+offset[2]])
        
        # Export and save
        avg = 'avg'+ str(phases[0])+str(phases[1])
        filename = ptcode+'_'+CTcode+'_'+cropname+'_'+avg+'.ssdf'
        file_out = os.path.join(basedir_s,ptcode, filename)
        ssdf.save(file_out, s)

