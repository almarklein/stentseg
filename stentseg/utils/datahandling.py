"""
Module to handle ECG-gated DICOM data. Saves volumes in ssdf.
Inputs: dicom_basedir, patient code, ctcode, basedir for ssdf, stenttype, 
        cropname, phases

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
            s[key] = vv.Aarray(s[key], s.sampling, s.origin)
        elif key.startswith('deform'):
            fields = s[key]
            s[key] = [vv.Aarray(field, s.sampling, s.origin) for field in fields]
    return s


def loadmodel(basedir, ptcode, ctcode, cropname, modelname='modelavgreg'):
    """ Load stent model. An ssdf struct is returned.
    """
    # Load
    fname = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, modelname)
    s = ssdf.load(os.path.join(basedir, ptcode, fname))
    # Turn into graph model
    from stentseg.stentdirect import stentgraph
    model = stentgraph.StentGraph()
    model.unpack(s.model)
    s.model = model
    return s


def savecropvols(vols, basedir, ptcode, ctcode, cropname, stenttype):
    """ Step B: Crop and Save SSDF
    Input: vols from Step A
    Save 10 volumes (cardiac cycle phases) in one ssdf file
    Cropper tool opens to manually set the appropriate cropping range in x,y,z
    Click 'finish' to continue saving
    Meta information (dicom), origin, stenttype and cropping range are saved
    """

    vol0 = vv.Aarray(vols[0], vols[0].meta.sampling)  # vv.Aarray defines origin: 0,0,0
    
    # Open cropper tool
    print('Set the appropriate cropping range for "%s" '
            'and click finish to continue' % cropname)
    print('Loading... be patient')
    fig = vv.figure()
    fig.title = 'Cropping for "%s"' % cropname
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
    s.croprange = rz, ry, rx  # in world coordinates 
    s.stenttype = stenttype
    
    # Export and save
    for volnr in range(0,len(vols)):
        phase = volnr*10
        s['vol%i'% phase]  = vols[volnr][rz[0]:rz[1], ry[0]:ry[1], rx[0]:rx[1]]
        s['meta%i'% phase] = vols[volnr].meta
        vols[volnr].meta.PixelData = None  # avoid ssdf warning
        
    filename = '%s_%s_%s_phases.ssdf' % (ptcode, ctcode, cropname)
    file_out = os.path.join(basedir,ptcode, filename )
    ssdf.save(file_out, s)


def saveaveraged(basedir, ptcode, ctcode, cropname, phases):
    """ Step C: Save average of a number of volumes (phases in cardiac cycle)
    Load ssdf containing all phases and save averaged volume as new ssdf
    """

    filename = '%s_%s_%s_phases.ssdf' % (ptcode, ctcode, cropname)
    file_in = os.path.join(basedir,ptcode, filename)
    if not os.path.exists(file_in):
        raise RuntimeError('Could not find ssdf for given input %s' % ptcode, ctcode, cropname)
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

    avg = 'avg'+ str(phases[0])+str(phases[1])
    filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, avg)
    file_out = os.path.join(basedir,ptcode, filename)
    ssdf.save(file_out, s_avg)


def cropaveraged(basedir, ptcode, ctcode, crop_in='stent', what='avg5090', crop_out = 'body'):
    """ Crop averaged volume of stent
    With loadvol, load ssdf containing an averaged volume and save new cropped ssdf
    """
    s = loadvol(basedir, ptcode, ctcode, crop_in, what)
    vol = s.vol

    # Open cropper tool
    print('Set the appropriate cropping range for "%s" '
            'and click finish to continue' % crop_out)
    print('Loading... be patient')
    fig = vv.figure()
    fig.title = 'Cropping for "%s"' % crop_out
    vol_crop = cropper.crop3d(vol, fig)
    fig.Destroy()
    
    if vol_crop.shape == vol.shape:
        raise RuntimeError('User cancelled (no crop)')
    
    # Calculate crop range from origin
    rz = int(vol_crop.origin[0] / vol_crop.sampling[0] + 0.5)
    rz = rz, rz + vol_crop.shape[0]
    ry = int(vol_crop.origin[1] / vol_crop.sampling[1] + 0.5)
    ry = ry, ry + vol_crop.shape[1]
    rx = int(vol_crop.origin[2] / vol_crop.sampling[2] + 0.5)
    rx = rx, rx + vol_crop.shape[2]
    
    # Initialize struct
    s2 = ssdf.new()
    s2.sampling = vol_crop.sampling  # z, y, x voxel size in mm 
    s2.stenttype = s.stenttype
    for key in dir(s):
        if key.startswith('meta'):
            suffix = key[4:]
            s2['meta'+suffix] = s['meta'+suffix]
    
    # Set new croprange, origin and vol
    offset = [i[0] for i in s.croprange]  # origin of crop_in
    s2.croprange = ([rz[0]+offset[0], rz[1]+offset[0]], 
                    [ry[0]+offset[1], ry[1]+offset[1]], 
                    [rx[0]+offset[2], rx[1]+offset[2]])
    s2.origin = vol_crop.origin
    s2.vol = vol_crop
    
    # Export and save
    filename = '%s_%s_%s_%s.ssdf'% (ptcode, ctcode, crop_out, what)
    file_out = os.path.join(basedir, ptcode, filename)
    ssdf.save(file_out, s2)

