"""
Module to handle ECG-gated DICOM data. Saves volumes in ssdf.
Inputs: dicom_basedir, patient code, ctcode, basedir for ssdf, stenttype, 
        cropname, phases

"""

import imageio
import visvis as vv
from visvis import ssdf
from visvis.utils import cropper 
from stentseg.apps.crop import crop3d # use local version while error with range floats
import os
import sys
import numpy as np
import scipy
from scipy import ndimage

#todo: verify resample_vol function 
def resample_vol(vol, xsampling=0.5, ysampling=0.5, zsampling=0.5):
    """ input: vol with vol.sampling
        output: vol with new sampling
    """
    currentSampling = vol.sampling # vol in z,y,x
    zscale = vol.sampling[0] / zsampling # z / znew
    yscale = vol.sampling[1] / ysampling # y / ynew
    xscale = vol.sampling[2] / xsampling # x / xnew
    vol_sampled = scipy.ndimage.interpolation.zoom(vol,[zscale,yscale,xscale],'float32')
    newSampling = (zsampling,ysampling,xsampling)
    # adjust vol with sampling and origin
    vol_sampled_type = vv.Aarray(vol_sampled, newSampling, vol.origin)
    vol = vol_sampled_type


def normalize_soft_limit(vol, limit):
    """ from pirt _soft_limit1()
    return normalized vol
    """
    if limit == 1:
        data = 1.0 - np.exp(-vol)
    else:
        f = np.exp(-vol/limit)
        data = -limit * (f-1) #todo: when vol[:] = .. ValueError: assignment destination is read-only
    return data 
    
    
def select_dir(*dirs):
    """ Given a series of directories, select the first one that exists.
    This helps to write code that works for multiple users.
    """
    for dir in dirs:
        if os.path.isdir(dir):
            return dir
    else:
        raise RuntimeError('None of the given dirs exists.')


def renamedcm(dicom_basedir, ptcode, ctcode):
    """ Renames 4D dcm data to shorter name using ptcode, ctcode and phase
    Renames the filenames in all 10 volumes, keeps the (uid) folder names.
    Also checks order of phases by reading the 90% volume.
    """

    dirname = os.path.join(dicom_basedir, ptcode, ptcode+'_'+ctcode)
    
    if not os.path.isdir(dirname):
        raise RuntimeError('Could not find data for given input %s' % ptcode, ctcode)
    
    while True:
        subfolder = os.listdir(dirname)
        if len(subfolder) == 1:  # data should only contain one main uid folder
            dirname = os.path.join(dirname, subfolder[0])
        else:
            break
    
    if not subfolder:
        raise RuntimeError('Could not find any files for given input %s' % ptcode, ctcode)
    
    # Check order of phases
    vol = imageio.volread(os.path.join(dirname,subfolder[9]), 'dicom')
    assert '90%' in vol.meta.SeriesDescription
    
    # Rename files in all 10 subfolders
    for volnr in range(0,10): # 0,10 for all phases from 0 up to 90%
        dirsubfolder = os.path.join(dirname,subfolder[volnr])
        perc = '%i%%' % (volnr*10)
        i = 1  # initialize file numbering
        for filename in os.listdir(dirsubfolder):
            if 'dcm' in filename:
                newFilename = '%s_%s_%s_%i.dcm' % (ptcode, ctcode, perc, i)
                os.rename(os.path.join(dirsubfolder,filename), os.path.join(dirsubfolder, newFilename))
                i +=1


def loadmesh(basedirMesh, ptcode=None, meshname=None, invertZ=True, fname=None):
    """ Load Mesh object, flip z and return Mesh
    meshname includes ctcode
    """
    if fname is None:
        try:
            mesh = vv.meshRead(os.path.join(basedirMesh, ptcode, meshname))
        except FileNotFoundError:
            mesh = vv.meshRead(os.path.join(basedirMesh, meshname))
    else:
        try:
            mesh = vv.meshRead(os.path.join(basedirMesh, ptcode, fname))
        except FileNotFoundError:
            mesh = vv.meshRead(os.path.join(basedirMesh, fname))
    if invertZ == True:
        # z is negative, must be flipped to match dicom orientation CT data
        mesh._vertices[:, -1] *= -1
    return mesh


def loadvol(basedir, ptcode=None, ctcode=None, cropname=None, what='phases', fname=None):
    """ Load volume data. An ssdf struct is returned. The volumes
    are made into Aarray's with their sampling and origin set.
    """
    if fname is None:
        fname = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, what)
    try:
        s = ssdf.load(os.path.join(basedir, ptcode, fname))
    except FileNotFoundError:
        s = ssdf.load(os.path.join(basedir, fname))
    for key in dir(s):
        if key.startswith('vol'):
            s[key] = vv.Aarray(s[key], s.sampling, s.origin)
        elif key.startswith('deform'):
            fields = s[key]
            s[key] = [vv.Aarray(field, s.sampling, s.origin) for field in fields] # origin of volume is added as meta-data
            #[vv.Aarray(field, s.sampling, [0,0,0]) for field in fields] would use wrong deform-data when deforming
    return s


def loadmodel(basedir, ptcode, ctcode, cropname, modelname='modelavgreg'):
    """ Load stent model. An ssdf struct is returned.
    """
    # Load
    fname = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, modelname)
    try:
        s = ssdf.load(os.path.join(basedir, ptcode, fname))
    except FileNotFoundError:
        s = ssdf.load(os.path.join(basedir, fname))
    # Turn into graph model
    from stentseg.stentdirect import stentgraph
    for key in dir(s):
        if key.startswith('model'):
            model = stentgraph.StentGraph()
            model.unpack(s[key])
            s[key] = model
        # also unpack seeds if exist 
        #-mind that also seeds need to be packed before save is possible again-
        elif key.startswith('seeds'):
            seeds = stentgraph.StentGraph()
            seeds.unpack(s[key])
            s[key] = seeds
        elif key.startswith('landmark'): # landmark validation
            landmarks = stentgraph.StentGraph()
            landmarks.unpack(s[key])
            s[key] = landmarks
    return s

# def loadmodel_location(basedir, ptcode, ctcode, cropname, modelname='stentseedsavgreg'):
#     """ Load stent model. An ssdf struct is returned.
#     """
#     # Load
#     fname = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, modelname)
#     s = ssdf.load(os.path.join(basedir, fname))
#     # Turn into graph model
#     from stentseg.stentdirect import stentgraph
#     for key in dir(s):
#         if key.startswith('model'):
#             model = stentgraph.StentGraph()
#             model.unpack(s[key])
#             s[key] = model
#     return s


def savecropvols(vols, basedir, ptcode, ctcode, cropname, stenttype, sampling=None, meta=None):
    """ Step B: Crop and Save SSDF
    Input: vols from Step A
    Save 2 or 10 volumes (cardiac cycle phases) in one ssdf file
    Cropper tool opens to manually set the appropriate cropping range in x,y,z
    Click 'finish' to continue saving
    Meta information (dicom), origin, stenttype and cropping range are saved
    """

    try:
        vol0 = vv.Aarray(vols[0], vols[0].meta.sampling)  # vv.Aarray defines origin: 0,0,0
    except AttributeError:
        print('AttributeError no meta with vol, use given sampling')
        vol0 = vv.Aarray(vols[0], sampling)
    
    # Open cropper tool
    print('Set the appropriate cropping range for "%s" '
            'and click finish to continue' % cropname)
    print('Loading... be patient')
    fig = vv.figure()
    fig.title = 'Cropping for "%s"' % cropname
    #vol_crop = cropper.crop3d(vol0, fig)
    vol_crop = crop3d(vol0, fig) # use local while error with cropper
    fig.Destroy()
    
    if vol_crop.shape == vol0.shape:
        print('User did not crop')
    
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
    s.ctcode = ctcode
    s.ptcode = ptcode
    
    # Export and save
    if len(vols) == 1: # when static ct
        s.vol = vols[0][rz[0]:rz[1], ry[0]:ry[1], rx[0]:rx[1]]
        s.meta = vols[0].meta
        vols[0].meta.PixelData = None  # avoid ssdf warning
        filename = '%s_%s_%s_phase.ssdf' % (ptcode, ctcode, cropname)
    else: # dynamic ct
        for volnr in range(0,len(vols)):
            phase = volnr*10
            if len(vols) == 2: # when only diastolic/systolic volume
                try:
                    phase = vols[volnr].meta.SeriesDescription[:2] # '78%, iDose'
                    phase = int(phase)
                except AttributeError: # has no meta
                    phase = volnr
            s['vol%i'% phase]  = vols[volnr][rz[0]:rz[1], ry[0]:ry[1], rx[0]:rx[1]]
            try:
                s['meta%i'% phase] = vols[volnr].meta
                vols[volnr].meta.PixelData = None  # avoid ssdf warning
            except AttributeError: # has no meta
                pass
                # s['meta%i'% phase] = meta # given input var
                # todo: meta type now error TypeError: unorderable types: DataElement() < DataElement()
        filename = '%s_%s_%s_phases.ssdf' % (ptcode, ctcode, cropname)
    file_out = os.path.join(basedir,ptcode, filename )
    ssdf.save(file_out, s)
    print()
    print('ssdf saved to {}.'.format(file_out) )


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
    if phases[1]<phases[0]:
        phaserange = list(range(0,100,10))
        for i in range(phases[1]+10,phases[0],10):
            phaserange.remove(i)
    else:
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
    s_avg.ctcode = s.ctcode
    s_avg.ptcode = s.ptcode

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

