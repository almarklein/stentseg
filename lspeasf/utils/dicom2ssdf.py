""" Read dicom and store ssdf
Author: Maaike Koenrades. Created 2019
"""

import os
import sys

import imageio

from stentseg.utils.datahandling import select_dir, loadvol
from stentseg.utils.datahandling import savecropvols, saveaveraged, cropaveraged

def dicom2ssdf(dicom_basedir,ptcode,ctcode,basedir, cropnames=['stent'], 
              savedistolicavg=False, visvol=True, visdynamic=False):
    """ read dicom volumes and store as ssdf format
    """
    #Step A
    vols2 = [vol2 for vol2 in imageio.get_reader(dicom_basedir, 'DICOM', 'V')]
    try:
        for i, vol in enumerate(vols2):
            print(vol.meta.ImagePositionPatient)
        for i, vol in enumerate(vols2):
            print(vol.shape)
        for i, vol in enumerate(vols2):
            print(vol.meta.AcquisitionTime)
            print(vol.meta.sampling)
            assert vol.shape == vols2[0].shape
            assert vol.meta.SeriesTime == vols2[0].meta.SeriesTime
    except AttributeError:
        print('Some meta information is not available')
        pass
         
    # check order of phases
    vols = vols2.copy()
    try:
        for i,vol in enumerate(vols):
            print(vol.meta.SeriesDescription)
            assert str(i*10) in vol.meta.SeriesDescription # 0% , 10% etc.
    except AttributeError: # meta info is missing
        print('vol.meta.SeriesDescription meta information is not available')
        pass
    except AssertionError: # not correct order, fix
        vols = [None] * len(vols2)
        for i, vol in enumerate(vols2):
            print(vol.meta.SeriesDescription)
            phase = int(vol.meta.SeriesDescription[:1]) 
            # use phase to fix order of phases
            vols[phase] = vol
    
        
    # Step B: Crop and Save SSDF
    # Load and show first volume: crop with a margin of at least ~25 mm
    print()
    print('Crop with margins ~25 mm around ROI for the registration algorithm')
    stenttype = None # deprecate, not needed
    for cropname in cropnames:
        savecropvols(vols, basedir, ptcode, ctcode, cropname, stenttype)
        # Step C: average diastolic phases
        if savedistolicavg:
            phases = 50,10 # use 7 phases from 50% to 10%
            saveaveraged(basedir, ptcode, ctcode, cropname, phases)
    
    # Visualize 1 phase
    import visvis as vv
    if visvol:
        vol1 = vols[1]
        colormap = {'r': [(0.0, 0.0), (0.17727272, 1.0)],
            'g': [(0.0, 0.0), (0.27272728, 1.0)],
            'b': [(0.0, 0.0), (0.34545454, 1.0)],
            'a': [(0.0, 1.0), (1.0, 1.0)]}
            
        fig = vv.figure(2); vv.clf()
        fig.position = 0, 22, 1366, 706
        a1 = vv.subplot(111)
        a1.daspect = 1, 1, -1
        renderStyle = 'mip'
        t1 = vv.volshow(vol1, clim=(0, 3000), renderStyle=renderStyle) # iso or mip
        if renderStyle == 'iso':
            t1.isoThreshold = 300
            t1.colormap = colormap   
        a1 = vv.volshow2(vol1, clim=(-500, 500),renderStyle=renderStyle)
        vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
        vv.title('One volume at 10\% procent of cardiac cycle')
    
    if visdynamic:
        from lspeas.utils.vis import showVolPhases
        showVol='mip'
        t = showVolPhases(basedir,vols2, showVol=showVol, 
                mipIsocolor=True, isoTh=310, clim=(60,3000), slider=True  )
        
    return vols