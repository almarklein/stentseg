""" Script to obtain centerline device limbs
 
"""

import numpy as np
import visvis as vv
import os
import scipy.io

from stentseg.utils import PointSet
from stentseg.utils.centerline import find_centerline, points_from_mesh, smooth_centerline, pp_to_graph
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel, loadmodel_location
from stentseg.utils.visualization import show_ctvolume
from stentseg.stentdirect import stentgraph
from stentseg.utils.centerline import points_from_nodes_in_graph
from stentseg.utils.picker import pick3d
from visvis import ssdf

TEST = 1

# Select the ssdf basedir
basedir = select_dir(r'D:\LSPEAS\LSPEAS_ssdf',
                    r'F:\LSPEAS_ssdf_backup', r'G:\LSPEAS_ssdf_backup')
basedirstl = select_dir(r'D:\Profiles\koenradesma\SURFdrive\UTdrive\LSPEAS\Analysis\Leg angulation',
                        r'C:\Users\Maaike\SURFdrive\UTdrive\LSPEAS\Analysis\Leg angulation')
basedirstl2 = r'D:\Profiles\koenradesma\SURFdrive\UTdrive\MedDataMimics\LSPEAS_Mimics\Tests'

# Select dataset
ptcode = 'LSPEAS_008'
ctcode = '12months'
cropname = 'stent'

showAxis = False  # True or False
showVol  = 'ISO'  # MIP or ISO or 2D or None
clim0  = (0,2500)
# clim0 = -550,500
isoTh = 250
what = 'avgreg'

s = loadvol(basedir, ptcode, ctcode, cropname, what) 

if TEST == 1:
    # Load model
    targetdir = os.path.join(basedirstl, ptcode)
    s2 = loadmodel_location(targetdir, ptcode, ctcode, cropname)
    pp = points_from_nodes_in_graph(s2.model)
    
    start1 = (170.8, 158.0, 177.4) # x,y,z - distal point
    end1 = (154.9, 125.7, 86.6)
    
    
elif TEST == 2:
    fname = os.path.join(basedirstl2, ptcode, 'LSPEAS_004_D_stent-l-th500.stl')
    # Get pointset from STL, remove duplicates
    pp = points_from_mesh(fname, invertZ = True)
    
    start1 = (146.1, 105.3, 69.3) # x,y,z
    end1 = (112.98, 100.08, 62.03)

# Get centerline
centerline1 = find_centerline(pp, start1, end1, 1, ndist=20, regfactor=0.2, regsteps=10, verbose=False)
centerline2 = smooth_centerline(centerline1, 20)

centerline_nodes = pp_to_graph(centerline2)
 
f = vv.figure(1); vv.clf()
f.position = 709.00, 30.00,  1203.00, 1008.00
a1 = vv.subplot(121)
vv.plot(pp, ms='.', ls='', alpha=0.2, mw = 7) # stent seed points
vv.plot(PointSet(list(start1)), ms='.', ls='', mc='g', mw=18) # start1
vv.plot(PointSet(list(end1)), ms='.', ls='', mc='m', mw=16)
vv.plot(centerline1, ms='.', ls='', mw=8, mc='c')
vv.plot(centerline2, ms='.', ls='', mw=8, mc='y')
a1.axis.visible = showAxis
a1.daspect = 1,1,-1

a2 = vv.subplot(122)
show_ctvolume(s.vol, None, showVol=showVol, clim=clim0, isoTh=isoTh, removeStent=False)
label = pick3d(vv.gca(), s.vol)
vv.plot(PointSet(list(start1)), ms='.', ls='', mc='g', mw=18) # start1
vv.plot(PointSet(list(end1)), ms='.', ls='', mc='m', mw=16)
centerline_nodes.Draw(mc='y', mw=8, lc='y', lw=2)
a2.axis.visible = showAxis
a2.daspect = 1,1,-1

a1.camera = a2.camera

## save ssdf and .mat file centerline
if True:
    storemat = os.path.join(targetdir, ptcode+'_'+ctcode+'_'+cropname+'_'+'centerline.mat')
    storevar = dict()
    storevar['centerline'] = centerline2
    storevar['startpoint'] = start1
    storevar['endpoint'] = end1
    scipy.io.savemat(storemat,storevar)
    print('')
    print('centerline2 was stored as.mat to {}'.format(storemat))
    print('')
    # save ssdf
    model = centerline_nodes
    # Build struct
    s2 = vv.ssdf.new()
    s2.sampling = s.sampling
    s2.origin = s.origin
    s2.stenttype = s.stenttype
    s2.croprange = s.croprange
    for key in dir(s):
            if key.startswith('meta'):
                suffix = key[4:]
                s2['meta'+suffix] = s['meta'+suffix]
    s2.what = what
    # Store model
    s2.startpoint = start1
    s2.endpoint = end1
    s2.centerline = centerline2
    s2.model = model.pack()
    # get filename
    filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'centerline')
    ssdf.save(os.path.join(targetdir, filename), s2)
    print("Centerline saved to ssdf in :")
    print(os.path.join(targetdir, filename))
    
    