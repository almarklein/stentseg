import numpy as np
import visvis as vv

from stentseg.utils import PointSet
from stentseg.utils.centerline import find_centerline, points_from_mesh, smooth_centerline


TEST = 12

if TEST == 1:
    import imageio
    im = imageio.imread('~/Desktop/test_centerline.png')[:,:,0]
    
    y, x = np.where(im < 200)
    pp = PointSet(np.column_stack([x, y]))

    start = (260, 60)
    ends = [(230, 510), (260, 510), (360, 510)]
    centerline = find_centerline(pp, start, ends, 8, ndist=20, regfactor=0.2, regsteps=10)
    
    vv.figure(1); vv.clf()
    a1 = vv.subplot(111)
    vv.plot(pp, ms='.', ls='')
    vv.plot(start[0], start[1],  ms='.', ls='', mc='g', mw=15)
    vv.plot([e[0] for e in ends], [e[1] for e in ends],  ms='.', ls='', mc='r', mw=15)
    
    vv.plot(centerline, ms='x', ls='', mw=15, mc='y')
    a1.daspectAuto = False

elif TEST > 10:
    
    if TEST == 11:
        fname = r'D:\data\maaike centerline\vessel mesh\LSPEAS_002\12months\LSPEAS_002_MGK Smoothed_Wrapped_DRGseEditRG 2_001.stl'
        start1 = (110, 100, -70)
        start2 = (87, 106, -40)
        ends = [(110, 120, -15)]
    elif TEST == 12:
        fname = r'D:\data\maaike centerline\vessel mesh\LSPEAS_003\12months\LSPEAS_003_MGK Smoothed_Wrapped_DRGseEditRG 2_001.stl'
        start1 = (190, 165, -60)
        start2 = (207, 184, -34)
        ends = [(177, 165, -17)]
    else:
        raise RuntimeError('Invalid test')
    
    # Get pointset from STL, remove duplicates
    pp = points_from_mesh(fname)
    
    #centerline1 = centerline2 = PointSet(3)
    # Find main centerline
    centerline1 = find_centerline(pp, start1, ends, 2,
                                  ndist=100, regfactor=0.2, regsteps=10, verbose=True)
    
    # Find centerline of branch, using (last part of) main centerline as an end.
    centerline2 = find_centerline(pp, start2, centerline1[-12:], 1,
                                  ndist=20, regfactor=0.1, regsteps=10, verbose=True)
    #centerline2 = smooth_centerline(centerline2, 3)
    
    vv.figure(1); vv.clf()
    a1 = vv.subplot(111)
    vv.plot(pp, ms='.', ls='', alpha=0.1)
    vv.plot(PointSet(list(start1)), ms='.', ls='', mc='g', mw=15)
    vv.plot(PointSet(list(start2)), ms='.', ls='', mw=16, mc='y')
    vv.plot([e[0] for e in ends], [e[1] for e in ends],  [e[2] for e in ends],  ms='.', ls='', mc='r', mw=15)
    
    vv.plot(centerline1, ms='.', ls='', mw=12, mc='y')
    vv.plot(centerline2, ms='.', ls='', mw=8, mc='y')
    a1.daspectAuto = False