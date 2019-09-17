""" Script to do the vasculature segmentation using marching cubes to 
obtain the wall/vessel surface and store the result.
will overwrite in pt folder
Copyright (C) Maaike Koenrades
"""
class _Do_Segmentation_Vessel:
    def __init__(self,ptcode,ctcode,basedir, threshold=300, show=True, 
                normalize=False, modelname='modelvessel'):
        
        import os
        
        import numpy as np
        import visvis as vv
        from visvis import ssdf
        
        from stentseg.utils import PointSet, _utils_GUI
        from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
        from stentseg.stentdirect.stentgraph import create_mesh
        from stentseg.stentdirect import stentgraph, StentDirect, getDefaultParams
        from stentseg.stentdirect import AnacondaDirect, EndurantDirect, NellixDirect
        from stentseg.utils.visualization import show_ctvolume
        from stentseg.utils.picker import pick3d, label2worldcoordinates, label2volindices
        import scipy
        from scipy import ndimage
        import copy
        from stentseg.utils.get_isosurface import get_isosurface3d, isosurface_to_graph, show_surface_and_vol
        
        # Select dataset to register
        cropname = 'prox'
        #phase = 10
        #what = 'phase'+str(phase)
        what = 'avgreg'
        
        # Load volumes
        s = loadvol(basedir, ptcode, ctcode, cropname, what)
        
        # sampling was not properly stored after registration for all cases: reset sampling
        vol_org = copy.deepcopy(s.vol)
        s.vol.sampling = [vol_org.sampling[1], vol_org.sampling[1], vol_org.sampling[2]] # z,y,x
        s.sampling = s.vol.sampling
        vol = s.vol
        
        # Get isosurface
        verts, faces, pp_vertices, mesh = get_isosurface3d(vol, threshold=threshold, showsurface=False)
        
        if show:
            self.label = show_surface_and_vol(vol, pp_vertices, showVol='MIP')
        
        # Convert to graph
        model = isosurface_to_graph(pp_vertices) # model with nodes
        
        # Store segmentation to disk
        p = threshold
        stentType = 'isosurface'
        
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
        s2.params = p
        s2.stentType = stentType
        # Store model (not also volume)
        s2.model = model.pack()
        
        
        # Save
        filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, modelname+ what)
        ssdf.save(os.path.join(basedir, ptcode, filename), s2)
        print('saved to disk as {}.'.format(filename) )