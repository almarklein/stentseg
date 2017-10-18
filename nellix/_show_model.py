
class _Show_Model:
    def __init__(self,ptcode,ctcode,basedir,meshWithColors=False):
        """
        Script to show the stent model. [ nellix]
        """
        
        import os
        import pirt
        import visvis as vv
        
        from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
        from pirt.utils.deformvis import DeformableTexture3D, DeformableMesh
        from stentseg.stentdirect.stentgraph import create_mesh
        from stentseg.utils.visualization import show_ctvolume
        from stentseg.motion.vis import create_mesh_with_abs_displacement
        import copy
        from stentseg.motion.dynamic import incorporate_motion_nodes, incorporate_motion_edges 

        import numpy as np
        
        cropname = 'prox'
        # params
        nr = 40
        motion = 'amplitude'  # amplitude or sum
        dimension = 'xyz'
        showVol = 'mip'  # MIP or ISO or 2D or None
        clim0 = (0,2000)
        clim2 = (0,2)
        motionPlay = 9, 2   # each x ms, a step of x %
        
        s = loadvol(basedir, ptcode, ctcode, cropname, what = 'deforms')
        m = loadmodel(basedir, ptcode, ctcode, cropname, 'centerline_modelavgreg')
        s2 = loadvol(basedir, ptcode, ctcode, cropname, what='avgreg')
        vol_org = copy.deepcopy(s2.vol)
        s2.vol.sampling = [vol_org.sampling[1], vol_org.sampling[1], vol_org.sampling[2]]
        s2.sampling = s2.vol.sampling
        vol = s2.vol
        
        # Load deformations (forward for mesh)
        deformkeys = []
        for key in dir(s):
            if key.startswith('deform'):
                deformkeys.append(key)
        deforms = [s[key] for key in deformkeys]
        deforms = [[field[::2,::2,::2] for field in fields] for fields in deforms]
        
        # These deforms are forward mapping. Turn into DeformationFields.
        # Also get the backwards mapping variants (i.e. the inverse deforms).
        # The forward mapping deforms should be used to deform meshes (since
        # the information is used to displace vertices). The backward mapping
        # deforms should be used to deform textures (since they are used in
        # interpolating the texture data).
        deforms_f = [pirt.DeformationFieldForward(*f) for f in deforms]
        deforms_b = [f.as_backward() for f in deforms_f]
        
        # Load the stent model and mesh+
        modelmesh = vv.ssdf.new()
        for key in dir(m):
            if key.startswith('model'):
                if meshWithColors:
                    try:
                        modelmesh[key] = create_mesh_with_abs_displacement(m[key], 
                                         radius = 0.7, dim = dimension, motion = motion)
                    except KeyError:
                        print('Centerline model has no pathdeforms so we create them')
                        # use unsampled deforms
                        deforms2 = [s[key] for key in deformkeys]
                        # deforms as backward for model
                        deformsB = [pirt.DeformationFieldBackward(*fields) for fields in deforms2]
                        # set sampling to original
                        # for i in range(len(deformsB)):
                        #         deformsB[i]._field_sampling = tuple(s.sampling)
                        # not needed because we use unsampled deforms
                        # Combine ...
                        incorporate_motion_nodes(m[key], deformsB, s2.origin)
                        convert_paths_to_PointSet(m[key])
                        incorporate_motion_edges(m[key], deformsB, s2.origin)
                        convert_paths_to_ndarray(m[key])
                        modelmesh[key] = create_mesh_with_abs_displacement(m[key], 
                                         radius = 0.7, dim = dimension, motion = motion)
                else:
                    modelmesh[key] = create_mesh(m[key], 0.7, fullPaths = False)
                
        ## Start vis
        f = vv.figure(nr); vv.clf()
        if nr == 1:
            f.position = 8.00, 30.00,  944.00, 1002.00
        else:
            f.position = 968.00, 30.00,  944.00, 1002.00
        a = vv.cla()
        a.axis.axisColor = 1,1,1
        a.axis.visible = False
        a.bgcolor = 0,0,0
        a.daspect = 1, 1, -1
        t = vv.volshow(vol, clim=clim0, renderStyle=showVol)
        vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
        if meshWithColors:
            vv.title('Model for ChEVAS %s  (colorbar \b{%s} of motion in mm in %s)' % (ptcode[7:], motion, dimension))
        else:
            vv.title('Model for ChEVAS %s' % (ptcode[7:]))
        # m = vv.mesh(modelmesh)
        # # m.faceColor = 'g'
        # m.clim = 0, 5
        # m.colormap = vv.CM_JET
        
        for i in modelmesh:
            # Create deformable mesh
            dm = DeformableMesh(a, modelmesh[i]) # in x,y,z
            dm.SetDeforms(*[list(reversed(deform)) for deform in deforms_f]) # from z,y,x to x,y,z
            if meshWithColors:
                dm.clim = clim2
                dm.colormap = vv.CM_JET
                vv.colorbar()
            else:
                dm.faceColor = 'g'
            
            # Run mesh
            a.SetLimits()
            # a.SetView(viewringcrop)
            dm.MotionPlay(motionPlay[0], motionPlay[1])  # (10, 0.2) = each 10 ms do a step of 20%
            dm.motionSplineType = 'B-spline'
            dm.motionAmplitude = 1.0  # For a mesh we can (more) safely increase amplitude
                
        ## Turn on/off axis
        # vv.figure(1); a1 = vv.gca(); vv.figure(2); a2= vv.gca()
        # 
        # switch = False
        # 
        # a1.axis.visible = switch
        # a2.axis.visible = switch
        
        ## Use same camera when 2 models are running
        # a1.camera = a2.camera
        
        ## Turn on/off moving mesh
        
        # dm.visible = False
        # dm.visible = True
        

def convert_paths_to_PointSet(g):
    """ for centerline graphs that were created with paths as list
    """
    from stentseg.utils import PointSet
    import numpy as np
    for n1, n2 in g.edges():
        path2 = PointSet(3, dtype=np.float32)
        # Obtain path of edge
        path = g.edge[n1][n2]['path']
        for p in path:
            path2.append(tuple(p.flat))
        g.edge[n1][n2]['path'] = path2.copy()

def convert_paths_to_ndarray(g):
    """ for dynamic centerline graphs that were created with paths as list
    """
    import numpy as np
    for n1, n2 in g.edges():
        # Obtain path of edge
        path = g.edge[n1][n2]['path']
        pathdeforms = g.edge[n1][n2]['pathdeforms']
        path2 = np.concatenate([a for a in path], axis=0) # or np.asarray
        pathdeforms3 = []
        for p in pathdeforms: # for every PointSet
            pathdeforms2 = np.concatenate([a for a in p], axis=0)
            pathdeforms3.append(pathdeforms2)
        g.edge[n1][n2]['path'] = path2.copy()
        g.edge[n1][n2]['pathdeforms'] = pathdeforms3.copy()
