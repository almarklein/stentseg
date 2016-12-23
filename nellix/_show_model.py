
class _Show_Model:
    def __init__(self,ptcode,basedir):
        """
        Script to show the stent model. [ nellix]
        """
        
        import os
        import pirt
        import visvis as vv
        
        from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
        from pirt.utils.deformvis import DeformableTexture3D, DeformableMesh
        from stentseg.stentdirect.stentgraph import create_mesh
        from stentseg.motion.vis import show_ctvolume
        from stentseg.motion.vis import create_mesh_with_abs_displacement
        import copy

        import numpy as np
        #from stentseg.motion.displacement import _calculateAmplitude, _calculateSumMotion
        #from stentseg.motion.displacement import calculateMeanAmplitude
        
        cropname = 'prox'
        ctcode = '12months'
        print ('ctcode')
        # params
        nr = 40
        motion = 'amplitude'  # amplitude or sum
        dimension = 'xyz'
        showVol  = 'mip'  # MIP or ISO or 2D or None
        clim0  = (0,2000)
        motionPlay = 1000/25, 0.2   # each x ms, a step of x %
        
        # Load deformations (forward for mesh)
        s = loadvol(basedir, ptcode, ctcode, cropname, what = 'deforms')
        m = loadmodel(basedir, ptcode, ctcode, cropname, 'centerline_modelavgreg')
        m.sampling = [m.sampling[1], m.sampling[1], m.sampling[2]]
        
        # deforms = [s['deform%i'%(i*10)] for i in range(10)]
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
                modelmesh[key] = create_mesh(m[key], 0.5, fullPaths = False)
                
        # radius : scalar
        #     The radius of the tube that is created. Radius can also be a 
        #     sequence of values (containing a radius for each point).
        
        #modelmesh = create_mesh_with_abs_displacement(model, radius = 1.0, dim = dimension, motion = motion)
        
        ## Start vis
        m = loadvol(basedir, ptcode, ctcode, cropname, what='avgreg')
        vol_org = copy.deepcopy(m.vol)
        m.vol.sampling = [vol_org.sampling[1], vol_org.sampling[1], vol_org.sampling[2]]
        m.sampling = m.vol.sampling
        vol = m.vol
        
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
        vv.title('Model for LSPEAS %s  -  %s  (colorbar \b{%s} of motion in mm in %s)' % (ptcode[7:], ctcode, motion, dimension))
        # m = vv.mesh(modelmesh)
        # # m.faceColor = 'g'
        # m.clim = 0, 5
        # m.colormap = vv.CM_JET
        
        for i in modelmesh:
            dm = []
            # Create deformable mesh
            dm = DeformableMesh(a, modelmesh[i]) # in x,y,z
            dm.SetDeforms(*[list(reversed(deform)) for deform in deforms_f]) # from z,y,x to x,y,z
            dm.clim = clim0
            # dm.colormap = vv.CM_JET
            # vv.colorbar()
            
            # Run mesh
            a.SetLimits()
            # a.SetView(viewringcrop)
            dm.MotionPlay(motionPlay[0], motionPlay[1])  # (10, 0.2) = each 10 ms do a step of 20%
            dm.motionSplineType = 'B-spline'
            dm.motionAmplitude = 1.0  # For a mesh we can (more) safely increase amplitude
            dm.faceColor = 'g'
                
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