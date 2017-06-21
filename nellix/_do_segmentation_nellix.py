""" Script to do the segmentation and store the result.

Run as script will save the model:overwrites! [script contains 1/0 break to prevent this]
"""
class _Do_Segmentation:
    def __init__(self,ptcode,ctcode,basedir):
        
        import os
        
        import numpy as np
        import visvis as vv
        from visvis import ssdf
        
        from stentseg.utils import PointSet, _utils_GUI
        from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
        from stentseg.stentdirect.stentgraph import create_mesh
        from stentseg.stentdirect import stentgraph, StentDirect, getDefaultParams, AnacondaDirect, EndurantDirect, NellixDirect
        from stentseg.utils.picker import pick3d, label2worldcoordinates, label2volindices
        # from stentseg.apps.graph_manualprune import interactiveClusterRemovalGraph
        import scipy
        from scipy import ndimage
        import copy

        # Select the ssdf basedir
        #basedir = r'C:\Users\User\Desktop\Nellix_chevas\CHEVAS_SSDF' 
        
        
        # Select dataset to register
        cropname = 'prox'
        phase = 10
        #dataset = 'avgreg'
        #what = str(phase) + dataset # avgreg
        what = 'avgreg'
        
        # Load volumes
        s = loadvol(basedir, ptcode, ctcode, cropname, what)
        
        #todo: MK redundant??
        vol_org = copy.deepcopy(s.vol)
        s.vol.sampling = [vol_org.sampling[1], vol_org.sampling[1], vol_org.sampling[2]] # z,y,x
        s.sampling = s.vol.sampling
        vol = s.vol
        
        
        ## Initialize segmentation parameters
        stentType = 'nellix'  # 'zenith';'nellix' runs modified pruning algorithm in Step3
        
        p = getDefaultParams(stentType)
        p.seed_threshold = [600]     # step 1 [lower th] or [lower th, higher th]
        p.mcp_speedFactor = 100           # step 2, costToCtValue; lower less cost for lower HU; higher more cost for lower HU
        p.mcp_maxCoverageFronts = 0.004    # step 2, base.py; replaces mcp_evolutionThreshold
        p.graph_weakThreshold = 200             # step 3, stentgraph.prune_very_weak
        p.graph_expectedNumberOfEdges = 2       # step 3, stentgraph.prune_weak
        p.graph_trimLength =  0                 # step 3, stentgraph.prune_tails
        p.graph_minimumClusterSize = 3         # step 3, stentgraph.prune_clusters
        p.graph_strongThreshold = 1800          # step 3, stentgraph.prune_weak and stentgraph.prune_redundant
        p.graph_angleVector = 5                 # step 3, corner detect
        p.graph_angleTh = 45                  # step 3, corner detect
        # p.seedSampleRate = 7                  # step 1, nellix
        p.whatphase = phase                 # step 1, select the mask algorithm depending on phase 
        
        ## Perform segmentation
        # Instantiate stentdirect segmenter object
        if stentType == 'anacondaRing':
                sd = AnacondaDirect(vol, p) # inherit _Step3_iter from AnacondaDirect class
                #runtime warning using anacondadirect due to mesh creation, ignore
        elif stentType == 'endurant':
                sd = EndurantDirect(vol, p)
        elif stentType == 'nellix':
                sd = NellixDirect(vol, p)
        else:
                sd = StentDirect(vol, p) 
        
        # Perform the three steps of stentDirect
        sd.Step1()
        
        ## Visualize
        
        fig = vv.figure(2); vv.clf()
        fig.position = 0.00, 22.00,  1920.00, 1018.00
        clim = (0,2000)
        
        # Show volume and model as graph
        a1 = vv.subplot(121)
        a1.daspect = 1,1,-1
        t = vv.volshow(vol, clim=clim)
        label = pick3d(vv.gca(), vol)
        sd._nodes1.Draw(mc='b', mw = 2)       # draw seeded nodes
        #sd._nodes2.Draw(mc='b', lc = 'g')    # draw seeded and MCP connected nodes
        vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
        
        # Show volume and cleaned up graph
        a2 = vv.subplot(122)
        a2.daspect = 1,1,-1
        sd._nodes1.Draw(mc='b', mw = 2)       # draw seeded nodes
        # t = vv.volshow(vol, clim=clim)
        # label = pick3d(vv.gca(), vol)
        # sd._nodes2.Draw(mc='b', lc='g')
        # sd._nodes3.Draw(mc='b', lc='g')
        vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
        
        # # Show the mesh
        #===============================================================================
        # a3 = vv.subplot(133)
        # a3.daspect = 1,1,-1
        # t = vv.volshow(vol, clim=clim)
        # pick3d(vv.gca(), vol)
        # #sd._nodes3.Draw(mc='b', lc='g')
        # m = vv.mesh(bm)
        # m.faceColor = 'g'
        # # _utils_GUI.vis_spared_edges(sd._nodes3)
        # vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
        #===============================================================================
        
        # Use same camera
        a1.camera = a2.camera #= a3.camera
        
        switch = True
        a1.axis.visible = switch
        a2.axis.visible = switch
        #a3.axis.visible = switch
        

        ## Store segmentation to disk
        
        # Get graph model
        model = sd._nodes1
                
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
        # Store model
        s2.model = model.pack()
        s2.vol = vol_org
        # s2.sampling_interp = s.sampling
        s2.vol_interp = vol
        
        # Save
        filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'model'+ what)
        ssdf.save(os.path.join(basedir, ptcode, filename), s2)
        print('saved to disk as {}.'.format(filename) )

        ## Make model dynamic (and store/overwrite to disk)
        #===============================================================================
        # 
        # import pirt
        # from stentsegf.motion.dynamic import incorporate_motion_nodes, incorporate_motion_edges  
        # 
        # # Load deforms
        # s = loadvol(basedir, ptcode, ctcode, cropname, '10deforms')
        # deformkeys = []
        # for key in dir(s):
        #     if key.startswith('deform'):
        #         deformkeys.append(key)
        # deforms = [s[key] for key in deformkeys]
        # deforms = [pirt.DeformationFieldBackward(*fields) for fields in deforms]
        # paramsreg = s.params
        # 
        # # Load model
        # s = loadmodel(basedir, ptcode, ctcode, cropname, 'model'+what)
        # model = s.model
        # 
        # # Combine ...
        # incorporate_motion_nodes(model, deforms, s.origin)
        # incorporate_motion_edges(model, deforms, s.origin)
        # 
        # # Save back
        # filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'model'+what)
        # s.model = model.pack()
        # s.paramsreg = paramsreg
        # ssdf.save(os.path.join(basedir, ptcode, filename), s)
        # print('saved to disk as {}.'.format(filename) )
        #===============================================================================