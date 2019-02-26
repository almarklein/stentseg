
class _Get_Centerline:
    def __init__(self,ptcode,ctcode,StartPoints,EndPoints,basedir):
        """ with start and endpoints provided, calculate centerline and save as
        ssdf in basedir as model and dynamic model
        """
        #todo: name of dynamic model is now deforms, should be dynamic   
        #import numpy as np
        import visvis as vv
        import numpy as np
        import os
        import copy
        
        from stentseg.utils import PointSet, _utils_GUI
        from stentseg.utils.centerline import (find_centerline, 
        points_from_nodes_in_graph, points_from_mesh, smooth_centerline)
        from stentseg.utils.datahandling import loadmodel, loadvol
        from stentseg.utils.visualization import show_ctvolume
        from stentseg.utils.picker import pick3d
        
        stentnr = len(StartPoints)
        
        cropname = 'prox'
        what = 'modelavgreg'
        what_vol = 'avgreg'
        vismids = True
        m = loadmodel(basedir, ptcode, ctcode, cropname, what)
        s = loadvol(basedir, ptcode, ctcode, cropname, what_vol)
        s.vol.sampling = [s.sampling[1], s.sampling[1], s.sampling[2]]
        s.sampling = s.vol.sampling
        
        start1 = StartPoints.copy()
        ends = EndPoints.copy()
        
        from stentseg.stentdirect import stentgraph
        ppp = points_from_nodes_in_graph(m.model)
        
        allcenterlines = [] # for pp
        allcenterlines_nosmooth = [] # for pp
        centerlines = [] # for stentgraph
        nodes_total = stentgraph.StentGraph()
        for j in range(stentnr):
            if j == 0 or not start1[j] == ends[j-1]:
                # if first stent or when stent did not continue with this start point
                nodes = stentgraph.StentGraph()
                centerline = PointSet(3) # empty
            
            # Find main centerline
            # if j > 3: # for stent with midpoints
            #     centerline1 = find_centerline(ppp, start1[j], ends[j], step= 1, 
            #     ndist=10, regfactor=0.5, regsteps=10, verbose=False)
            
            #else:
            centerline1 = find_centerline(ppp, start1[j], ends[j], step= 1, 
                ndist=10, regfactor=0.5, regsteps=1, verbose=False)
                                        # centerline1 is a PointSet
                

            print('Centerline calculation completed')
            
            # ========= Maaike =======
            smoothfactor = 15  # Mirthe used 2 or 4
            
            # check if cll continued here from last end point
            if not j == 0 and start1[j] == ends[j-1]:
                # yes we continued
                ppart = centerline1[:-1] # cut last but do not cut first point as this is midpoint
            else:
                # do not use first points, as they are influenced by user selected points
                ppart = centerline1[1:-1]
            
            for p in ppart:
                centerline.append(p)
            
            # if last stent or stent does not continue with next start-endpoint
            if j == stentnr-1 or not ends[j] == start1[j+1]:
                # store non-smoothed for vis
                allcenterlines_nosmooth.append(centerline)
                pp = smooth_centerline(centerline, n=smoothfactor)
                # add pp to list
                allcenterlines.append(pp) # list with PointSet per centerline
                self.allcenterlines = allcenterlines 
            
                # add pp as nodes    
                for i, p in enumerate(pp):
                    p_as_tuple = tuple(p.flat)
                    nodes.add_node(p_as_tuple)
                    nodes_total.add_node(p_as_tuple)
                # add pp as one edge so that pathpoints are in fixed order
                pstart = tuple(pp[0].flat)
                pend = tuple(pp[-1].flat)
                nodes.add_edge(pstart, pend, path = pp  )
                nodes_total.add_edge(pstart, pend, path = pp  )
                # add final centerline nodes model to list
                centerlines.append(nodes)
                
            # ========= Maaike =======
        
        ## Store segmentation to disk
         
        # Build struct
        s2 = vv.ssdf.new()
        s2.sampling = s.sampling
        s2.origin = s.origin
        s2.stenttype = m.stenttype
        s2.croprange = m.croprange
        for key in dir(m):
                if key.startswith('meta'):
                    suffix = key[4:]
                    s2['meta'+suffix] = m['meta'+suffix]
        s2.what = what
        s2.params = s.params #reg
        s2.paramsseeds = m.params
        s2.stentType = 'nellix'
        s2.StartPoints = StartPoints
        s2.EndPoints = EndPoints
        # keep centerlines as pp also [Maaike]
        s2.ppallCenterlines = allcenterlines
        for k in range(len(allcenterlines)):
            suffix = str(k)
            pp = allcenterlines[k]
            s2['ppCenterline' + suffix] = pp
        
        s3 = copy.deepcopy(s2)
        s3['model'] = nodes_total.pack()

        # Store model for each centerline
        for j in range(len(centerlines)):
            suffix = str(j)
            model = centerlines[j]
            s2['model' + suffix] = model.pack()
        
        # Save model with seperate centerlines.
        filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'centerline_'+what)
        vv.ssdf.save(os.path.join(basedir, ptcode, filename), s2)
        print('saved to disk as {}.'.format(filename) )
        
        # Save model with combined centerlines
        filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'centerline_total_'+what)
        vv.ssdf.save(os.path.join(basedir, ptcode, filename), s3)
        print('saved to disk as {}.'.format(filename) )
        
        # remove intermediate centerline points
        # start1 = map(tuple, start1) 
        # ends = map(tuple, ends)
        startpoints_clean = copy.deepcopy(start1)
        endpoints_clean = copy.deepcopy(ends)
        duplicates = list(set(start1) & set(ends))
        for i in range(len(duplicates)):
            startpoints_clean.remove(duplicates[i])
            endpoints_clean.remove(duplicates[i])
        
        #Visualize
        f = vv.figure(10); vv.clf()
        a1 = vv.subplot(121)
        a1.daspect = 1, 1, -1
       
        vv.plot(ppp, ms='.', ls='', alpha=0.6, mw=2)
        for j in range(len(startpoints_clean)):
            vv.plot(PointSet(list(startpoints_clean[j])), ms='.', ls='', mc='g', mw=20) # startpoint green
            vv.plot(PointSet(list(endpoints_clean[j])),  ms='.', ls='', mc='r', mw=20) # endpoint red
        for j in range(len(allcenterlines)):
            vv.plot(allcenterlines[j], ms='.', ls='', mw=10, mc='y')        
        vv.title('Centerlines and seed points')
        vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
        # for j in range(len(allcenterlines_nosmooth)):
        #     vv.plot(allcenterlines_nosmooth[j], ms='o', ls='', mw=10, mc='c', alpha=0.6)
        
        a2 = vv.subplot(122)
        a2.daspect = 1, 1, -1
        
        vv.plot(ppp, ms='.', ls='', alpha=0.6, mw=2)
        # vv.volshow(s.vol, clim=clim, renderStyle = 'mip')           
        t = show_ctvolume(s.vol, axis=a2, showVol='ISO', clim =(0,2500), isoTh=250, 
                        removeStent=False, climEditor=True)
        label = pick3d(vv.gca(), s.vol)
        for j in range(len(startpoints_clean)):
            vv.plot(PointSet(list(startpoints_clean[j])), ms='.', ls='', mc='g', 
                    mw=20, alpha=0.6) # startpoint green
            vv.plot(PointSet(list(endpoints_clean[j])),  ms='.', ls='', mc='r', 
                    mw=20, alpha=0.6) # endpoint red
        for j in range(len(allcenterlines)):
            vv.plot(allcenterlines[j], ms='o', ls='', mw=10, mc='y', alpha=0.6)
        
        # show midpoints (e.g. duplicates)
        if vismids:
            for p in duplicates:
                vv.plot(p[0], p[1], p[2], mc= 'm', ms = 'o', mw= 10, alpha=0.6)
        
        a2.axis.visible = False
        
        vv.title('Centerlines and seed points')
        
        a1.camera = a2.camera
        
        f.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView( event, [a1,a2]) )
        f.eventKeyDown.Bind(lambda event: _utils_GUI.ViewPresets(event, [a1,a2]) )
        
        # Pick node for midpoint to redo get_centerline
        self.pickedCLLpoint = _utils_GUI.Event_pick_graph_point(nodes_total, s.vol, label, nodesOnly=True) # x,y,z
        # use key p to select point
        
        #===============================================================================
        vv.figure(11)
        vv.gca().daspect = 1,1,-1
        t = show_ctvolume(s.vol, showVol='ISO', clim =(0,2500), isoTh=250, 
                        removeStent=False, climEditor=True)
        label2 = pick3d(vv.gca(), s.vol)
        for j in range(len(startpoints_clean)):
            vv.plot(PointSet(list(startpoints_clean[j])), ms='.', ls='', mc='g', 
                    mw=20, alpha=0.6) # startpoint green
            vv.plot(PointSet(list(endpoints_clean[j])),  ms='.', ls='', mc='r', 
                    mw=20, alpha=0.6) # endpoint red
        vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')    
        #===============================================================================
                 
        ## Make model dynamic (and store/overwrite to disk) 
        import pirt
        from stentseg.motion.dynamic import incorporate_motion_nodes, incorporate_motion_edges  
         
        # Load deforms
        filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'deforms')
        s1 = vv.ssdf.load(os.path.join(basedir, ptcode, filename))
        deformkeys = []
        for key in dir(s1):
            if key.startswith('deform'):
                deformkeys.append(key)
        deforms = [s1[key] for key in deformkeys]
        deforms = [pirt.DeformationFieldBackward(*fields) for fields in deforms]
        for i in range(len(deforms)):
                deforms[i]._field_sampling = tuple(s1.sampling)
        paramsreg = s1.params
         
        # Load model
        s2 = loadmodel(basedir, ptcode, ctcode, cropname, 'centerline_'+what)
        s3 = loadmodel(basedir, ptcode, ctcode, cropname, 'centerline_total_'+what)
        
        # Combine ...
        for key in dir(s2):
                if key.startswith('model'):
                    incorporate_motion_nodes(s2[key], deforms, s.origin)
                    incorporate_motion_edges(s2[key], deforms, s.origin)
                    model = s2[key]
                    s2[key] = model.pack()
        # Combine ...
        for key in dir(s3):
                if key.startswith('model'):
                    incorporate_motion_nodes(s3[key], deforms, s.origin)
                    incorporate_motion_edges(s3[key], deforms, s.origin)
                    model = s3[key]
                    s3[key] = model.pack()
                    
        # Save
        s2.paramsreg = paramsreg
        filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'centerline_'+what+'_deforms')
        vv.ssdf.save(os.path.join(basedir, ptcode, filename), s2)
        print('saved to disk as {}.'.format(filename) )
        
        # Save
        s3.paramsreg = paramsreg
        filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'centerline_total_'+what+'_deforms')
        vv.ssdf.save(os.path.join(basedir, ptcode, filename), s3)
        print('saved to disk as {}.'.format(filename) )
        