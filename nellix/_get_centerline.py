
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
        
        from stentseg.utils import PointSet
        from stentseg.utils.centerline import (find_centerline, 
        points_from_nodes_in_graph, points_from_mesh, smooth_centerline)
        from stentseg.utils.datahandling import loadmodel, loadvol
        from stentseg.utils.visualization import show_ctvolume

        stentnr = len(StartPoints)
        
        cropname = 'prox'
        what = 'modelavgreg'
        what_vol = 'avgreg'
        m = loadmodel(basedir, ptcode, ctcode, cropname, what)
        s = loadvol(basedir, ptcode, ctcode, cropname, what_vol)
        s.vol.sampling = [s.sampling[1], s.sampling[1], s.sampling[2]]
        s.sampling = s.vol.sampling
        
        start1 = StartPoints.copy()
        ends = EndPoints.copy()
        
        from stentseg.stentdirect import stentgraph
        ppp = points_from_nodes_in_graph(m.model)
        
        allcenterlines = []
        centerlines = []
        nodes_total = stentgraph.StentGraph()
        for j in range(stentnr):
            if j == 0 or not start1[j] == ends[j-1]:
                nodes = stentgraph.StentGraph()
            # Find main centerline
            # regsteps = distance of centerline points from where the start/end 
            # point have no affect on centerline finding
            # centerline1 = find_centerline(ppp, start1[j], ends[j], step= 0.5, 
            #                               substep=0.25, ndist=40, regfactor=0.9, 
            #                               regsteps=0.5, verbose=False)
            #                               # reg 0.9 for smoother; Mirthe
            centerline1 = find_centerline(ppp, start1[j], ends[j], step= 1, 
                                          ndist=20, regfactor=0.5, 
                                          regsteps=1, verbose=False)
                                          # reg 0.9 for smoother 
            print('Centerline calculation completed')
            # do not use first 2 points, as they are influenced by user selected points
            pp = centerline1[2:-2]
            # smooth the cut centerline
            pp = smooth_centerline(pp, n=20) # Mirthe used 4
            # pp = centerline1
          
            allcenterlines.append(pp) # list with PointSet per centerline
            self.allcenterlines = allcenterlines 
            
            for i, p in enumerate(pp[:-1]):
                p_as_tuple = tuple(p.flat)
                p1_as_tuple = tuple(pp[i+1].flat)
                # nodes.add_node(p_as_tuple)
                path = PointSet(3, dtype=np.float32)
                for p in [p_as_tuple, p1_as_tuple]:
                    path.append(p)
                nodes.add_edge(p_as_tuple, p1_as_tuple, path = path  )
                nodes_total.add_edge(p_as_tuple, p1_as_tuple, path = path  )
                
            if j == stentnr-1 or not ends[j] == start1[j+1]:
                centerlines.append(nodes)
        
        
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
        s2.params = s.params
        s2.stentType = 'nellix'
        s2.StartPoints = StartPoints
        s2.EndPoints = EndPoints
        # keep centerlines as pp also [Maaike]
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
        start1 = tuple(map(tuple, start1)) # added MK for when stored as array
        ends = tuple(map(tuple, ends)) # "
        startpoints_clean = copy.deepcopy(start1)
        endpoints_clean = copy.deepcopy(ends)
        duplicates = list(set(start1) & set(ends))
        for i in range(len(duplicates)):
            startpoints_clean.remove(duplicates[i])
            endpoints_clean.remove(duplicates[i])
        
        #Visualize
        vv.figure(10); vv.clf()
        a1 = vv.subplot(121)
        a1.daspect = 1, 1, -1
       
        vv.plot(ppp, ms='.', ls='', alpha=0.6, mw=2)
        for j in range(len(startpoints_clean)):
            vv.plot(PointSet(list(startpoints_clean[j])), ms='.', ls='', mc='g', mw=20) # startpoint green
            vv.plot(PointSet(list(endpoints_clean[j])),  ms='.', ls='', mc='r', mw=20) # endpoint red
        for j in range(stentnr):
            vv.plot(allcenterlines[j], ms='.', ls='', mw=10, mc='y')        
        vv.title('Centerlines and seed points')
        vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
        
        
        a2 = vv.subplot(122)
        a2.daspect = 1, 1, -1
        
        vv.plot(ppp, ms='.', ls='', alpha=0.6, mw=2)
        for j in range(len(startpoints_clean)):
            vv.plot(PointSet(list(startpoints_clean[j])), ms='.', ls='', mc='g', mw=20) # startpoint green
            vv.plot(PointSet(list(endpoints_clean[j])),  ms='.', ls='', mc='r', mw=20) # endpoint red
        for j in range(stentnr):
            vv.plot(allcenterlines[j], ms='.', ls='', mw=10, mc='y')
        clim = (0,2000)
        # vv.volshow(s.vol, clim=clim, renderStyle = 'mip')           
        t = show_ctvolume(s.vol, axis=a2, showVol='MIP', clim =(0,2500), isoTh=250, 
                        removeStent=False, climEditor=True)
        a2.axis.visible = False
        
        vv.title('Centerlines and seed points')
        
        a1.camera = a2.camera
        
        
        #===============================================================================
        # vv.figure()
        # vv.gca().daspect = 1,1,-1
        # t = vv.volshow(vol, clim=(-500, 1500))
        # nodes.Draw(mc='b', mw = 6, lc = 'g',lw=3)       # draw seeded nodes
        # vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')    
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
        