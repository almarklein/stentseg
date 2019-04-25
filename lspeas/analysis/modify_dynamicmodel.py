# created august 2016 - Maaike Koenrades

""" Modify dynamic model and make dynamic again

"""

def on_key(event):
    """KEY commands for user interaction
    UP/DOWN = show/hide nodes
    DELETE  = remove edge [select 2 nodes] or pop node [select 1 node] or remove seed sd._nodes1 closest to [picked point]
    PageDown= remove graph posterior (y-axis) to [picked point] (use for spine seeds)'
    ALT     = SHOW RESULT after remove residual clusters, pop, corner
    CTRL    = add selected point (SHIFT+Rclick) as seed in sd._nodes1')
    """
    if event.key == vv.KEY_DOWN:
        # hide nodes
        for node_point in node_points:
            node_point.visible = False
    if event.key == vv.KEY_UP:
        # show nodes
        for node_point in node_points:
            node_point.visible = True
    if event.key == vv.KEY_DELETE:
        if len(selected_nodes) == 2:
            # remove edge
            select1 = selected_nodes[0].node
            select2 = selected_nodes[1].node
            c, ct, p, l = _utils_GUI.get_edge_attributes(model, select1, select2)
            model.remove_edge(select1, select2)
            # visualize removed edge, show keys and deselect nodes
            selected_nodes[1].faceColor = 'b'
            selected_nodes[0].faceColor = 'b'
            selected_nodes.clear()
            a = vv.gca()
            view = a.GetView()
            pp = Pointset(p)
            line = vv.solidLine(pp, radius = 0.2)
            line.faceColor = 'r'
            a.SetView(view)
        if len(selected_nodes) == 1:
            # pop node
            select1 = selected_nodes[0].node
            stentgraph._pop_node(model, select1) # asserts degree == 2
            selected_nodes[0].faceColor = 'w'
            selected_nodes.clear()
    if event.text == 's':
        # additional smooth
        stentgraph.smooth_paths(model, 2)
        reDrawModel(vol, model)
    if event.key == vv.KEY_ALT:
        # ALT will POP nodes
        stentgraph.pop_nodes(model)
        reDrawModel(vol, model)
    if event.key == vv.KEY_ESCAPE:
        g = make_model_dynamic(model, deforms, origin)
        reDrawModel(vol, g, interactive=False)
        # Save back
        filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, modelname)
        s.model = g.pack()
        print('Done, model dynamic')
        # ssdf.save(os.path.join(basedir, ptcode, filename), s)
        # print('saved dynamic to disk in {} as {}.'.format(basedir, filename) )

def reDrawModel(vol, model, interactive=True, ax=None):
        if ax is None:
            ax = vv.gca()
        view = ax.GetView()
        ax.Clear()
        lim = (0,2500)
        t = vv.volshow(vol, clim=lim, renderStyle='mip')
        pick3d(vv.gca(), vol)
        model.Draw(mc='g', mw = 10, lc='g')
        if interactive:
            #todo: doesnt work?
            node_points = _utils_GUI.interactive_node_points(model, scale=0.6)
            _utils_GUI.node_points_callbacks(node_points, selected_nodes, pick=False)
        ax.SetView(view)
        

def make_model_dynamic(model, deforms, origin):
    # modelpacked = model.pack()
    # g = stentgraph.StentGraph()
    # g.unpack(modelpacked)
    # First clear deforms in graph 
    for n in model.nodes():
        d = model.node[n]
        # use dictionary comprehension to delete key
        for key in [key for key in d if key == 'deforms']: del d[key]
    
    for n1,n2 in model.edges():
        d = model.edge[n1][n2]
        # use dictionary comprehension to delete key
        for key in [key for key in d if key == 'pathdeforms']: del d[key]
    
    incorporate_motion_nodes(model, deforms, origin) # adds deforms PointSets
    incorporate_motion_edges(model, deforms, origin) # adds deforms PointSets
    
    return model

def nodeInteraction(model, vol, selected_nodes): 
    """ modify edges in dynamic model and make dynamic again
    """
    f = vv.figure(); vv.clf()
    f.position = 968.00, 30.00,  944.00, 1002.00
    a = vv.gca()
    a.axis.axisColor = 1,1,1
    a.axis.visible = False
    a.bgcolor = 0,0,0
    a.daspect = 1, 1, -1
    lim = (0,2500)
    t = vv.volshow(vol, clim=lim, renderStyle='mip')
    pick3d(vv.gca(), vol)
    model.Draw(mc='b', mw = 10, lc='g')
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('User interaction to modify model %s  -  %s' % (ptcode[-3:], ctcode))
    
    # create clickable nodes
    node_points = _utils_GUI.interactive_node_points(model, scale=0.6)
    
    # bind event handlers
    f.eventKeyDown.Bind(on_key)
    for node_point in node_points:
        node_point.eventDoubleClick.Bind(lambda event: _utils_GUI.select_node(event, selected_nodes) )
    print('')
    print('UP/DOWN = show/hide nodes')
    print('DELETE  = remove edge [select 2 nodes] or pop node [select 1 node]')
    print('ESCAPE  = make model dynamic')
    print('')
    
    return node_points


if __name__ == '__main__':
    
    import pirt
    from stentseg.motion.dynamic import incorporate_motion_nodes, incorporate_motion_edges
    from stentseg.stentdirect import stentgraph
    from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
    import os
    import visvis as vv
    from visvis import ssdf
    from visvis import Pointset
    from stentseg.utils.picker import pick3d, get_picked_seed
    from stentseg.utils import PointSet, _utils_GUI, visualization
    
    basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                        r'D:\LSPEAS\LSPEAS_ssdf',
                        r'F:\LSPEAS_ssdf_BACKUP',r'G:\LSPEAS_ssdf_BACKUP')

    ptcode = 'LSPEAS_011'
    ctcode = '1month'
    cropname = 'ring'
    modelname = 'modelavgreg'
    
    s = loadmodel(basedir, ptcode, ctcode, cropname, modelname)
    model = s.model.copy()
    
    # Load volume
    s = loadvol(basedir, ptcode, ctcode, cropname, 'avgreg')
    vol = s.vol
    
    # Load deforms
    s_deforms = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
    deformkeys = []
    for key in dir(s_deforms):
        if key.startswith('deform'):
            deformkeys.append(key)
    deforms = [s_deforms[key] for key in deformkeys]
    deforms = [pirt.DeformationFieldBackward(*fields) for fields in deforms]
    
    selected_nodes = list()
    node_points = nodeInteraction(model, vol, selected_nodes)
    
    origin = s_deforms.origin # origin of vol

    