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
    if event.key == vv.KEY_ESCAPE:
        make_model_dynamic(model, deforms, origin)
        a = vv.gca()
        view = a.GetView()
        a.Clear()
        lim = (0,2500)
        t = vv.volshow(vol, clim=lim, renderStyle='mip')
        pick3d(vv.gca(), vol)
        model.Draw(mc='g', mw = 10, lc='g')
        a.SetView(view)
        print('Done, model dynamic')


def make_model_dynamic(model, deforms,  modelOrigin):
    #todo: why are deforms stored as PointSet and not ndarray as in saved dynamic models?
    # #first clear deforms in graph, to maintain ndarray and not get PointSet
    # for n1,n2 in model.edges():
    #     attre = model.edge[n1][n2]
    #     if 'pathdeforms' in attre:
    #         del attre['pathdeforms']
    #     for n in [n1, n2]:
    #         attrn = model.node[n]
    #         if 'deforms' in attrn:
    #             del attrn['deforms']
    # combine model with deforms        
    incorporate_motion_nodes(model, deforms, modelOrigin)
    incorporate_motion_edges(model, deforms, modelOrigin)
    # # try pack unpack, nu wel ndarray?
    # s.model = model.pack()
    # model = stentgraph.StentGraph()
    # model.unpack(s.model)


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
    
    #load deforms
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
    
    
    