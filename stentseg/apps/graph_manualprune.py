def interactiveClusterRemovalGraph(graph, radius=0.7, 
        faceColor=(0.5,1.0,0.3), selectColor=(1.0,0.3, 0.3) ):
    """ showGraphAsMesh(graph, radius=0.7, 
                faceColor=(0.5,1.0,0.3), selectColor=(1.0,0.3, 0.3) )
    
    Manual delete clusters in the graph. Show the given graph as a mesh, or to 
    be more precize as a set of meshes representing the clusters of the graph. 
    By holding the mouse over a mesh, it can be selected, after which it can be 
    deleted by pressing delete. Use sd._nodes3 for graph when in segmentation.
    
    Returns the axes in which the meshes are drawn.
    
    """
    import networkx as nx
    from stentseg.stentdirect import stentgraph
    from stentseg.stentdirect.stentgraph import create_mesh
    
    # Get clusters of nodes
#     clusters = graph.CollectGroups() # old code
    clusters = list(nx.connected_components(graph))
    
    # Build meshes 
    meshes = []
    for cluster in clusters:
        g = graph.copy()
        for c in clusters:
            if not c == cluster: 
                g.remove_nodes_from(c)
        
        # Convert to mesh (this takes a while)
#         bm = g.CreateMesh(radius) # old code
        bm = create_mesh(g, radius = 0.7)
        
        # Store
        meshes.append(bm)
    
    # Define callback functions
    def meshEnterEvent(event):
        event.owner.faceColor = selectColor
    def meshLeaveEvent(event):
        event.owner.faceColor = faceColor
    def figureKeyEvent(event):
        if event.key == vv.KEY_DELETE:
            m = event.owner.underMouse
            if hasattr(m, 'faceColor'):
                m.Destroy()
                graph.remove_nodes_from(clusters[m.index])
     
    # Visualize
    a = vv.gca()
    fig = a.GetFigure()
    for i, bm in enumerate(meshes):
#         m = vv.Mesh(a, bm) # old code
        m = vv.mesh(bm)
        m.faceColor = faceColor
        m.eventEnter.Bind(meshEnterEvent)
        m.eventLeave.Bind(meshLeaveEvent)
        m.hitTest = True
        m.index = i
    #
    fig.eventKeyDown.Bind(figureKeyEvent)
    a.SetLimits()
    a.bgcolor = 'k'
    a.axis.axisColor = 'w'
    a.axis.visible = True
    a.daspect = 1, 1, -1
    
    # Prevent the callback functions from going out of scope
    a._callbacks = meshEnterEvent, meshLeaveEvent, figureKeyEvent
    
    # Done return axes
    return a
    