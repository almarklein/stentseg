# todo: can easily be made independent of stentgraph if only we
# make sure that graph.Graph has a CreateMesh method

def showGraphAsMesh(graph, radius=0.7, 
        faceColor=(0.5,1.0,0.3), selectColor=(1.0,0.3, 0.3) ):
    """ showGraphAsMesh(graph, radius=0.7, 
                faceColor=(0.5,1.0,0.3), selectColor=(1.0,0.3, 0.3) )
    
    Show the given graph as a mesh, or to be more precize as a set
    of meshes representing the clusters of the graph. By holding the
    mouse over a mesh, it can be selected, after which it can be deleted
    by pressing delete.
    
    Returns the axes in which the meshes are drawn.
    
    """
    
    # Get clusters of nodes
    clusters = graph.CollectGroups()
    
    # Build meshes 
    meshes = []
    for cluster in clusters:
        
        # Create graph
        g = stentGraph.StentGraph()
        for node in cluster:
            g.AppendNode(node)
        
        # Convert to mesh (this takes a while)
        bm = g.CreateMesh(radius)
        
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
     
    # Visualize
    a = vv.gca()
    fig = a.GetFigure()
    for bm in meshes:
        m = vv.Mesh(a, bm)
        m.faceColor = faceColor
        m.eventEnter.Bind(meshEnterEvent)
        m.eventLeave.Bind(meshLeaveEvent)
        m.hitTest = True
    #
    fig.eventKeyDown.Bind(figureKeyEvent)
    a.SetLimits()
    a.bgcolor = 'k'
    a.axis.axisColor = 'w'
    
    # Prevent the callback functions from going out of scope
    a._callbacks = meshEnterEvent, meshLeaveEvent, figureKeyEvent
    
    # Done return axes
    return a
    