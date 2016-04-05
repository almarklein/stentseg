""" Module to obtain the different parts of the Anaconda dual ring

Hooks, struts, 2nd ring, top ring
""" 


def add_nodes_edge_to_newmodel(modelnew, model,n,neighbour):
    """ Get edge and nodes with attributes from model and add to newmodel
    """
    c = model.edge[n][neighbour]['cost']
    ct = model.edge[n][neighbour]['ctvalue']
    p = model.edge[n][neighbour]['path']
    pdeforms = model.edge[n][neighbour]['pathdeforms']
    modelnew.add_node(n, deforms = model.node[n]['deforms'])
    modelnew.add_node(neighbour, deforms = model.node[neighbour]['deforms'])
    modelnew.add_edge(n, neighbour, cost = c, ctvalue = ct, path = p, pathdeforms = pdeforms)
    return


def _get_model_hooks(model):
    """Get model hooks
    Return model without hooks and model with hooks only
    """
    import numpy as np
    from stentseg.stentdirect import stentgraph
    
    # initialize 
    model_noHooks = model.copy()
    model_hooks = stentgraph.StentGraph() # graph for hooks
    hooknodes = list() # remember nodes that belong to hooks 
    for n in sorted(model.nodes() ):
        if model.degree(n) == 1:
            neighbour = list(model.edge[n].keys())
            neighbour = neighbour[0]
            add_nodes_edge_to_newmodel(model_hooks,model,n,neighbour)
            hooknodes.append(neighbour)
            model_noHooks.remove_node(n) # this also removes the connecting edge
    
    return model_hooks, model_noHooks
    

def get_model_struts(model, nstruts=8):
    """Get struts between R1 and R2
    Detects them based on z-orientation and length
    Runs _get_model_hooks 
    """
    from stentseg.stentdirect.stent_anaconda import _edge_length
    from stentseg.stentdirect import stentgraph
    import numpy as np
    
    # remove hooks if still there
    models = _get_model_hooks(model)
    model_hooks, model_noHooks = models[0], models[1]
    # initialize
    model_h_s = model_hooks.copy() # struts added to hook model
    model_struts = stentgraph.StentGraph()
    directions = []
    for n1, n2 in model_noHooks.edges():
        e_length = _edge_length(model, n1, n2)
        if (3.5 < e_length < 12): # struts OLB21 are 4.5-5.5mm OLB34 9-9.5
            vector = np.subtract(n1,n2) # nodes, paths in x,y,z
            vlength = np.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)
            direction = abs(vector / vlength)
            directions.append([direction, n1, n2]) # direction and nodes
#             print(direction)
    d = np.asarray(directions) # n x 3 (direction,n1,n2) x 3 (xyz)
    ds = sorted(d[:,0,2], reverse = True) # highest z direction first
    for i in range(nstruts):
        indice = np.where(d[:,0,2]==ds[i])[0][0] # [0][0] to get int in array in tuple
        n1 = tuple(d[indice,1,:])
        n2 = tuple(d[indice,2,:])
        add_nodes_edge_to_newmodel(model_struts,model,n1,n2)  
        add_nodes_edge_to_newmodel(model_h_s,model,n1,n2)  
    
    model_R1R2 = model_noHooks.copy()
    model_R1R2.remove_edges_from(model_struts.edges())
#     print('************')
    
    return model_struts, model_hooks, model_R1R2, model_h_s, model_noHooks


def get_model_rings(model_R1R2):
    """Get top ring and 2nd ring from model containing two sepatate rings.
    First run _get_model_struts
    """
    import networkx as nx
    import numpy as np

    model_R2 = model_R1R2.copy()
    model_R1 = model_R1R2.copy()
    # struts must be removed
    clusters = list(nx.connected_components(model_R1R2))
    assert len(clusters) == 2  # 2 rings
    c1, c2 = np.asarray(clusters[0]), np.asarray(clusters[1])
    c1_z, c2_z = c1[:,2], c2[:,2]  # x,y,z
    if c1_z.mean() < c2_z.mean(): # then c1/cluster[0] is topring; daspect = 1, 1,-1
        model_R2.remove_nodes_from(clusters[0])
        model_R1.remove_nodes_from(clusters[1])
    else:
        model_R2.remove_nodes_from(clusters[1])
        model_R1.remove_nodes_from(clusters[0])
        
    return model_R1, model_R2    
