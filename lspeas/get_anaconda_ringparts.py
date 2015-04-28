""" Module to obtain the different parts of the Anaconda dual ring

Hooks, struts, 2nd ring, top ring
""" 


def _get_model_hooks(model):
    """
    
    """
    import os
    import visvis as vv
    import numpy as np
    from stentseg.utils import PointSet
    from stentseg.stentdirect import stentgraph
    from stentseg.stentdirect.stentgraph import create_mesh
    
    # initialize 
    model_noHooks = model.copy()
    model_hooks = stentgraph.StentGraph() # graph for hooks
    hooknodes = list() # remember nodes that belong to hooks 
    for n in sorted(model.nodes() ):
        if model.degree(n) == 1:
            neighbour = list(model.edge[n].keys())
            neighbour = neighbour[0]
            c = model.edge[n][neighbour]['cost']
            ct = model.edge[n][neighbour]['ctvalue']
            p = model.edge[n][neighbour]['path']
            pdeforms = model.edge[n][neighbour]['pathdeforms']
            model_hooks.add_node(n, deforms = model.node[n]['deforms'])
            model_hooks.add_node(neighbour, deforms = model.node[neighbour]['deforms'])
            model_hooks.add_edge(n, neighbour, cost = c, ctvalue = ct, path = p, pathdeforms = pdeforms)
            hooknodes.append(neighbour)
            model_noHooks.remove_node(n) # this also removes the connecting edge
    # If any, pop remaining degree 2 nodes
    stentgraph.pop_nodes(model_noHooks)
    
    return model_hooks, model_noHooks
    

def _get_model_struts(model):
    """Get struts between top and 2nd ring
    Finds triangles and quadrangles formed by the struts
    """
    from stentseg.stentdirect.stent_anaconda import _edge_length
    
    # initialize
    model_noHooks_noStruts = model_noHooks.copy()
    model_struts = stentgraph.StentGraph()
    # remove hooks
    models = get_model_hooks(model)
    model_noHooks = models[1]
#todo: simplify: use edge direction -> z high for struts
    # get edges that belong to struts and form triangles
    topnodes = list() # remember nodes that belong to top ring 
    for n in sorted(model_noHooks.nodes() ):
        if model_noHooks.degree(n) == 4: # triangle node top ring
            nn = list(model_noHooks.edge[n].keys())
            nncopy = nn.copy()
            for node1 in nn:
                nncopy.remove(node1)  # check combination once
                for node2 in nncopy:
                    if model_noHooks.has_edge(node1,node2):
                        topnodes.append(n)
                        model_struts.add_node(n, deforms = model_noHooks.node[n]['deforms'])
                        for node in (node1,node2):
                            c = model_noHooks.edge[n][node]['cost']
                            ct = model_noHooks.edge[n][node]['ctvalue']
                            p = model_noHooks.edge[n][node]['path']
                            pdeforms = model_noHooks.edge[n][node]['pathdeforms']
                            model_struts.add_node(node, deforms = model_noHooks.node[node]['deforms'])
                            model_struts.add_edge(n, node, cost = c, ctvalue = ct, path = p, pathdeforms = pdeforms)
    # get edges that belong to struts and form quadrangles
    lengths = list()
    for (n1, n2) in sorted(model_noHooks.edges() ):
        # get edgelength
        lengths.append(_edge_length(model_noHooks, n1, n2))
    lengths.sort(reverse = True) # longest first
    shortestringedge = lengths[7] # anacondaring contains 8 long ring edges
    for (n1, n2) in model_noHooks.edges():
        # get neighbours
        nn1 = list(model_noHooks.edge[n1].keys())
        nn2 = list(model_noHooks.edge[n2].keys())
        for node1 in nn1:
            if node1 == n2:
                continue  # go to next iteration, do not consider n1-n2
            for node2 in nn2:
                if node2 == n1:
                    continue  # go to next iteration, do not consider n1-n2
                if model_noHooks.has_edge(node1,node2):
                    # assume nodes in top ring (or 2nd) are closer than strut nodes between rings
                    hookedges = [(n1,n2), (n1, node1), (n2, node2), (node1, node2)]
                    e_length = list()
                    e_length.append(_edge_length(model_noHooks, n1, n2))
                    e_length.append(_edge_length(model_noHooks, n1, node1))
                    e_length.append(_edge_length(model_noHooks, n2, node2))   
                    e_length.append(_edge_length(model_noHooks, node1, node2))
                    e_lengthMinIndex = e_length.index(min(e_length))
                    ringnodes = hookedges[e_lengthMinIndex] # hooknodes with edge at ring
                    hookedges.remove((ringnodes)) # from array
                    for node in ringnodes:
                        for nodepair in hookedges:
                            if node in nodepair:
                                # add nodes and edges if they indeed belong to struts
                                if _edge_length(model_noHooks, nodepair[0], nodepair[1]) < shortestringedge:
                                    model_struts.add_node(nodepair[0],deforms = model_noHooks.node[nodepair[0]]['deforms'])
                                    model_struts.add_node(nodepair[1],deforms = model_noHooks.node[nodepair[1]]['deforms'])
                                    c = model_noHooks.edge[nodepair[0]][nodepair[1]]['cost']
                                    ct = model_noHooks.edge[nodepair[0]][nodepair[1]]['ctvalue']
                                    p = model_noHooks.edge[nodepair[0]][nodepair[1]]['path']
                                    pdeforms = model_noHooks.edge[nodepair[0]][nodepair[1]]['pathdeforms']
                                    model_struts.add_edge(nodepair[0], nodepair[1], cost = c, ctvalue = ct, path = p, pathdeforms= pdeforms)
    
    # remove strut edges from model
    model_noHooks_noStruts.remove_edges_from(model_struts.edges())
    
    return model_struts, model_noHooks_noStruts


def get_model_rings(model):
    """Get top ring and 2nd ring
    """
    import networkx as nx
    
    model_2nd = model.copy()
    model_top = model.copy()
    models = _get_model_struts(model)
    model_noHooks_noStruts = models[1]
    
    # get clusters of nodes
    clusters = list(nx.connected_components(model_noHooks_noStruts))
    assert len(clusters) == 2  # 2 rings
    
    c1 = np.asarray(clusters[0])
    c1_z = c1[:,2] # x,y,z
    c2 = np.asarray(clusters[1])
    c2_z = c2[:,2] # x,y,z
    if c1_z.mean() < c2_z.mean(): # then c1 -> cluster[0] is topring; daspect = 1, 1,-1
        model_2nd.remove_nodes_from(clusters[0])
        model_top.remove_nodes_from(clusters[1])
    else:
        model_2nd.remove_nodes_from(clusters[1])
        model_top.remove_nodes_from(clusters[0])
    return model_top, model_2nd

