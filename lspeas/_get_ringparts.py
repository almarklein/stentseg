""" Module to obtain the different parts of the Anaconda rings

Hooks, struts, 2nd ring, top ring
""" 

import os
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
import numpy as np
from stentseg.utils import PointSet
from stentseg.stentdirect import stentgraph


# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_ssdf',)

# Select dataset to register
ptcode = 'LSPEAS_001'
ctcode = '1month'
cropname = 'ring'

# Load the stent model and mesh
s2 = loadmodel(basedir, ptcode, ctcode, cropname)
model = s2.model
modeloriginal = model.copy()

## Visualize 
f = vv.figure(1); vv.clf()
f.position = 0.00, 22.00,  1366.00, 706.00
a = vv.gca()
a.axis.axisColor = 1,1,1
a.axis.visible = False
a.bgcolor = 0,0,0
a.daspect = 1, -1, -1
model.Draw(mc='b', mw = 10, lc='g')
#model_hooks.Draw(mc='r', mw = 10, lc='r')
#model_struts.Draw(mc='m', mw = 10, lc='m')
#model_top.Draw(mc='y', mw = 10, lc='y')
#model_2nd.Draw(mc='c', mw = 10, lc='c')

vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
viewringcrop = {'azimuth': -68.20258814427977,
 'daspect': (1.0, -1.0, -1.0),
 'elevation': 19.282744282744293,
 'fov': 0.0,
 'loc': (109.39546280132387, 106.85695511791796, 59.00197909677347),
 'roll': 0.0,
 'zoom': 0.030437580943452218}
a.SetView(viewringcrop)

## Get hooks in vessel wall

# Create graph for hooks
model_hooks = stentgraph.StentGraph()

remove = True
for n in model.nodes():
    if model.degree(n) == 1:
        neighbour = list(model.edge[n].keys())
        neighbour = neighbour[0]
        assert model.degree(neighbour) == 4
        c = model.edge[n][neighbour]['cost']
        ct = model.edge[n][neighbour]['ctvalue']
        p = model.edge[n][neighbour]['path']
        model_hooks.add_node(n, deforms = model.node[n]['deforms'])
        model_hooks.add_node(neighbour, deforms = model.node[neighbour]['deforms'])
        model_hooks.add_edge(n, neighbour, cost = c, ctvalue = ct, path = p)
        if remove == True:
            model.remove_node(n)

# Visualize
model_hooks.Draw(mc='r', mw = 10, lc='r')

## Get struts between top and 2nd ring
from stentseg.stentdirect.stentgraph_anacondaRing import _edge_length

model_struts = stentgraph.StentGraph()

#stentgraph.pop_nodes(model) 
#todo: haalt node niet weg (bij 002 discharge en 003 discharge)

remove = True

# get edges that belong to struts and form triangles
# hooks in vessel wall must be removed
topnodes = list() # remember nodes that belong to top ring 
for n in model.nodes():
    if model.degree(n) == 4:
        nn = list(model.edge[n].keys())
        nncopy = nn.copy()
        for node1 in nn:
            nncopy.remove(node1)  # check combination once
            for node2 in nncopy:
                if model.has_edge(node1,node2):
                    topnodes.append(n)
                    model_struts.add_node(n, deforms = model.node[n]['deforms'])
                    for node in (node1,node2):
                        c = model.edge[n][node]['cost']
                        ct = model.edge[n][node]['ctvalue']
                        p = model.edge[n][node]['path']
                        model_struts.add_node(node, deforms = model.node[node]['deforms'])
                        model_struts.add_edge(n, node, cost = c, ctvalue = ct, path = p)
                            
# get edges that belong to struts and form quadrangles
edges = model.edges()
lengths = list()
for (n1, n2) in edges:
    # get edgelength
    lengths.append(_edge_length(model, n1, n2))
lengths.sort(reverse = True)
shortestringedge = lengths[7] # anacondaring contains 8
for (n1, n2) in edges:
    # get neighbours
    nn1 = list(model.edge[n1].keys())
    nn2 = list(model.edge[n2].keys())
    for node1 in nn1:
        if node1 == n2:
            continue  # go to next iteration, do not consider n1-n2
        for node2 in nn2:
            if node2 == n1:
                continue  # go to next iteration, do not consider n1-n2
            if model.has_edge(node1,node2):
                # assume nodes in top ring (or 2nd) are closer than strut nodes
                hookedges = [(n1,n2), (n1, node1), (n2, node2), (node1, node2)]
                e_length = list()
                e_length.append(_edge_length(model, n1, n2))
                e_length.append(_edge_length(model, n1, node1))
                e_length.append(_edge_length(model, n2, node2))   
                e_length.append(_edge_length(model, node1, node2))
                e_lengthMinindex = e_length.index(min(e_length))
                ringnodes = hookedges[e_lengthMinindex]
                hookedges.remove((ringnodes))
                for node in ringnodes:
                    for nodepair in hookedges:
                        if node in nodepair:
                            # add nodes and edges if they indeed belong to struts
                            if _edge_length(model, nodepair[0], nodepair[1]) < shortestringedge:
                                model_struts.add_node(nodepair[0],deforms = model.node[nodepair[0]]['deforms'])
                                model_struts.add_node(nodepair[1],deforms = model.node[nodepair[1]]['deforms'])
                                c = model.edge[nodepair[0]][nodepair[1]]['cost']
                                ct = model.edge[nodepair[0]][nodepair[1]]['ctvalue']
                                p = model.edge[nodepair[0]][nodepair[1]]['path']
                                model_struts.add_edge(nodepair[0], nodepair[1], cost = c, ctvalue = ct, path = p)

# remove strut edges from model
if remove == True:
    model.remove_edges_from(model_struts.edges())

# Visualize
model_struts.Draw(mc='m', mw = 10, lc='m')

## Get top ring and 2nd ring
import networkx as nx

model_2nd = model.copy()
model_top = model.copy()
# struts must be removed
# must be at least 1 triangle of struts
clusters = list(nx.connected_components(model))
assert len(clusters) == 2  # 2 rings
for cluster in clusters:
    top = False
    for topnode in topnodes:
        if topnode in cluster:
            top = True # cluster is top ring
    if top == True:
        model_2nd.remove_nodes_from(cluster)
    else:
        model_top.remove_nodes_from(cluster)

# or use z axis for distinction

# Visualize
view = a.GetView()
a.Clear()
a.axis.axisColor = 1,1,1
a.axis.visible = False
a.bgcolor = 0,0,0
a.daspect = 1, -1, -1
a.SetView(view)
model.Draw(mc='b', mw = 10, lc='g')
model_top.Draw(mc='y', mw = 10, lc='y')
model_2nd.Draw(mc='c', mw = 10, lc='c')
