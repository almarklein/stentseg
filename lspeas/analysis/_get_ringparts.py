""" Module to obtain the different parts of the Anaconda rings
For TESTING with visalization
Hooks, struts, 2nd ring, top ring
""" 

import os
import visvis as vv
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
import numpy as np
from stentseg.utils import PointSet
from stentseg.stentdirect import stentgraph
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.motion.vis import show_ctvolume


# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf', r'F:\LSPEAS_ssdf_backup')

# Select dataset to register
ptcode = 'LSPEAS_002'
ctcode = 'discharge'
cropname = 'ring'
modelname = 'modelavgreg'

# Load the stent model and mesh
s2 = loadmodel(basedir, ptcode, ctcode, cropname, modelname)
model = s2.model
model2 = model.copy()

# Load static CT image to add as reference
s = loadvol(basedir, ptcode, ctcode, cropname, 'avgreg')
vol = s.vol

## Visualize 
f = vv.figure(2); vv.clf()
f.position = 968.00, 30.00,  944.00, 1002.00
a = vv.gca()
a.axis.axisColor = 1,1,1
a.axis.visible = False
a.bgcolor = 0,0,0
a.daspect = 1, 1, -1
t = vv.volshow(vol, clim=(0, 2500), renderStyle='mip')
model.Draw(mc='b', mw = 10, lc='g')
#model_hooks.Draw(mc='r', mw = 10, lc='r')
#model_struts.Draw(mc='m', mw = 10, lc='m')
#model_top.Draw(mc='y', mw = 10, lc='y')
#model_2nd.Draw(mc='c', mw = 10, lc='c')

vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
# viewringcrop = 
# a.SetView(viewringcrop)

## Get hooks in vessel wall

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


# Create graph for hooks
model_hooks = stentgraph.StentGraph()

remove = True
hooknodes = list() # remember nodes that belong to hooks 
for n in model.nodes():
    if model.degree(n) == 1:
        endnode = n
        neighbour = list(model.edge[n].keys())
        neighbour = neighbour[0]
        add_nodes_edge_to_newmodel(model_hooks,model,n,neighbour)
        hooknodes.append(neighbour)
        if remove == True:
            model.remove_node(n)

# Pop remaining degree 2 nodes
stentgraph.pop_nodes(model) 

# Visualize
model_hooks.Draw(mc='r', mw = 10, lc='r')

# arrayn1 = np.asarray(n1)
# arrayn2 = np.asarray(n2)
# vector = np.subtract(arrayn1,arrayn2) # nodes, paths in x,y,z
# mag = np.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)
# direction_x = vector[0] / mag
# direction_y = vector[1] / mag
# direction_z = vector[2] / mag


## Get Struts using direction

# initialize
model_struts = stentgraph.StentGraph()
nstruts = 8
directions = []
for n1, n2 in model.edges():
    e_length = stentgraph._edge_length(model, n1, n2)
    if (5 < e_length < 12):
        vector = np.subtract(n1,n2) # nodes, paths in x,y,z
        vlength = np.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)
        direction = abs(vector / vlength)
        directions.append([direction, n1, n2]) # direction and nodes
        print(direction)
d = np.asarray(directions) # n x 3 (direction,n1,n2) x 3 (xyz)
ds = sorted(d[:,0,2], reverse = True) # highest z direction first
for i in range(nstruts):
    indice = np.where(d[:,0,2]==ds[i])[0][0] # [0][0] to get int in array in tuple
    n1 = tuple(d[indice,1,:])
    n2 = tuple(d[indice,2,:])
    add_nodes_edge_to_newmodel(model_struts,model,n1,n2)  
    add_nodes_edge_to_newmodel(model_hooks,model,n1,n2)  

model_h_s = model_hooks.copy()
model.remove_edges_from(model_struts.edges())

# Visualize
model_struts.Draw(mc='m', mw = 10, lc='m')

## Get struts between top and 2nd ring
# 
# model_struts = stentgraph.StentGraph()
# 
# remove = True
# 
# # get edges that belong to struts and form triangles
# # hooks in vessel wall must be removed
# topnodes = list() # remember nodes that belong to top ring 
# for n in model.nodes():
#     if model.degree(n) == 4:
#         nn = list(model.edge[n].keys())
#         nncopy = nn.copy()
#         for node1 in nn:
#             nncopy.remove(node1)  # check combination once
#             for node2 in nncopy:
#                 if model.has_edge(node1,node2):
#                     topnodes.append(n)
#                     model_struts.add_node(n, deforms = model.node[n]['deforms'])
#                     for node in (node1,node2):
#                         c = model.edge[n][node]['cost']
#                         ct = model.edge[n][node]['ctvalue']
#                         p = model.edge[n][node]['path']
#                         model_struts.add_node(node, deforms = model.node[node]['deforms'])
#                         model_struts.add_edge(n, node, cost = c, ctvalue = ct, path = p)
#                             
# # get edges that belong to struts and form quadrangles
# edges = model.edges()
# lengths = list()
# for (n1, n2) in edges:
#     # get edgelength
#     lengths.append(stentgraph._edge_length(model, n1, n2))
# lengths.sort(reverse = True) # longest first
# shortestringedge = lengths[7] # anacondaring contains 8 long ring edges
# for (n1, n2) in edges:
#     # get neighbours
#     nn1 = list(model.edge[n1].keys())
#     nn2 = list(model.edge[n2].keys())
#     for node1 in nn1:
#         if node1 == n2:
#             continue  # go to next iteration, do not consider n1-n2
#         for node2 in nn2:
#             if node2 == n1:
#                 continue  # go to next iteration, do not consider n1-n2
#             if model.has_edge(node1,node2):
#                 # assume nodes in top ring (or 2nd) are closer than strut nodes between rings
#                 hookedges = [(n1,n2), (n1, node1), (n2, node2), (node1, node2)]
#                 e_length = list()
#                 e_length.append(stentgraph._edge_length(model, n1, n2))
#                 e_length.append(stentgraph._edge_length(model, n1, node1))
#                 e_length.append(stentgraph._edge_length(model, n2, node2))   
#                 e_length.append(stentgraph._edge_length(model, node1, node2))
#                 e_lengthMinindex = e_length.index(min(e_length))
#                 ringnodes = hookedges[e_lengthMinindex] # hooknodes with edge at ring
#                 hookedges.remove((ringnodes)) # from array
#                 for node in ringnodes:
#                     for nodepair in hookedges:
#                         if node in nodepair:
#                             # add nodes and edges if they indeed belong to struts
#                             if stentgraph._edge_length(model, nodepair[0], nodepair[1]) < shortestringedge:
#                                 model_struts.add_node(nodepair[0],deforms = model.node[nodepair[0]]['deforms'])
#                                 model_struts.add_node(nodepair[1],deforms = model.node[nodepair[1]]['deforms'])
#                                 c = model.edge[nodepair[0]][nodepair[1]]['cost']
#                                 ct = model.edge[nodepair[0]][nodepair[1]]['ctvalue']
#                                 p = model.edge[nodepair[0]][nodepair[1]]['path']
#                                 model_struts.add_edge(nodepair[0], nodepair[1], cost = c, ctvalue = ct, path = p)
# 
# # remove strut edges from model
# if remove == True:
#     model.remove_edges_from(model_struts.edges())
# 
# # Visualize
# model_struts.Draw(mc='m', mw = 10, lc='m')

## Get top ring and 2nd ring
import networkx as nx

model_R2 = model.copy()
model_R1 = model.copy()
# struts must be removed
clusters = list(nx.connected_components(model))
assert len(clusters) == 2  # 2 rings
c1, c2 = np.asarray(clusters[0]), np.asarray(clusters[1])
c1_z, c2_z = c1[:,2], c2[:,2]  # x,y,z
if c1_z.mean() < c2_z.mean(): # then c1/cluster[0] is topring; daspect = 1, 1,-1
    model_R2.remove_nodes_from(clusters[0])
    model_R1.remove_nodes_from(clusters[1])
else:
    model_R2.remove_nodes_from(clusters[1])
    model_R1.remove_nodes_from(clusters[0])

# Visualize
a = vv.gca()
# view = a.GetView()
a.Clear()
a1 = vv.subplot(121)
t = vv.volshow(vol, clim=(0, 2500), renderStyle='mip')
# model.Draw(mc='b', mw = 10, lc='g')
model_R1.Draw(mc='y', mw = 10, lc='y') # R1 = yellow
model_R2.Draw(mc='c', mw = 10, lc='c') # R2 = cyan
# model_top.Draw(mc='b', mw = 10, lc='g')
# modelmesh = create_mesh(model_top, 0.6)  # Param is thickness
# m = vv.mesh(modelmesh)
# m.faceColor = 'g'
a1.axis.axisColor = 1,1,1
a1.axis.visible = False
a1.bgcolor = 0,0,0
a1.daspect = 1, 1, -1
a2 = vv.subplot(122)
t = vv.volshow(vol, clim=(0, 2500), renderStyle='mip')
model_h_s.Draw(mc='m', mw = 10, lc='m') # hooks and struts = magenta
a2.axis.axisColor = 1,1,1
a2.axis.visible = False
a2.bgcolor = 0,0,0
a2.daspect = 1, 1, -1
# a2.SetView(view), a2.SetView(view)

