""" Script to analyze distances between segmented rings

Jaimy Simmering Graduation Project Technical Medicine 2018
"""

import os

import math
import numpy as np
from numpy import * 
import visvis as vv
from visvis import ssdf
from visvis import Pointset
import scipy.io
import plotly.plotly as py
import plotly.graph_objs as go
import networkx as nx
import matplotlib.pyplot as plt
# import utils_graphs_pointsets as py

from stentseg.utils import PointSet, _utils_GUI, visualization
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from stentseg.stentdirect import stentgraph, getDefaultParams, initStentDirect
from stentseg.utils.picker import pick3d, get_picked_seed
from stentseg.utils.visualization import DrawModelAxes
from stentseg.utils.visualization import show_ctvolume
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.utils.utils_graphs_pointsets import points_from_edges_in_graph
from stentseg.utils.centerline import points_from_nodes_in_graph
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel, loadmodel_location



# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf',
                     r'E:\LSPEAS_ssdf_backup',r'G:\LSPEAS_ssdf_backup')
targetdir = select_dir(r'C:\Users\Gebruiker\Google Drive\Afstuderen\Rings')
targetdir1 = select_dir(r'C:\Users\Gebruiker\Dropbox\M3 Vaatchirurgie Enschede\Py_analysis', r'C:\Users\Gebruiker\Dropbox\LSPEAS Stages-gedeelde map\M3_20180106_Jaimy\Distances rings 30072018')

basedir1 = select_dir(r'C:\Users\Gebruiker\Google Drive\Afstuderen\Rings',r'C:\Users\Gebruiker\Dropbox\LSPEAS Stages-gedeelde map\M3_20180106_Jaimy\Distances rings 30072018') 
basedir2 = select_dir(r'E:\LSPEAS_ssdf_backup')

# Select dataset to register
ptcode = 'LSPEAS_008'
ctcode = 'discharge'
cropname = 'stent'
n = 26 # number of rings
what = 'avgreg' # avgreg
normalize = True

showAxis = False  # True or False
showVol  = 'ISO'  # MIP or ISO or 2D or None
clim0  = (0,2500)
isoTh = 250

targetdir2 = os.path.join(targetdir1, ptcode)
s = loadvol(basedir2, ptcode, ctcode, cropname, what)

nrs = [1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26]

FIG1 = False


## Load all rings of the limb
for i in nrs:
    ring = 'ringR%s' %(i)
    filename = '%s_%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'model'+what,ring)
    s["ringR" + str(i)] = ssdf.load(os.path.join(basedir1, filename))


## Visualize centerlines rings
s2 = loadmodel_location(targetdir2, ptcode, ctcode, cropname)
pmodel = points_from_nodes_in_graph(s2.model)
if FIG1 == True:
    f = vv.figure(1); vv.clf()
    f.position = 709.00, 30.00,  1203.00, 1008.00
    a1 = vv.subplot(121)
    show_ctvolume(s.vol, None, showVol=showVol, clim=clim0, isoTh=isoTh, removeStent=False)
    label = pick3d(vv.gca(), s.vol)
    a1.axis.visible = showAxis
    a1.daspect = 1,1,-1 
    
    a2 = vv.subplot(122)
    vv.plot(pmodel, ms='.', ls='', alpha=0.2, mw = 2) # stent seed points
    for i in nrs:  
        pp = s["ringR" + str(i)].ringpoints
        vv.plot(pp[0],pp[1],pp[2], ms='.', ls='', mw=3, mc='c')
    a2.axis.visible = showAxis
    a2.daspect = 1,1,-1 

# a1.camera = a2.camera

## Distances
# Obtain minimal distances from all points on ring to a point on ring below
#  ONE RING PAIR
#     
# prox_ring = s.ringR1.ringpoints.shape[1]
# dist_ring = s.ringR2.ringpoints.shape[1]
# dist = [1] * prox_ring * dist_ring
# for k in range(0,prox_ring):
#     for j in range(dist_ring):
#         # for i in range(0,j+k):
#             # j = int(i/dist_ring)
#             # k = int(i/prox_ring)
#         i= j+k+(k*(dist_ring-1))
#         dist[i] = math.sqrt((s.ringR1.ringpoints.T[k,0]-s.ringR2.ringpoints.T[j,0])**2 + (s.ringR1.ringpoints.T[k,1]-s.ringR2.ringpoints.T[j,1])**2 +(s.ringR1.ringpoints.T[k,2]-s.ringR2.ringpoints.T[j,2])**2)
#         
# dist = reshape(dist,(prox_ring,dist_ring)) 
# 
# 
# min_dists = dist.min(1)
# min_dist1 = min_dists.min()
# max_dist1 = min_dists.max() 
# mean_dist1 = min_dists.mean()   
# loc_min1 = min_dists.argmin()
# loc_max1 = min_dists.argmax()
# 
# print('minimal distance = ', min_dist1)
# print('maximal distance = ', max_dist1)
# print('mean distance = ', mean_dist1)
    
## ONE LIMB: 
# define variables
prox_ring = dist_ring = [1] * n
dist = {}
min_dists = {}
min_dist1 = {}
max_dist1 = {}
mean_dist1 = {}
loc_min1 = {}
loc_max1 ={}
min_dists2 = {}
loc_min2 = {}
loc_max2 ={}
line_min1x = {}
line_min1y = {}
line_min1z = {}
line_max1x = {}
line_max1y = {}
line_max1z = {}

# Obtain minimal distances from all points on a ring to a point on ring below
for a in nrs[:n:2]:
    prox_ring = s["ringR" + str(a)].ringpoints.shape[1]
    dist_ring = s["ringR" + str(a+1)].ringpoints.shape[1]
    distance =  [1] * prox_ring * dist_ring
    for k in range(0,prox_ring):
        for j in range(dist_ring):
            # for i in range(0,j+k):
                # j = int(i/dist_ring)
                # k = int(i/prox_ring)
            i= j+k+(k*(dist_ring-1))
            distance[i] = math.sqrt((s["ringR" + str(a)].ringpoints.T[k,0]-s["ringR" + str(a+1)].ringpoints.T[j,0])**2 + (s["ringR" + str(a)].ringpoints.T[k,1]-s["ringR" + str(a+1)].ringpoints.T[j,1])**2 +(s["ringR" + str(a)].ringpoints.T[k,2]-s["ringR" + str(a+1)].ringpoints.T[j,2])**2)
            
    dist["ringR" + str(a)] = reshape(distance,(prox_ring,dist_ring)) 
   
    min_dists["ringR" + str(a)] = dist["ringR" + str(a)].min(1)
    min_dist1["ringR" + str(a)] = min_dists["ringR" + str(a)].min()
    max_dist1["ringR" + str(a)] = min_dists["ringR" + str(a)].max() 
    mean_dist1["ringR" + str(a)] = min_dists["ringR" + str(a)].mean()
    loc_min1["ringR" + str(a)] = min_dists["ringR" + str(a)].argmin()
    loc_max1["ringR" + str(a)] = min_dists["ringR" + str(a)].argmax()
    
    min_dists2["ringR" + str(a+1)] = dist["ringR" + str(a)].min(0)
    loc_min2["ringR" + str(a+1)] = min_dists2["ringR" + str(a+1)].argmin()
    loc_max2["ringR" + str(a+1)] = min_dists2["ringR" + str(a+1)].argmax()
    
    # print('minimal distance', "ringR" + str(a), ' = ', min_dist1["ringR" + str(a)])
    # print('maximal distance', "ringR" + str(a), ' = ', max_dist1["ringR" + str(a)])
    # print('mean distance', "ringR" + str(a), ' = ', mean_dist1["ringR" + str(a)])    
    
    # define lines of minimal and maximal distances
    line_min1x["ringR" + str(a)] = [s["ringR" + str(a)].ringpoints.T[loc_min1["ringR" + str(a)],0], s["ringR" + str(a+1)].ringpoints.T[loc_min2["ringR" + str(a+1)],0]]
    line_min1y["ringR" + str(a)] = [s["ringR" + str(a)].ringpoints.T[loc_min1["ringR" + str(a)],1], s["ringR" + str(a+1)].ringpoints.T[loc_min2["ringR" + str(a+1)],1]]
    line_min1z["ringR" + str(a)] = [s["ringR" + str(a)].ringpoints.T[loc_min1["ringR" + str(a)],2], s["ringR" + str(a+1)].ringpoints.T[loc_min2["ringR" + str(a+1)],2]]
    
    line_max1x["ringR" + str(a)] = [s["ringR" + str(a)].ringpoints.T[loc_max1["ringR" + str(a)],0], s["ringR" + str(a+1)].ringpoints.T[loc_max2["ringR" + str(a+1)],0]]
    line_max1y["ringR" + str(a)] = [s["ringR" + str(a)].ringpoints.T[loc_max1["ringR" + str(a)],1], s["ringR" + str(a+1)].ringpoints.T[loc_max2["ringR" + str(a+1)],1]]
    line_max1z["ringR" + str(a)] = [s["ringR" + str(a)].ringpoints.T[loc_max1["ringR" + str(a)],2], s["ringR" + str(a+1)].ringpoints.T[loc_max2["ringR" + str(a+1)],2]]

loc_overall_min = min(min_dist1, key=min_dist1.get)
overall_min = min_dist1[loc_overall_min]
print('minimal distance limb =', overall_min, 'below',loc_overall_min)

loc_overall_max = max(max_dist1, key=max_dist1.get)
overall_max = max_dist1[loc_overall_max]
print('maximal distance limb =', overall_max, 'below',loc_overall_max)


# line_min2["ringR" + str(a)] = s["ringR" + str(a+1)].ringpoints.T[loc_min2["ringR" + str(a+1)],:]]
        
## Visualize distances
f2 = vv.figure(2); vv.clf()
f2.position = 709.00, 30.00,  1203.00, 1008.00
a3 = vv.subplot(111)
vv.plot(pmodel, ms='.', ls='', alpha=0.2, mw = 2) # stent seed points
for i in nrs:  
    pp = s["ringR" + str(i)].ringpoints
    vv.plot(pp[0],pp[1],pp[2], ms='.', ls='', mw=3, mc='c')
    for a in nrs[:n:2]:
        vv.plot(line_min1x["ringR" + str(a)],line_min1y["ringR" + str(a)],line_min1z["ringR" + str(a)], ls='-', lw=2, lc='r')
        vv.plot(line_max1x["ringR" + str(a)],line_max1y["ringR" + str(a)],line_max1z["ringR" + str(a)], ls='-', lw=2, lc='g')
a3.axis.visible = showAxis
a3.daspect = 1,1,-1         
    
    
    