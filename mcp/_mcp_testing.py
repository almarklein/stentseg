import sys, os, time

import numpy as np
import scipy as sp, scipy.ndimage
import visvis as vv
vv.use('qt4')

from points import Point, Pointset, Aarray
import diffgeo
import ssdf

## Test MCP on 2D data again
import mcp
reload(mcp)

im = s.ims[0]

#map = m.CalculateMap(costs, (10,10))

pp1 = getStentSurePositions(im,800)
testp = pp1[20]#im.index_to_point(96,70)
pp1.remove(testp)

la = mcp.McpSimple(np.exp(im/200), testp)
map = mcp.minimumCostPath(la)
map[map==np.inf] = 1

f = vv.figure(101); f.Clear()
t = vv.imshow(la.time,clim=(0,0.5))
#l = vv.plot(pp2,ls='',ms='.')



   
## Test MPC on 2D data
im = s.ims[0]

costs = Aarray(np.exp(-im/200), sampling=im.sampling)
#map = m.CalculateMap(costs, (10,10))

pp1 = getStentSurePositions(im,800)
testp = pp1[20]#im.index_to_point(96,70)
pp1.remove(testp)
map = m.CalculateMap(costs, testp,pp1)
map = Aarray(map, sampling=im.sampling)
pp2 = m.CalculatePath()

f = vv.figure(101); f.Clear()
t = vv.imshow(map,clim=(0,0.1))
l = vv.plot(pp2,ls='',ms='.')

    

## Test our method to find stent-sure-voxels

# load vol
patnr = 1
try:
    db = ssdf.load('d:/almar/projects/_p/onderzoek/vol%02.0i.ssdf' % patnr)
except IOError:
    db = ssdf.load('C:/projects/PYTHON/onderzoek/vol%02.0i.ssdf' % patnr)

# make anisotropic array of volume
vol = Aarray(db.vol128, sampling=(1,0.5,0.5), dtype=np.int16)
pp = getStentSurePositions(vol,1000)

# visualize
f = vv.figure(101); f.Clear()

vv.volshow(vol)
vv.gca().daspect = 1,1,-1
t=vv.plot(pp,ls='')
t.alpha = 0.8

##
if 0:
## view other stent
    import ct
    import visvis as vv
    volo = ct.load('d:/almar/dicomdata/pat15/4d/phase7')
    vol = volo[160:-100,100:-100,50:-50]
    vv.figure(102); vv.clf(); vv.volshow(vol)

## Test heap
    import mcpx
    a = [-1 for i in range(50)]
    h = mcpx.BinaryHeapWithCrossRef(a)
    h.Push(2,1)
    h.Push(99,3)
    h.Push(5,4)
    h.Push(8,5)
    h.Push(3,0)
    h.Push(77,6)
    h.Push(11,7)

## Test MCP speed

# ===== MAZE.gif
# DJ does this in his C++ (mexfile) implementation in 0.72 seconds
# 3.4 seconds required here
# 2.1 seconds after I found the sqrt(2) and the alwyas-Push() bugs :)
# 1.1 seconds after more optimizations
# 0.68 seconds using push_if_lower_fast() - and finding the np.inf issue
# 0.46 seconds using push_fast() and a frozen array
# 0.31 seconds when using the frozen array for edges too
# 0.23 seconds when not using the overloadable methods
#
# MCPTEST.png
# DJ does this in his C++ (mexfile) implementation in 0.99 seconds
# 1.05 seconds
# 1.16 seconds when also visiting frozen voxels
# 0.85 seconds when not using the overloadable methods


import visvis as vv
from points import Point, Pointset, Aarray
import numpy as np
import mcp, mcpx, ssdf, time
reload(mcp)

# load image
basepath = 'c:/almar/data/images/'
im = vv.imread(basepath+'maze.gif')
#im = vv.imread(basepath+'pathInstance.png')
#im = vv.imread(basepath+'mcptest.png')
speed = im.astype(np.float32)
   
# create seedpoint
pp = Pointset(2) 
# pp.append(500,500)
pp.append(10,10)

# create mcp object and run
m = mcp.McpSimple(1/speed, [pp[0]])
#m = mcpx.McpSimple2(speed, [pp[0]])
#m = mcpx.McpSimple2(speed, [pp[0]])
t0 = time.time()
#mcp.minimumCostPathFast(m,20000000)
m.EvolveFront()
t1 = time.time()

# visualize
print t1-t0,'secs'
f = vv.figure(100); f.Clear()
f.position = -750, 360, 560, 720
vv.imshow(m.cumCosts, clim=(0,30000)) # 30000