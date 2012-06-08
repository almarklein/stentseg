""" SCRIPT
This script visualizes the mcp method on a number of images. 
The process is recorder such that it can be stored as an swf movie.
"""

import visvis as vv
vv.use('PySide')
from visvis.pypoints import Point, Pointset, Aarray
import numpy as np
from pyzolib import ssdf

# Import mcp by going down one dir from this script
import os, sys
os.chdir('..')
sys.path.insert(0, '')
from mcp import McpVisual

im2show  = None

# load image
impath = 'D:/almar/projects/ims/' 
imagenr = 4
if imagenr == 0:
    costs = np.ones((30,60),dtype=np.float32)
    costs = Aarray( costs ,(1.0,0.5))
    costs[0,0]=0; costs[-1,0]=0; costs[0,-1]=0; costs[-1,-1]=0
    pp = Pointset(2); pp.append(costs.get_end().x/2,costs.get_end().y/2)
    stepsize = 5
elif imagenr == 1:
    im = vv.imread(impath+'maze.gif').astype(np.float32)
    pp = Pointset(2); pp.append(10,10)
    im = im[0:100,0:100]   
    im2show = im 
    costs = 1/im
    stepsize = 10
elif imagenr == 2:
    im = vv.imread(impath+'pathInstance.png').astype(np.float32)*3000/255.0
    im = Aarray( im[80:100,64:80,0] ,(1.0,0.5))
    costs = 1/2**((im)/100.0)
    costs[costs<0]=0.1
    pp = Pointset(2); pp.append(4,6)
    stepsize = 0.01
    im2show = im
elif imagenr == 3:
    im = vv.imread(impath+'pathInstance.png').astype(np.float32)*3000/255.0
    im = Aarray( im[6:26,25:65,0] ,(1.0,0.5))
    costs = 1/2**((im)/100.0)
    pp = Pointset(2); pp.append(10,9)
    stepsize = 1
    im2show = im
elif imagenr == 4:
    s = 30
    costs = np.ones((s,s),dtype=np.float32)
    x0, y0 = s/2, 5
    for y in range(costs.shape[0]):
        yy = abs(y-s/2)
        for x in range(costs.shape[1]):                        
            costs[y,x] = ( (x-x0)**2 + (yy-y0)**2 )**0.7
            #costs[y,x] = 1 - np.cos((x-x0)/3.0)* np.cos((y-y0)/3.0)
#     im2show = costs.copy()
    costs[costs==0]=1
    costs=1/costs    
    pp = Pointset(2); pp.append(15,15)
    stepsize = 10    

# prepare figure
f = vv.figure(100); vv.clf()
f.position = -1000.00, 550.00,  600.00, 300.00

# go!
m = McpVisual(costs, [pp[0]])
rec = vv.record(f)
m.EvolveFront(stepsize, im2show)
# rec.Stop()
# rec.ExportToSwf(r'D:\almar\projects\movs\mcp-1.swf')

##
def sample(event):
    x,y = event.x2d * m.time.sampling[1], event.y2d * m.time.sampling[0]
    print m.time[int(y+0.5),int(x+0.5)]
a = m._t2.GetAxes()
a.eventMouseDown.Bind(sample)