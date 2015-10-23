""" Script to do the segmentation and store the result.

Do not run file but execute cells (overwrites!)
"""

import os

import numpy as np
import visvis as vv
from visvis import ssdf

from stentseg.utils import PointSet
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.stentdirect import StentDirect, getDefaultParams, AnacondaDirect, EndurantDirect
from stentseg.utils.picker import pick3d

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf',
                     r'F:\LSPEAS_ssdf_backup',r'G:\LSPEAS_ssdf_backup')

# Select dataset to register
ptcode = 'FANTOOM_20150625'
ctcode = '5_Prof6_Water_Thorax_Pos2'
cropname = 'ring'
what = 'avgreg'

# Load volumes
s = loadvol(basedir, ptcode, ctcode, cropname, what)
vol = s.vol


## Initialize segmentation parameters
stentType = 'endurant'  # 'anacondaRing' runs modified pruning algorithm in Step3

p = getDefaultParams(stentType)
p.seed_threshold = 600                 # step 1
p.mcp_speedFactor = 110                 # step 2, speed image (delta), costToCtValue
p.mcp_maxCoverageFronts = 0.006         # step 2, base.py; replaces mcp_evolutionThreshold
p.graph_weakThreshold = 500            # step 3, stentgraph.prune_very_weak
p.graph_expectedNumberOfEdges = 3       # step 3, stentgraph.prune_weak
p.graph_trimLength =  2                 # step 3, stentgraph.prune_tails
p.graph_minimumClusterSize = 10         # step 3, stentgraph.prune_clusters
p.graph_strongThreshold = 5000          # step 3, stentgraph.prune_weak and stentgraph.prune_redundant
# p.graph_min_strutlength = 6             # step 3, stent_anaconda prune_redundant
# p.graph_max_strutlength = 12            # step 3, stent_anaconda prune_redundant
p.graph_angleVector = 5                 # step 3, corner detect
p.graph_angleTh = 40                    # step 3, corner detect


## Perform segmentation
cleanNodes = True  # True when NOT using GUI

# Instantiate stentdirect segmenter object
if stentType == 'anacondaRing':
        sd = AnacondaDirect(vol, p) # inherit _Step3_iter from AnacondaDirect class
        #runtime warning using anacondadirect due to mesh creation, ignore
elif stentType == 'endurant':
        sd = EndurantDirect(vol, p)
else:
        sd = StentDirect(vol, p)

# Perform the three steps of stentDirect
sd.Step1()
sd.Step2()
sd.Step3(cleanNodes)

# Create a mesh object for visualization (argument is strut tickness)
bm = create_mesh(sd._nodes3, 0.6) # new


# Visualize
fig = vv.figure(3); vv.clf()
fig.position = 0.00, 22.00,  1920.00, 1018.00
clim = (0,2500)
#viewringcrop = 

# Show volume and model as graph
a1 = vv.subplot(131)
t = vv.volshow(vol, clim=clim)
pick3d(vv.gca(), vol)
sd._nodes1.Draw(mc='b', mw = 6)       # draw seeded nodes
# sd._nodes2.Draw(mc='b', lc = 'g')    # draw seeded and MCP connected nodes

# Show volume and cleaned up graph
a2 = vv.subplot(132)
t = vv.volshow(vol, clim=clim)
pick3d(vv.gca(), vol)
sd._nodes2.Draw(mc='b', lc='g')
# sd._nodes3.Draw(mc='b', lc='g')
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')

# Show the mesh
a3 = vv.subplot(133)
a3.daspect = 1,1,-1
t = vv.volshow(vol, clim=clim)
pick3d(vv.gca(), vol)
sd._nodes3.Draw(mc='b', lc='w')
m = vv.mesh(bm)
m.faceColor = 'g'
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')

# Use same camera
a1.camera = a2.camera = a3.camera

#a1.SetView(viewringcrop)

switch = False
a1.axis.visible = switch
a2.axis.visible = switch
a3.axis.visible = switch

## Prevent save when 'run as script'
print('Model not yet saved to disk, run next cells')
1/0

## Store segmentation to disk

# Get graph model
model = sd._nodes3

# Build struct
s2 = vv.ssdf.new()
# We do not need croprange, but keep for reference
s2.sampling = s.sampling
s2.origin = s.origin
s2.stenttype = s.stenttype
s2.croprange = s.croprange
for key in dir(s):
        if key.startswith('meta'):
            suffix = key[4:]
            s2['meta'+suffix] = s['meta'+suffix]
s2.what = what
s2.params = p
s2.stentType = stentType
# Store model
s2.model = model.pack()
#s2.mesh = ssdf.new()

# Save
filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'model'+what)
ssdf.save(os.path.join(basedir, ptcode, filename), s2)
print("model saved to disk.")


## Make model dynamic (and store/overwrite to disk)

import pirt
from stentseg.motion.dynamic import incorporate_motion_nodes, incorporate_motion_edges  

# Load deforms
s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
deforms = [s['deform%i'%(i*10)] for i in range(10)]
deforms = [pirt.DeformationFieldBackward(*fields) for fields in deforms]
paramsreg = s.params

# Load model
s = loadmodel(basedir, ptcode, ctcode, cropname, 'model'+what)
model = s.model

# Combine ...
incorporate_motion_nodes(model, deforms, s.origin)
incorporate_motion_edges(model, deforms, s.origin)

# Save back
filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'model'+what)
s.model = model.pack()
s.paramsreg = paramsreg
ssdf.save(os.path.join(basedir, ptcode, filename), s)
print("dynamic model saved to disk.")