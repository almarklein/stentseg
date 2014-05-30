""" Script to do the segmentation and store the result.

"""

import os

import numpy as np
import visvis as vv
from visvis import ssdf

from stentseg.utils import PointSet
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.stentdirect import StentDirect, getDefaultParams

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_ssdf',)

# Select dataset to register
ptcode = 'LSPEAS_003'
ctcode = 'discharge'
cropname = 'ring'


## Perform segmentation

# Load volumes
s = loadvol(basedir, ptcode, ctcode, cropname, 'avg5090')
vol = s.vol

# Initialize segmentation parameters
stentType = 'anacondaRing'

p = getDefaultParams(stentType)
p.graph_weakThreshold = 100             # step 3, stentgraph.prune_very_weak
p.graph_expectedNumberOfEdges = 2       # step 3, stentgraph.prune_weak
#p.graph_trimLength =  3                # step 3, stentgraph.prune_tails
p.graph_strongThreshold = 1200          # step 3, stentgraph.prune_weak and stentgraph.prune_redundant
p.seed_threshold = 700                  # step 1
#p.graph_minimumClusterSize = 8         # step 3, stentgraph.prune_clusters
p.mcp_speedFactor = 140                 # step 2, speed image (delta), costToCtValue
p.mcp_maxCoverageFronts = 0.015         # step 2, base.py; replaces mcp_evolutionThreshold
#p.graph_min_strutlength = 7            # step 3, stentgraph.prune_weak and stentgraph.prune_redundant
#p.graph_max_strutlength = 12           # step 3, stentgraph.prune_weak and stentgraph.prune_redundant

# Instantiate stentdirect segmenter object
sd = StentDirect(vol, p)

# Perform the three steps of stentDirect
sd.Step1()
sd.Step2()
sd.Step3()

# Create a mesh object for visualization (argument is strut tickness)
bm = create_mesh(sd._nodes3, 0.6) # new

# Get graph model
model = sd._nodes3


## Visualize 

fig = vv.figure(3); vv.clf()
fig.position = 0, 22, 1366, 706

# Show cleaned up
a2 = vv.subplot(121)
a2.daspect = 1,-1,-1
t = vv.volshow(vol)
t.clim = 0, 2500
sd._nodes3.Draw(mc='g', mw = 6, lc='b')
vv.xlabel('x'); vv.ylabel('y'); vv.zlabel('z')

# Show the mesh
a3 = vv.subplot(122)
a3.daspect = 1,-1,-1
m = vv.mesh(bm)
m.faceColor = 'g'
vv.xlabel('x'); vv.ylabel('y'); vv.zlabel('z')

# Use same camera
a2.camera = a3.camera


## Store segmentation to disk

# Build struct
s2 = vv.ssdf.new()
# We do not need origin and croprange, but keep them for reference
s2.sampling = s.sampling
s2.origin = s.origin
s2.stenttype = s.stenttype
s2.croprange = s.croprange
# Store model
s2.model = model.pack()
#s2.mesh = ssdf.new()

# Save
filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'model')
ssdf.save(os.path.join(basedir, ptcode, filename), s2)


## Make model dynamic

import pirt

# Load deforms
s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
deforms = [s['deform%i'%(i*10)] for i in range(10)]
deforms = [pirt.DeformationFieldBackward(*fields) for fields in deforms]

# Load model
s = loadmodel(basedir, ptcode, ctcode, cropname)
model = s.model

# Combine ...
incorporate_motion(model, deforms, s.origin)

# Save back
filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'model')
s.model = model.pack()
ssdf.save(os.path.join(basedir, ptcode, filename), s)
