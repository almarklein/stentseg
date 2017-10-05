""" Store segmentation result to ssdf

"""


## Store segmentation to disk

# Get graph model
model = sd._nodes3
seeds = sd._nodes1

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
s2.seeds = seeds.pack()
#s2.mesh = ssdf.new()

# Save
filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'model'+what)
ssdf.save(os.path.join(basedir, ptcode, filename), s2)
print('saved to disk in {} as {}.'.format(basedir, filename) )


## Make model dynamic (and store/overwrite to disk)

import pirt
from stentseg.motion.dynamic import incorporate_motion_nodes, incorporate_motion_edges 
from visvis import ssdf 

# Load deforms
s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
deforms = [s['deform%i'%(i*10)] for i in range(10)]
deforms = [pirt.DeformationFieldBackward(*fields) for fields in deforms]
paramsreg = s.params

# Load model
s = loadmodel(basedir, ptcode, ctcode, cropname, 'model'+what)
model = s.model

# Combine ...
incorporate_motion_nodes(model, deforms, s.origin) # adds deforms PointSets
incorporate_motion_edges(model, deforms, s.origin) # adds deforms PointSets

# Save back
filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'model'+what)
s.model = model.pack()
s.paramsreg = paramsreg
ssdf.save(os.path.join(basedir, ptcode, filename), s)
print('saved dynamic to disk in {} as {}.'.format(basedir, filename) )


# option to make independent on number of phases
# deformkeys = []
# for key in dir(s):
#     if key.startswith('deform'):
#         deformkeys.append(key)
# deforms = [s[key] for key in deformkeys]
# deforms = [pirt.DeformationFieldBackward(*fields) for fields in deforms]

