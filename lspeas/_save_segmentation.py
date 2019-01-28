""" Store segmentation result to ssdf

"""
from lspeas.utils.storesegmentation import save_segmentation, add_segmentation_ssdf

# Get graph model
model = sd._nodes3
seeds = sd._nodes1

## Store segmentation to disk

save_segmentation(basedir, ptcode, ctcode, cropname, seeds, model, s, 
                      stentType=stentType, what=what, params=p)



## To add to existing ssdf as seperate graph in ssdf
graphname =  'spine' # 'branch_sma_dist'
params = p
add_segmentation_ssdf(basedir, ptcode, ctcode, cropname, seeds, model,  
                        graphname, modelname='modelavgreg', 
                        stentType=stentType, params=params)


                        
## To add to existing graph in saved ssdf (merge with existing model)
add_segmentation_graph(basedir, ptcode, ctcode, cropname, seeds, model, 
                       graphname='', modelname='modelavgreg')



## Make model dynamic (load and store/overwrite to disk)

import pirt
from stentseg.motion.dynamic import incorporate_motion_nodes, incorporate_motion_edges 
from visvis import ssdf 

#todo: create function in save_segmentation
# Load deforms
s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
deformkeys = []
for key in dir(s):
    if key.startswith('deform'):
        deformkeys.append(key)
deforms = [s[key] for key in deformkeys]
deforms = [pirt.DeformationFieldBackward(*fields) for fields in deforms]
paramsreg = s.params

# Load model
s = loadmodel(basedir, ptcode, ctcode, cropname, 'model'+what)
for key in dir(s):
    if key.startswith('model'):
        model = s[key]
    
        # Combine ...
        incorporate_motion_nodes(model, deforms, s.origin) # adds deforms PointSets
        incorporate_motion_edges(model, deforms, s.origin) # adds deforms PointSets
    
        s[key] = model.pack()
    
    # also pack seeds before save
    if key.startswith('seeds'):
        s[key] = s[key].pack()
        
# Save back
filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'model'+what)
s.paramsreg = paramsreg
ssdf.save(os.path.join(basedir, ptcode, filename), s)
print('saved dynamic to disk in {} as {}.'.format(basedir, filename) )
