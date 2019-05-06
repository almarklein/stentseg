""" Store segmentation result to ssdf

"""
from lspeas.utils.storesegmentation import save_segmentation, add_segmentation_ssdf

# Get graph model
model = sd._nodes3
seeds = sd._nodes1

## Option 1: Store segmentation to disk

save_segmentation(basedir, ptcode, ctcode, cropname, seeds, model, s, 
                    stentType=stentType, what=what, params=p)



## Option 2: Add to existing ssdf as seperate graph
if False:
    graphname =  'spine' # 'branch_sma_dist'
    params = p
    add_segmentation_ssdf(basedir, ptcode, ctcode, cropname, seeds, model,  
                    graphname, modelname='modelavgreg', 
                    stentType=stentType, params=params)


                        
## Option 3: Add to existing ssdf in existing graph (merge with existing model)
if False:
    add_segmentation_graph(basedir, ptcode, ctcode, cropname, seeds, model, 
                    graphname='', modelname='modelavgreg')




## Make model dynamic (load and store/overwrite to disk)

from lspeas.utils.storesegmentation import make_model_dynamic

make_model_dynamic(basedir, ptcode, ctcode, cropname, what='avgreg')
