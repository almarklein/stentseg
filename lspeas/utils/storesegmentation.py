""" Functionality to store the segmentation in an ssdf

Author: Maaike Koenrades
"""
import visvis as vv
from visvis import ssdf
from stentseg.utils.datahandling import loadmodel
import os

def save_segmentation(basedir, ptcode, ctcode, cropname, seeds, model, s, 
                      stentType=None, what='avgreg', params=None):
    """ store segmentation in new ssdf or load and add
    s = ssdf of volume from which we segment
    s2 = ssdf of segmentation model
    """
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
    s2.params = params
    s2.stentType = stentType
    # Store model
    s2.model = model.pack()
    s2.seeds = seeds.pack()
    
    # Save
    filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'model'+what)
    ssdf.save(os.path.join(basedir, ptcode, filename), s2)
    print('saved to disk in {} as {}.'.format(basedir, filename) )
    
def add_segmentation_ssdf(basedir, ptcode, ctcode, cropname, seeds, model,  
                        graphname, s2=None, modelname='modelavgreg', 
                        stentType=None, params=None):
    """ Add a model to an existing ssdf but separate from the existing graph
    s2 = ssdf of existing model to add new segmentation graph in ssdf
    graphname = name of new graph to save in ssdf
    """
    if s2 is None:
        s2 = loadmodel(basedir, ptcode, ctcode, cropname, modelname) # unpacks to graph
    s2['params'+graphname] = params
    s2['stentType'+graphname] = stentType
    # Store model
    if model is not None: # step 2 and 3 were performed otherwise NoneType
        s2['model'+graphname] = model
        s2['seeds'+graphname] = seeds
    else: # store seeds as model
        s2['model'+graphname] = seeds.copy()
        s2['seeds'+graphname] = seeds
    
    # pack all graphs to ssdf for save
    for key in dir(s2):
        if key.startswith('model'):
            s2[key] = s2[key].pack()
        if key.startswith('seeds'):
            s2[key] = s2[key].pack()
    
    # Save
    filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, modelname)
    ssdf.save(os.path.join(basedir, ptcode, filename), s2)
    print('saved to disk in {} as {}.'.format(basedir, filename) )

def add_segmentation_graph(basedir, ptcode, ctcode, cropname, seeds, model, 
                        s2=None, graphname='', modelname='modelavgreg'):
    """ add segmentation to an existing graph
    s2 = ssdf of existing segmentation model to add graph to this model
    graphname = model is default storename for model but can also be
                modelbranch for example if this was used in add_segmentation_ssdf
    """
    if s2 is None:
        s2 = loadmodel(basedir, ptcode, ctcode, cropname, modelname) # unpacks
        
    graphold = s2['model'+graphname]
    seedsold = s2['seeds'+graphname]
    
    # add seeds graph
    seedsold.add_nodes_from(seeds.nodes(data=True)) # including attributes if any
    # add nodes and edges to graph model
    if model is not None: # step 2 and 3 were performed otherwise NoneType
        graphold.add_nodes_from(model.nodes(data=True))
        graphold.add_edges_from(model.edges(data=True))
    else: # store seeds with model
        graphold.add_nodes_from(seeds.nodes(data=True))
    
    # Store model
    s2['model'+graphname] = graphold
    s2['seeds'+graphname] = seedsold
    
    # pack all graphs to ssdf for save
    for key in dir(s2):
        if key.startswith('model'):
            s2[key] = s2[key].pack()
        if key.startswith('seeds'):
            s2[key] = s2[key].pack()
    
    # Save
    filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, modelname)
    ssdf.save(os.path.join(basedir, ptcode, filename), s2)
    print('saved to disk in {} as {}.'.format(basedir, filename) )
    
    
    
    
    
    
    

