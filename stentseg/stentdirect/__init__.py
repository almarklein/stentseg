# -*- coding: utf-8 -*-
# Copyright (c) 2014, Almar Klein
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

""" Package stentDirect

Provides a class to perform a step by step approach to segment 
the stent, showing results in between. 

Some old notes on memory usage
------------------------------

Notes on memory usage. On the full 512x512x360 datasets a memory error is
produced. I found that for each patient a 256x256x256 dataset can be 
obtained (by cropping) in which the stent is fully visible. This cropping
has to be done by hand (some attempts to automate this failed), and I 
wrote a nice tool for that. A size of a power of two is preferred, otherwise
I cannot render the data on my laptop (which has an ATI card).

So assuming a dataset of 256x256x256, the memory requirement is:
- 32 MB for the int16 dataset represented in CT-values
- 64 MB for the float32 speed map
- 64 MB for the float32 time map
- 64 MB for the int32 crossref array (in the binary heap)
- 64 MB for the int32 traceback array
- 16 MB for the int8 edge map array (ndindex)
- 64 MB for the int32 idmap
total: 5*64+32+16 = 368 MB

"""

# Imports
import sys, os, time
from visvis import ssdf

from .base import StentDirect, StentDirect_oldMCP

# Get default params
def getDefaultParams(stentType=''):
    """ getDefaultParams()
    Get the paramater stuct filled with defaults.
    These defaults may not be optimal for you stent type and/or scanner
    configuration.
    """
    
    # Generic params
    params = ssdf.new()
    # The threshold for detecting seed points
    params.seed_threshold = 650 
    # The scale factor for the data to create speed image 
    params.mcp_speedFactor = 100         
    # The MCP threshold. Small=faster, but will miss connections if too small!   
    params.mcp_evolutionThreshold = 0.06    
    # The Expected Number of Connections
    params.graph_expectedNumberOfEdges = 2  
    # The th to determine a really strong connection
    params.graph_strongThreshold = 1200     
    # The th to determine a really weak connection
    params.graph_weakThreshold = 100        
    # The size of tails to trim and clusters to remove
    params.graph_trimLength = 3             
    params.graph_minimumClusterSize = 8
    
    # Stent type dependencies    
    if stentType == 'zenith':
        params.graph_expectedNumberOfEdges = 2
    elif stentType == 'talent':
        params.graph_expectedNumberOfEdges = 2
    elif stentType == 'aneurx':
        params.graph_expectedNumberOfEdges = 4
        params.graph_minimumClusterSize = 400
    elif stentType:
        raise ValueError('Unknown stent type %s' % stentType)

    # Done
    return params

