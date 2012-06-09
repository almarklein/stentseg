""" stentseg

Library for segmentation of stent grafts. This library provides functionality
to perform segmentation of a stent graft from 3D data.

In the current situation the algorithms are assumed to return a model
of the graph based on a geometric graph, but additional models may be
added later.

Notice: this library was fitst written for Python2.x, but I may change to
Python3.x and drop suppory for Python2.x. 

"""

from . import utils
from . import stentdirect
