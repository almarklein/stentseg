""" stentseg

This library provides functionality to perform segmentation of a stent
graft from 3D CT data.

This package contains code from multiple algorithms to segment the
stent grafts. These represent the different attempt that I did
during my PhD. The only algorithmm that is exposed is the 3-step
stent-direct algorithm.

In the current situation the algorithms are assumed to return a model
of the graph based on a geometric graph, but additional models may be
added later.

This library is written for Python 3, and probably also works for Python 2.
It is written in pure Python, so installation is easy.

"""

__version__ = '0.1'

from stentseg import utils
from stentseg import stentdirect
