""" stentseg

This library provides functionality to perform segmentation of a stent
graft from 3D CT data.

This package contains code from multiple algorithms to segment the
stent grafts. These represent the different attempt that I did
during my PhD. The only algorithmm that is exposed is the 3-step
stent-direct algorithm.

This library is written for Python 3, and probably also works for Python 2.
It is written in pure Python, so installation is easy.

If you publish research in which this software is used, we would 
appreciate it if you refer to our paper:
    
    Klein, Almar and van der Vliet, J.A. and Oostveen, L.J. and Hoogeveen, 
    Y. and Schultze Kool, L.J. and Renema, W.K.J. and Slump, C.H. (2012) 
    Automatic segmentation of the wire frame of stent grafts from CT data. 
    Medical Image Analysis, 16 (1). pp. 127-139. ISSN 1361-8415

"""

__version__ = '0.1'

from stentseg import utils
from stentseg import stentdirect
