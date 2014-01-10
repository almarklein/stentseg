# Stentseg

This library provides functionality to perform segmentation of a stent
graft from 3D CT data.

This package contains code from multiple algorithms to segment the
stent grafts. These represent the different attempt that I did
during my PhD. The only algorithmm that is exposed is the 3-step
stent-direct algorithm.

This library is written for Python 3, and probably also works for Python 2.

## Installation

You can place this directory on your PYTHONPATH, copy the stentseg_proxy.py
one directory below (so it is on your PYTHONPATH as well), and then 
rename it to stentseg.py.

Alternatively you can run ``python setup.py install``.


## License

Copyright (c) 2014, Almar Klein

Distributed under the (new) BSD License. See LICENSE.txt for more info.

