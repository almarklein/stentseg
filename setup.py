""" Stentseg setup script.

"""

import os
from distutils.core import setup

name = 'stentseg'
description = 'Segmentation of stent grafts from 3D CT data.'


# Get version and docstring
__version__ = None
__doc__ = ''
docStatus = 0 # Not started, in progress, done
initFile = os.path.join(os.path.dirname(__file__), 'stentseg', '__init__.py')
for line in open(initFile).readlines():
    if (line.startswith('__version__')):
        exec(line.strip())
    elif line.startswith('"""'):
        if docStatus == 0:
            docStatus = 1
            line = line.lstrip('"')
        elif docStatus == 1:
            docStatus = 2
    if docStatus == 1:
        __doc__ += line


setup(
    name = name,
    version = __version__,
    author = 'Almar Klein',
    author_email = 'almar.klein@gmail.com',
    license = '(new) BSD',
    
    url = 'https://bitbucket.org/almarklein/stentseg',
    download_url = 'https://bitbucket.org/almarklein/stentseg',    
    keywords = "CT stent aneurysm medical segmentation",
    description = description,
    long_description = __doc__,
    
    platforms = 'any',
    provides = ['stentseg'],
    install_requires = ['numpy', 'visvis', 'scipy', 'skimage'],
    
    packages = ['stentseg',
                'stentseg.utils', 
                'stentseg.stentdirect', 
                'stentseg.apps',
               ],
    package_dir = {'stentseg': 'stentseg'},
    #package_data = {'stentseg': ['data/*']},
    zip_safe = False,
    
    classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.3',
          ],
    )
