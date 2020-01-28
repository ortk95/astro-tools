"""
Tools
=====

Package to combine various useful tools which are common to different routines.

Submodules
----------
script
    General tools for use in scripts such as calling shell commands and
    printing fancy output. Contains tools which do not logically fit into any
    other submodule.

file
    Tools for use in file I/O such as interacting with FITS files and data.

image
    Tools for image manipulation including image cleaning, display and
    analysis.

path
    Supplies common file path locations used in different routines. Provides a
    single location to update file structure for different areas of code.

spectrum
    Tools for analysing and fitting spectra.

mapping
    Tools for mapping data and displaying mapped data.

reference
    Tools for dealing with reference information and interfacing with BibTeX
    data.

science
    Scientific functions and models such as reflectance modelling.

maths
    Mathematical functions and useful routines to extend numpy-like behaviour.
"""

__version__ = '1.2'
__date__ = '2020-01-28'
__author__ = 'Oliver King'

from . import script
from . import file
from . import image
from . import path
from . import spectrum
from . import mapping
from . import reference
from . import science
from . import maths
