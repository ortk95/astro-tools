# astro-tools
Collection of Python tools for astronomy.

## Setup
### Requirements
This code was developed and tested in [Python 3.7](https://docs.python.org/3.7/). Many scripts may work with earlier Python 3 versions, but some features (like [f-strings](https://docs.python.org/3/whatsnew/3.6.html#pep-498-formatted-string-literals) for versions < 3.6) will cause errors.

The Python packages required for running all functions are:

* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Astropy](http://www.astropy.org/)
* [Astroquery](https://astroquery.readthedocs.io/en/latest/)
* [Photutils](https://photutils.readthedocs.io/en/stable/)
* [PIL](http://www.pythonware.com/products/pil/)
* [BibtexParser](https://bibtexparser.readthedocs.io/en/master/)
* [Miepython](https://github.com/scottprahl/miepython)

Many of these are automatically included in Anaconda distributions. Additional modules can be installed using the usual `conda install ...` or `pip install ...`. The modules are listed here in approximate order of importance — some of the later functions are only used once, so can be commented out if not needed.

### Usage
This code can generally be downloaded and run with no additional setup or installation. Some functions may rely on paths defined in [`tools.path`](tools/path.py) which will need to be defined correctly for your file system.


## Documentation
Documentation is included in the [docstrings](https://www.python.org/dev/peps/pep-0257/) (strings enclosed by triple quotes `"""..."""`) within the source code at the top of each module/function. This typically includes a description of the function, its parameters and return values. Examples of function usage in Python begin with `>>>`, the output of the function is sometimes given underneath:

```python
"""
>>> print('Hello world')
Hello world
"""
```

Examples of script usage from the system shell are also sometimes given in the docstrings, indicated by lines beginning with `$`.


## File organisation
### Common tools
General functions which can be used in multiple scripts are contained in the `tools` module (i.e. the [`tools`](tools) directory). These can all be imported into a Python script with a single line (`import tools`) providing access to the whole range of functions. For example, the following example code use functions from the `tools` module to read a FITS file, print out the FITS header and show the first slice of the cube:

```python
import tools
cube, hdr = tools.file.read_fits('spectral_cube.fits.gz')
tools.file.print_header(hdr)
tools.image.show_image(cube[0], contours=True, show=True)
```

The tools module is split into a series of sub-modules, each with a functions based around different 'themes'.

 Submodule | Theme
---|---
[`file`](tools/file.py) | File I/O, e.g. `read_fits()` to read FITS files, `check_path()` to check a directory tree exists and create it if necessary.
[`image`](tools/image.py) | Image processing, visualisation and analysis, e.g. `show_image()` to easily display images, `stack_images()` to create a multicolour image.
[`maths`](tools/maths.py) | Mathematical functions and operations, e.g. `normalise()` to normalise an iterable, `rms()` to calculate the RMS error between two iterables.
[`science`](tools/science.py) | Scientific functions and models, e.g. `planck_lambda()` to calculate blackbody radiance, `unit_str()` to convert a value to a string with appropriate units.
[`mapping`](tools/mapping.py) | Positional and map related functions, e.g. `get_ephemerides()` to load ephemeris information, `longlat_to_map()` to generate a map from an image and its associated long/lat coordinates.
[`spectrum`](tools/spectrum.py) | Functions for spectral analysis, e.g. `get_wavelengths()` to get a list of wavelengths from a FITS file, `rebin_spectrum()` to rebin a spectrum to different wavelength range/resolution.
[`reference`](tools/reference.py) | Functions for handling reference information, e.g. `load_bib()` to load a BibTex file and `get_author_str()` to generate a string with appropriate 'et al.' etc.
[`script`](tools/script.py) | General functions, e.g. `cprint()` to print colourful text, `progress_bar()` to print a progress bar, `sort_mixed()` to sort a list of strings correctly by numerical value.
[`path`](tools/path.py) | Submodule to define, generate and set file paths from a single location.

See the source code and associated docstring documentation for all the functions contained in `tools`.

Note that `import tools` will only work if the `tools` directory is on the system path (i.e. the parent directory of `tools` is present in `sys.path`). This can be achieved by placing the `tools` directory in the current working directory (like for the example scripts below). This can also be achieved by manually adding the parent directory to the system path:

```python
import sys
sys.path.append('/path/to/parent/directory')
import tools
```

### Scripts
Scripts in the root directory use general functions from `tools` to carry out specific tasks.

* [`find_observing_times.py`](find_observing_times.py) uses ephemeris information to identify good observing times for a specific target and location.
* [`sphere_time_calculation.py`](sphere_time_calculation.py) simplifies calculation of observing times for the VLT/SPHERE instrument.
* [`time_code.py`](time_code.py) allows easy comparison of the execution times of code snippets.
*  [`git_commit.py`](git_commit.py) can be used to simplify the process of creating git commits and pushing them to a remote branch (e.g. GitHub). It is self-contained (i.e. it does not use the `tools` module), so can be downloaded and used as a stand-alone script. See the docstrings in the file for more details.
