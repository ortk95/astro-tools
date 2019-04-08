"""
File path record.

Central location to modify and save paths in file system.
"""
import os


def generate_path(*args):
    """
    Create consistently formatted file/directory path string.

    Parameters
    ----------
    args : str
        Series of strings representing parts of path

    Returns
    -------
    str
    """
    path = os.path.sep.join(args)
    path = os.path.normpath(path)
    return path


# Path to root directory, all paths relative to this. ROOT is currently taken
# from the location of this file (path.py) - a fixed string can be specified
# here instead.
ROOT = generate_path(os.path.split(__file__)[0], '..')

# Other useful path constants - set here so they can be used in multiple places
# and managed from this central location for easy reorganisation.
DATA = generate_path(ROOT, 'data')
CODE = ROOT

# Useful path generating functions
def data(*args):
    return generate_path(DATA, *args)

def root(*args):
    return generate_path(ROOT, *args)

def code(*args):
    return generate_path(CODE, *args)

def trash(*args):
    return generate_path(TRASH, *args)
