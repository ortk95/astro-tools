"""
Module for mathematical operations and functions.
"""
import math

import numpy as np


def normalise(values, percentile=False, top=1, bottom=0):
    """
    Normalise iterable.

    Parameters
    ----------
    values : list
        Iterable to normalise.

    percentile : bool or float or int
        Percentile to set top and bottom values e.g. percentile = 5 sets the
        5th percentile of values as 0 and the 95th as 1.

    top, bottom : float
        Limits of normalised values.

    Returns
    -------
    Normalised values
    """
    assert top > bottom
    values = np.array(values)
    if percentile:
        vmin = np.nanpercentile(values, percentile)
        vmax = np.nanpercentile(values, 100-percentile)
    else:
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)

    # Put into 0 to 1 range
    if vmax != vmin:
        values = (values - vmin)/(vmax - vmin)
    else:
        values = values - vmin
    return values*(top - bottom) + bottom


def consistent_dims(*args):
    """
    Forces input arguments to be consistent dimensions.

    >>> tools.maths.consistent_dims(1, [1,2,3])
    [array([1., 1., 1.]), array([1, 2, 3])]

    Parameters
    ----------
    args
        Series of arguments which are either scalars or arrays of same shape.

    Returns
    -------
    List of arrays of same shape
    """
    dims = [np.ndim(a) for a in args]
    if len(set(dims)) == 1:
        # Short circuit case for all same dimensions
        return [np.atleast_1d(a) for a in args]
    assert len(set(dims)) == 2
    assert min(dims) == 0
    shape = np.shape(args[dims.index(max(dims))])
    args = [a*np.ones(shape) if np.ndim(a) == 0 else np.array(a) for a in args]
    return args


def list_max(a, b):
    """
    Finds maximum value at each element between two lists.

    Arguments are passed to consistent_dims() first.

    >>> tools.maths.list_max([1,2,3], [3,2,1])
    array([3, 2, 3])
    >>> tools.maths.list_max(range(10), 5)
    array([5., 5., 5., 5., 5., 5., 6., 7., 8., 9.])

    Parameters
    ----------
    a, b
        Lists to find max at each element.

    Returns
    -------
    List of max values from arrays.
    """
    with np.errstate(invalid='ignore'):
        return np.max(consistent_dims(a, b), axis=0)


def list_min(a, b):
    """
    Finds minimum value at each element between two lists.

    Arguments are passed to consistent_dims() first.

    >>> tools.maths.list_min([1,2,3], [3,2,1])
    array([1, 2, 1])
    >>> tools.maths.list_min(range(10), 5)
    array([0., 1., 2., 3., 4., 5., 5., 5., 5., 5.])

    Parameters
    ----------
    a, b
        Lists to find min at each element.

    Returns
    -------
    List of min values from arrays.
    """
    with np.errstate(invalid='ignore'):
        return np.min(consistent_dims(a, b), axis=0)


def rms(a, b, optimize=False):
    """
    Find RMS error between two lists as sqrt(mean((a - b)**2)).

    Parameters
    ----------
    a, b : list
        Lists to find RMS of

    optimize : bool
        Toggle optimizations. This assumes a and b are same shape and both
        numpy arrays already.

    Returns
    -------
    float
        RMS value
    """
    if not optimize:
        a = np.array(a)
        b = np.array(b)
        assert a.shape == b.shape
    c = a - b
    return math.sqrt(np.mean(c*c)) # Use explicit multiplication instead of power as this is slightly faster


def nearest_idx(array, value):
    """
    Find nearest index in array to a value.

    More general than list.index() method which only works for an exact match.

    >>> tools.maths.nearest_idx([1,2,3], 2.1)
    2

    Parameters
    ----------
    array : list

    value : float or int
        Value to find closest index for

    Returns
    -------
    int
    """
    diff_arr = np.abs(np.array(array) - value)
    diff_arr[np.isnan(diff_arr)] = np.inf
    return diff_arr.argmin()


def rotation_matrix(angle, axis, degrees=True):
    """
    Creates 3D rotation matrix for specified angle and axis.

    Parameters
    ----------
    angle : float
        Angle for rotation (units specified by `degrees`).

    axis : {'x', 'y', 'z'} or {1, 2, 3}
        Axis for rotation, can be 'x'/0, 'y'/1 or 'z'/2.

    degrees : bool
        Toggle between angle units of degrees and radians.

    Returns
    -------
    M : array
        3x3 rotation matrix array.
    """
    if degrees:
        angle = np.deg2rad(angle)
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[c, -s], [s, c]])  # 2D rotation matrix
    M = np.eye(3)  # 3D identity matrix
    if axis in [0, 'x']:
        M[1:, 1:] = R
    elif axis in [1, 'y']:
        M[0, 0] = R[0, 0]
        M[-1, 0] = R[-1, 0]
        M[0, -1] = R[0, -1]
        M[-1, -1] = R[-1, -1]
    else:
        M[:2, :2] = R
    return M
