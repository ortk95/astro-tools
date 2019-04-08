"""
Various scientific functions.

Assume wavelengths are in microns, all other values are SI.
"""
import astropy
import astropy.units as u
import miepython
import numpy as np
from scipy.constants import h, c, k

import tools

UNIT_SYMBOLS = ['A', 'K', 's', 'm', 'kg', 'g', 'cd', 'mol', 'h', 'min', 'as']
SI_PREFIXES = {'Y': 24, 'Z': 21, 'E': 18, 'P': 15, 'T': 12, 'G': 9, 'M': 6, 'k': 3, '': 0,
               'm': -3, 'µ': -6, 'n': -9, 'p': -12, 'f': -15, 'a': -18, 'z': -21, 'y': -24,
               'c': -2, 'u': -6}


#%% UTILITIES -------------------------------------------------------------------------------------

def unit_str(value, unit='', fmt='g', use_c=False, use_ascii=False, space=False):
    """
    Produces string containing value and unit with appropriate SI prefix.

    >>> unit_str(12300, 'µm', use_c=True, space=True)
    '1.23 cm'
    >>> unit_str(range(500,2000,500), 'g')
    ['500g', '1kg', '1.5kg']

    Parameters
    ----------
    value
        List of numbers or single number.

    unit : str
        Unit of input data.

    Returns
    -------
    str or list
        String/list of strings containing formatted values with their
        associated units.

    Other Parameters
    ----------------
    fmt : str
        Desired string formatting of values (see Python string format
        mini-language).

    use_c : bool
        Toggle allowing c to be used as an SI prefix.

    use_ascii : bool
        Toggle to use ASCII strings for units ('u' for micro) instead of full
        Unicode ('µ' for micro).

    space : bool
        Toggle space between number and unit.
    """
    try:
        return [unit_str(v, unit, fmt, use_c, use_ascii, space) for v in value]
    except TypeError:
        pass
    si_dict = dict(sorted(SI_PREFIXES.items(), key=lambda kv: kv[1], reverse=True))
    if unit:
        if len(unit) > 1:
            value *= 10**si_dict[unit[0]]
            unit = unit[1:]
    if not use_c:
        si_dict.pop('c', None)
    if use_ascii:
        si_dict.pop('µ', None)
    else:
        si_dict.pop('u', None)

    value_power = np.log10(value)
    if value == 0:
        prefix = ''
        power = si_dict[prefix]
    else:
        for prefix, power in si_dict.items():
            if power <= value_power:
                break
    value *= 10**(-power)
    space_str = ' ' if space else ''
    return f'{value:{fmt}}{space_str}{prefix}{unit}'


def microns_to_metres(wavelengths, microns=True):
    """
    Convert microns to metres if needed.

    Parameters
    ----------
    wavelengths
        Input wavelengths in m/µm depending on microns.

    microns : bool
        Toggle between wavelengths in microns or m.

    Returns
    -------
    wavelengths in m
    """
    if microns:
        return np.array(wavelengths)*1e-6
    else:
        return np.array(wavelengths)


def absolute_to_apparent_magnitude(abs_mag, d_au):
    """
    Converts absolute magnitude to apparent magnitude.

    Parameters
    ----------
    abs_mag : float
        Absolute magnitude.

    d_au : float
        Distance in AU.

    Returns
    -------
    float
    """
    d_pc = float(((d_au*u.AU)/u.pc).decompose())
    return abs_mag + 5*(np.log10(d_pc)-1)


def wavelength_to_frequency(wavelengths, microns=True):
    """
    Convert wavelength to frequency.

    Parameters
    ----------

    Parameters
    ----------
    wavelengths
        Input wavelengths in m/µm depending on microns.

    microns : bool
        Toggle between wavelengths in microns or m.

    Returns
    -------
    frequency
    """
    wavelengths = microns_to_metres(wavelengths, microns)
    return c/wavelengths


def linear_fit(x_out, x_in, y_in):
    """
    Return linear fit line from given pair of x and y coordinates.

    Parameters
    ----------
    x_out : list
        List of x coordinates to calculate y for.

    x_in, y_in : list
        Pairs of x and y coordinates on line to define linear fit.

    Returns
    -------
    y = gradient*x_in + intercept
    """
    dx = x_in[1] - x_in[0]
    dy = y_in[1] - y_in[0]
    gradient = dy/dx
    intercept = y_in[0] - gradient*x_in[0]
    return gradient*np.array(x_out) + intercept


#%% SCIENTIFIC FUNCTIONS --------------------------------------------------------------------------

def planck_lambda(wavelengths, T, microns=True):
    """
    Planck function per unit wavelength.

    Parameters
    ----------
    wavelengths
        Input wavelengths in m/µm depending on microns.

    T : float
        Temperature in K.

    microns : bool
        Toggle between wavelengths in microns or m.

    Returns
    -------
    B_lambda
    """
    wavelengths = microns_to_metres(wavelengths, microns)
    return (2*h*c**2/wavelengths**5)/(np.expm1(h*c/(wavelengths*k*T)))


def planck_nu(wavelengths, T, microns=True):
    """
    Planck function per unit frequency.

    Parameters
    ----------
    wavelengths
        Input wavelengths in m/µm depending on microns.

    T : float
        Temperature in K.

    microns : bool
        Toggle between wavelengths in microns or m.

    Returns
    -------
    B_nu
    """
    nu = wavelength_to_frequency(wavelengths, microns)
    return ((2*h*nu**3)/(c**2))/(np.expm1(h*nu/(k*T)))


def mie(wavl, r, n=1, k=None):
    """
    Wrapper to calcuilate mie scattering parameters using miepython.

    Parameters
    ----------
    wavl
        Wavelength values, same units as r.

    r
        Particle radius values, same units as wavl.

    n
        Real refractive index values. If k is None, n can be the complex
        refractive index in form n - ik.

    k
        Imaginary refractive index values.

    Returns
    -------
    Q_ext, Q_sca, Q_back, g
    """
    if k is None:
        m = np.array(n)
    else:
        m = np.array(n) - 1j*np.array(k)
    x = 2*np.pi*np.array(r)/np.array(wavl)
    Q_ext, Q_sca, Q_back, g = miepython.mie(m, x)
    return Q_ext, Q_sca, Q_back, g


def oren_nayar_reflectance(theta_r, theta_i, phi_r, phi_i, sigma, rho, degrees=True,
                           cos_correction=False):
    """
    Oren-Nayar reflectance model.

    Parameters
    ----------
    theta_r, theta_i : float or array
        Reflected and incident polar angles.

    phi_r, phi_i : float or array
        Reflected and incident azimuth angles.

    sigma : float or array
        Surface roughness.

    rho : float
        Surface albedo.

    degrees : bool
        Toggle between degrees and radians.

    cos_correction : bool
        Toggle enabling viewing angle cos correction.

    Returns
    -------
    reflectance

    References
    ----------
    Oren and Nayar (1994) Generalization of Lambert's reflectance model
    DOI: 10.1145/192161.192213

    """
    phi = phi_r - phi_i
    if degrees:
        theta_r = np.deg2rad(theta_r)
        theta_i = np.deg2rad(theta_i)
        phi = np.deg2rad(phi)
        sigma = np.deg2rad(sigma)
    theta_r, theta_i, phi, sigma = tools.maths.consistent_dims(theta_r, theta_i, phi, sigma)
    # Ensure theta values >= 0
    with np.errstate(invalid='ignore'):
        # Allow nan values to propagate silently
        phi[np.where(theta_i < 0)] += np.pi
        phi[np.where(theta_r < 0)] += np.pi
    theta_i = np.abs(theta_i)
    theta_r = np.abs(theta_r)
    alpha = tools.maths.list_max(theta_r, theta_i)
    beta = tools.maths.list_min(theta_r, theta_i)
    c1 = 1 - 0.5*sigma**2/(sigma**2 + 0.33)
    with np.errstate(invalid='ignore'):
        c2_condition = np.array(np.cos(phi) >= 0, dtype=float)
    c2_t = 0.45*sigma**2/(sigma**2 + 0.09)*np.sin(alpha)
    c2_f = 0.45*sigma**2/(sigma**2 + 0.09)*(np.sin(alpha) - (2*beta/np.pi)**3)
    c2 = c2_t*c2_condition + c2_f*(1 - c2_condition) # Select values depending on condition
    c3 = 0.125*(sigma**2/(sigma**2 + 0.09))*(4*alpha*beta/np.pi**2)**2
    # E_0*np.cos(theta_i) factor not included in li components as it cancels in final calculation.
    l1 = rho/np.pi*(c1
                    + np.cos(phi)*c2*np.tan(beta)
                    + (1 - np.abs(np.cos(phi)))*np.tan((alpha + beta)/2)*c3)
    l2 = 0.17*rho**2/np.pi*(sigma**2/(sigma**2 + 0.13))*(
            1 - np.cos(phi)*(2*beta/np.pi)**2)
    l_tot = l1 + l2

    with np.errstate(invalid='ignore'):
        # Set brightness to zero where there is no line of sight
        l_tot[np.where(np.abs(np.mod(theta_r, 2*np.pi)) > np.pi/2)] = 0
        l_tot[np.where(np.abs(np.mod(theta_i, 2*np.pi)) > np.pi/2)] = 0
    if cos_correction:
        return l_tot
    else:
        return l_tot*np.cos(theta_i)


def oren_nayar_disc_model(coords, x0, y0, r0, theta_i=0, phi_i=0, sigma=0, rho=0.5,
                          replace_nan=True):
    """
    Calculate the Oren-Nayar model for an image of a sphere.

    Parameters
    ----------
    coords : int or tuple of arrays
        Size of output image (int) or tuple of x, y coordinate images.

    x0, y0 : float
        Coordinates of centre of disc.

    r0 : float
        Radius of disc in pixels.

    theta_i, phi_i : float
        Incident light polar and azimuth angles.

    sigma : float
        Surface roughness.

    rho : float
        Surface albedo.

    replace_nan : bool
        Replace nan values in model with 0. Useful for simulating image against
        black sky.

    Returns
    -------
    Modelled image
    """
    if np.isscalar(coords):
        x_img, y_img = np.meshgrid(np.arange(coords), np.arange(coords))
    else:
        x_img, y_img = coords

    with np.errstate(divide='ignore', invalid='ignore'):
        x_img = (x_img - x0)/r0
        y_img = (y_img - y0)/r0
        r_img = np.sqrt(x_img**2 + y_img**2)
        z_img = np.sqrt(1 - x_img**2 - y_img**2)
        x_img[np.where(r_img > 1)] = np.nan
        y_img[np.where(r_img > 1)] = np.nan
        z_img[np.where(r_img > 1)] = np.nan
    theta_r_img, phi_r_img = tools.mapping.xyz_to_thetaphi(x_img, y_img, z_img)
    theta_i_img, phi_i_img = tools.mapping.xyz_to_thetaphi(x_img, y_img, z_img, theta_i, phi_i)
    model = oren_nayar_reflectance(theta_r_img, theta_i_img, phi_r_img, phi_i_img, sigma, rho)
    if replace_nan:
        model[np.where(np.isnan(model))] = 0
    return model


def convolve_airy_disc(img, radius, **kwargs):
    """
    Convolve airy disc of given radius with image.

    Parameters
    ----------
    img : array
        Image to convolve airy disc to. If cube, convolution applied to each
        frame independently.

    radius : float or int
        Radius of airy disc.

    kwargs
        Additional arguments passed to astropy.convolution.convolve_fft().

    Returns
    -------
    Convolved image.
    """
    if not radius:
        # Short circuit case for perfect zero diffracton airy disc
        return img
    if np.ndim(img) > 2:
        return np.array([convolve_airy_disc(f, radius) for f in img])
    kernel = astropy.convolution.AiryDisk2DKernel(radius, mode='oversample')
    return astropy.convolution.convolve_fft(img, kernel, **kwargs)


def get_disc_size(hdr):
    """
    Get disc size in arcseconds of target from FITS header.

    Parameters
    ----------
    hdr
        FITS header

    Returns
    -------
    radius_as
        Radius of disc in arcseconds.
    """
    dist = tools.mapping.get_ephemerides(hdr)['delta']
    dist = float(dist/u.m) # Distance to Europa in m
    target = hdr['OBJECT'].casefold()
    # Radii in m
    r_dict = {
            'io': 3637.4e3/2,
            'europa': 1561e3,
            'ganymede': 5268.2e3/2,
            'callisto': 4820.6e3/2
            }
    r = r_dict[target]
    radius_as = np.rad2deg(np.arcsin(r/dist))*60*60
    return radius_as
