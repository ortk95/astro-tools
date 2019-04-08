"""
Module containing spectral analysis functions.
"""
import os
import urllib

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

import tools

# Wavelengths of VLT/SPHERE IFS in YJH mode
SPHERE_WAVELENGTHS = [0.95300000, 0.97205263, 0.99110526, 1.01015789, 1.02921053,
                      1.04826316, 1.06731579, 1.08636842, 1.10542105, 1.12447368,
                      1.14352632, 1.16257895, 1.18163158, 1.20068421, 1.21973684,
                      1.23878947, 1.25784211, 1.27689474, 1.29594737, 1.31500000,
                      1.33405263, 1.35310526, 1.37215789, 1.39121053, 1.41026316,
                      1.42931579, 1.44836842, 1.46742105, 1.48647368, 1.50552632,
                      1.52457895, 1.54363158, 1.56268421, 1.58173684, 1.60078947,
                      1.61984211, 1.63889474, 1.65794737, 1.67700000]
SPHERE_WAVELENGTHS = np.array(SPHERE_WAVELENGTHS)
SPHERE_WAVELENGTHS.setflags(write=False) # Make array read only


def get_wavelengths(header, return_units=False, replace_units=True):
    """
    Get wavelengths from FITS header.

    Parameters
    ----------
    header
        FITS header object.

    return_units : bool
        Toggle returning string containing unit information.

    replace_units : bool
        Toggle replacing 'Microns' with 'µm'

    Returns
    -------
    List of wavelengths and optional unit string
    """
    if header['CTYPE3'] != 'WAVE':
        print(f'WARNING: header item CTYPE3 = {header["CTYPE3"]}')
    units = header['CUNIT3']
    crval3 = header['CRVAL3']
    cd3_3 = header['CD3_3']
    naxis3 = header['NAXIS3']
    wavl = crval3 + cd3_3*np.arange(0, naxis3)
    if replace_units:
        if units == 'Microns':
            units = 'µm'
    if return_units:
        return wavl, units
    else:
        return wavl


def plot(fits, x_idx, y_idx):
    """
    Plot spectrum from specific location in FITS file.

    Parameters
    ----------
    fits : str or tuple
        Path to fits file or (cube, hdr) tuple.

    x_idx, y_idx : int
        Coordinates of point to plot spectrum
    Returns
    -------
    Spectrum
    """
    img, hdr = tools.file.read_fits(fits)
    wavl, wavl_units = get_wavelengths(hdr, return_units=True)
    ss = img[:, y_idx, x_idx]
    plt.plot(wavl, ss)
    plt.xlabel(f'Wavelength ({wavl_units})')
    return ss


def load_star_reference_spectrum(spectral_type, luminosity_class='', wavelengths_out=None):
    """
    Loads reference stellar spectrum from ESO database.

    Spectrum will be loaded from local storage, and only downloaded if it isn't
    in local storage. Spectra from
    https://www.eso.org/sci/observing/tools/standards/IR_spectral_library.html


    Parameters
    ----------
    spectral_type : str
        Star spectral type (e.g. 'G0').

    luminosity_class : str
        Star luminosity class (e.g. 'V').

    wavelengths_out : list
        Optionally specify wavelengths to rebin the spectrum to.

    Returns
    -------
    Wavelengths and spectrum
    """
    if luminosity_class == '' and spectral_type[-1].isdigit():
        print('Assuming main sequence luminosity class (V)')
        luminosity_class = 'v'
    spectral_type = spectral_type.lower()
    luminosity_class = luminosity_class.lower()
    filename = f'uk{spectral_type}{luminosity_class}.fits'
    filepath = tools.path.gbdata('calibration_spectra', filename)
    if not os.path.isfile(filepath):
        # Download file if it doesn't already exist
        print('Downloading reference spectrum fron ESO website at:')
        url = f'ftp://ftp.eso.org/web/sci/observing/tools/standards/IR_spectral_library/{filename}'
        print(' ' + url)
        response = urllib.request.urlopen(url)
        data = response.read()
        with open(filepath, 'wb') as f:
            f.write(data)
    spectrum, hdr = tools.file.read_compressed_fits(filepath)
    wavelengths = hdr['CRVAL1'] + hdr['CD1_1']*np.arange(0, hdr['NAXIS1']) # Angstroms
    wavelengths /= 1e4 # Convert to microns
    if wavelengths_out is None:
        return wavelengths, spectrum
    else:
        return rebin_spectrum(wavelengths, spectrum, wavelengths_out)


def rebin_spectrum(x_in, y_in, x_out, intermediate_rebin=100, strip=True, require_data=False):
    """
    Rebin spectrum to different wavelength range.

    Rebins interpolating original spectrum, oversampling then averaging in
    output bins.

    Parameters
    ----------
    x_in : list
        Input wavelengths.

    y_in : list
        Input spectra.

    x_out : list
        Output wavelengths.

    intermediate_rebin : int
        Factor to oversample at intermediate rebin. Larger values will be more
        accurate but take more memory.

    strip : bool
        Strip out unused wings of spectrum before interpolation to increase
        speed. Useful for input which covers a large wavelength range and
        output which covers a narrow range.

    require_data : bool
        Require at least one input data point in the bin to produce a non-NaN
        output value. Useful for down-sampling high resolution data to preserve
        blanked regions, but will cause issues when up-sampling low resolution
        data.

    Returns
    -------
    y_out
    """
    x_in, y_in = zip(*sorted(zip(x_in, y_in)))
    x_in = np.array(x_in)
    y_in = np.array(y_in)
    x_out = np.array(x_out)
    if strip:
        dx = (np.nanmax(x_out) - np.nanmin(x_out))*0.25
        if np.nanmax(x_in) > np.nanmax(x_out) + dx or np.nanmin(x_in) < np.nanmin(x_out) - dx:
            keep_idxs = np.where(np.logical_and(np.nanmin(x_out) - dx < x_in,
                                                x_in < np.nanmax(x_out) + dx))
            y_in = y_in[keep_idxs]
            x_in = x_in[keep_idxs]
    if len(x_in) == 0:
        return np.full_like(x_out, np.nan)
    if intermediate_rebin and intermediate_rebin != 1 and len(x_in) > 1:
        dx = min(max(np.diff(x_in)), min(np.diff(x_out)))/intermediate_rebin
        x_hr = np.arange(min(*x_in, *x_out), max(*x_in, *x_out), dx)
        y_hr = interpolate.PchipInterpolator(x_in, y_in, extrapolate=False)(x_hr)
    else:
        x_hr, y_hr = x_in, y_in
    bin_edges = 0.5*(x_out[1:] + x_out[:-1])
    bin_edges = [2*x_out[0] - bin_edges[0], *bin_edges, 2*x_out[-1] - bin_edges[-1]]
    y_out = np.full_like(x_out, np.nan)
    for idx, _ in enumerate(x_out):
        x_min = bin_edges[idx]
        x_max = bin_edges[idx + 1]
        if require_data and not len(y_in[(x_in >= x_min) & (x_in < x_max)]):
            continue
        y_out[idx] = np.mean(y_hr[(x_hr >= x_min) & (x_hr < x_max) & (~np.isnan(y_hr))])
    return y_out


def plot_spectra_map(fits, locations=10, img_frame=0, img_frame_line=True):
    """
    Plot map of spectra from a FITS file.

    Parameters
    ----------
    fits : str or tuple
        Path to fits file or (cube, hdr) tuple.

    locations : int or list
        Interval to get spectra from map or coords of spectra ot get.

    img_frame : int
        Index of image frame to show.

    img_frame_line : bool
        Put vertical line corresponding to img_frame on plot.
    """
    img, hdr = tools.file.read_fits(fits)
    if np.isscalar(locations):
        locations = np.meshgrid(np.arange(0, img.shape[-1], locations),
                                np.arange(0, img.shape[-2], locations))
        locations = list(zip(locations[0], locations[1]))
    locations = np.array(locations)
    if locations.ndim == 1:
        locations = np.array([locations])

    wavl, wavl_units = get_wavelengths(hdr, return_units=True)

    plt.clf()
    plt.subplot(1, 2, 1)
    tools.image.show_image(img[img_frame], show=False)
    img_wavl = wavl[img_frame]
    plt.title('Image at {:.3f} {}'.format(img_wavl, wavl_units))
    for loc in locations:
        x_idx = loc[0]
        y_idx = loc[1]
        plt.scatter(x_idx, y_idx, marker='x')

    plt.subplot(1, 2, 2)
    if img_frame_line:
        plt.axvline(x=img_wavl, color='k', linestyle=':', linewidth=0.5, alpha=0.5)
    for loc in locations:
        x_idx = loc[0]
        y_idx = loc[1]
        plot((img, hdr), x_idx, y_idx)
    re_info = tools.file.get_reduction_info(hdr, re_idx=-1)['TAR FILE']
    tools.image.add_fig_metadata(re_info + ' | ')
