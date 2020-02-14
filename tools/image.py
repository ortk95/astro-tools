"""
Module containing image manipulation and display code.
"""
import warnings
from datetime import datetime

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import photutils
import scipy.ndimage
from matplotlib.colors import hsv_to_rgb
from scipy.interpolate import interp2d
from scipy.ndimage.filters import median_filter, uniform_filter
from scipy.ndimage.measurements import center_of_mass

import tools

test_if_ipython = tools.script.test_if_ipython
progress_bar = tools.script.progress_bar


#%% VISUALISATION ---------------------------------------------------------------------------------

def show_image(img, title=None, colorbar=True,  show=None, contours=False, ticks=True,
               percentile=False, difference=False, map_img=False, origin='bottom', **kwargs):
    """
    Plots image with `plt.imshow()` with better defaults.

    Only required argument is img.

    Arguments
    ---------
    img
        Image to plot.

    title : str
        Image title. None for no title.

    colorbar : bool or str or dict
        Toggle colorbar display. Set tols str for the colorbar label or a dict
        to pass custom arguments to plt.colorbar().

    show : bool or None
        Choose if image is immediately shown to user. None will show if the
        current session is iPython.

    contours : bool or dict
        Toggle contour plot on image, set as dict to use as kwarg for
        plt.contour().

    ticks : bool
        Toggle x & y ticks.

    percentile : float
        Cap vmin/vmax at given percentile of the data.

    difference : bool or float
        Automatically produce image with RdBu_r etc. Set to value to define the
        midpoint of the data (otherwise will assume data is distributed around
        0).

    map_img : bool or str
        Toggle defaults for maps. Set to 'NW' to put compass directions in axis
        ticks or 'label' to label axes.

    origin : str
        Image origin location, passed to plt.imshow().

    Additional arguments passed to plt.imshow().

    Returns
    -------
    im : output of plt.imshow()
    """
    if show or show is None:
        plt.clf()

    if colorbar:
        if isinstance(colorbar, str):
            colorbar = {'label': colorbar}
        elif not isinstance(colorbar, dict):
            colorbar = {}

        if 'extend' not in colorbar:
            max_overflow = 'vmax' in kwargs and kwargs['vmax'] < np.nanmax(img)
            min_underflow = 'vmin' in kwargs and kwargs['vmin'] > np.nanmin(img)
            if max_overflow and min_underflow:
                colorbar['extend'] = 'both'
            elif max_overflow:
                colorbar['extend'] = 'max'
            elif min_underflow:
                colorbar['extend'] = 'min'
    if percentile:
        kwargs['vmin'] = np.nanpercentile(img, percentile)
        kwargs['vmax'] = np.nanpercentile(img, 100 - percentile)
        if colorbar is not False:
            colorbar['extend'] = 'both'
    if difference:
        if difference is True:
            difference = 0
        vm = np.nanmax(np.abs(img - difference))
        if percentile:
            vm = max(abs(kwargs.pop('vmin') - difference), abs(kwargs.pop('vmax') - difference))
        default_params = dict(cmap='RdBu_r', vmin=difference-vm, vmax=difference+vm)
        kwargs = {**default_params, **kwargs}
    if map_img:
        default_params = dict(extent=[360, 0, -90, 90])
        kwargs = {**default_params, **kwargs}
        xx = range(0, 360 + 1, 45)
        yy = range(-90, 90 + 1, 45)
        if map_img == 'nw':
            n_str = 'N'
            w_str = 'W'
        else:
            n_str = w_str = ''
        plt.xticks(ticks=xx, labels=['' if x % 90 else f'{x}°{w_str}' for x in xx])
        plt.yticks(ticks=yy, labels=['' if y % 90 else f'{y}°{n_str}' for y in yy])
        grid_style = dict(color='k', linestyle=':', linewidth=1, zorder=-1, alpha=0.1)
        for x in xx:
            plt.axvline(x, **grid_style)
        for y in yy:
            plt.axhline(y, **grid_style)
        if map_img == 'label':
            plt.xlabel('West Longitude')
            plt.ylabel('Latitude')

    im = plt.imshow(img, origin=origin, **kwargs)
    if title:
        plt.title(title)

    if colorbar is not False:
        plt.colorbar(**colorbar)

    if contours:
        default_params = dict(colors='k', linewidths=1, alpha=0.5)
        if 'extent' in kwargs:
            default_params['extent'] = kwargs['extent']
        if isinstance(contours, dict):
            default_params.update(**contours)
        plt.contour(img, **default_params)

    if not ticks:
        plt.xticks([])
        plt.yticks([])
    if show:
        plt.show()
    elif show is None:
        if test_if_ipython():
            plt.show()
        else:
            plt.pause(1e-3)
            plt.show(block=False)
    return im


def stack_images(*args, scale='independent', align=False, r_img=None, g_img=None, b_img=None,
                 h=0, s=1, h_range=1, plot=True, colorbar=None, colorbar_label='',
                 return_hues=False, legend=False, **kwargs):
    """
    Stack and optionally align images into multiple colour image.

    Arguments
    ---------
    *args : images to use for colour frames. Frames will have evenly distributed hues with rgb
        channels averaged over all frames. Image is then rescaled up to provide better contrast.

    r_img, g_img, b_img : images to use for red, green and blue frames. Set to None for black.
        Used to manually set colours rather than use automatic hue choice. If any *args are set,
        will add r, g, b to list of colours and ignore manual setting. Each channel is independent.

    scale : option to scale frames independently ('independent') to normalise flux, or to scale all
        frames by same ('same') amount to conserve relative flux.

    align : toggle image alignment. If images are not aligned, they must all be the same size.

    h : starting hue for first frame.

    s : saturation to use for all frames.

    h_range : range to use for h values.

    plot : toggle plotting of image, uses `show_image()`.

    colorbar : add colorbar of frame values. `colorbar` should be a list of values corresponding to
        image frames.

    colorbar_label : label to use for colorbar (if shown).

    return_hues : toggle returning of list of hues.

    legend : toggle showing legend.

    Additional arguments passed to `align_images()`.

    Returns
    -------
    Three colour image composed of input images.
    """
    if args and len(args) == 1:
        args = args[0]
    # Deal with None arguments
    if len(args) == 0:
        manual_colours = True
        img_list = [r_img, g_img, b_img]
    else:
        manual_colours = False
        img_list = list(args)
        for img in [r_img, g_img, b_img]:
            if img is not None:
                img_list.append(img)

    # Deal with rgb manually set
    shape_list = [(0, 0)]*len(img_list)
    shape = None
    dtype = None
    for idx in range(len(img_list)):
        # Get relevant info from images
        img = img_list[idx]
        if img is not None:
            shape_list[idx] = img.shape
            if dtype is None:
                dtype = img.dtype
            if shape is None:
                shape = img.shape
    for idx in range(len(img_list)):
        # Deal with blank images (from rgb arguments)
        if align and img_list[idx] is None:
            img_list[idx] = np.zeros(shape)

    if align:
        img_list = align_images(*img_list, **kwargs)
        shape = img_list[0].shape
    else:
        shape = [max([s[0] for s in shape_list]),
                 max([s[1] for s in shape_list])]
        # Replace None with black frames
        for idx in range(len(img_list)):
            if img_list[idx] is None:
                img_list[idx] = np.zeros((shape[0], shape[1]), dtype=dtype)

    img_list = np.array(img_list, dtype=dtype)

    # Scale to appropriate range
    for idx in range(len(img_list)):
        if scale == 'independent':
            max_v = np.nanmax(img_list[idx])
        else:
            max_v = np.nanmax(img_list)
        if max_v == 0:
            max_v = 1
        img_list[idx] = img_list[idx]/max_v

    # Change to coloured image from individual frames
    rgb_img = np.zeros((shape[0], shape[1], 3), dtype=dtype)
    if manual_colours:
        for idx in range(len(img_list)):
            rgb_img[:, :, idx] = img_list[idx]
        hue_list = [0, 0.3333333, 0.666666]
    else:
        hue_list = (h + np.array(range(len(img_list)))/len(img_list))*h_range
        hue_list %= 1 # Keep values in range 0 to 1
        for img, hue in zip(img_list, hue_list):
            hsv_img = np.zeros((shape[0], shape[1], 3), dtype=dtype)
            hsv_img[:, :, 0] = hue*np.ones(img.shape)
            hsv_img[:, :, 1] = s*np.ones(img.shape)
            hsv_img[:, :, 2] = img
            frame_rgb_img = hsv_to_rgb(hsv_img)
            rgb_img += frame_rgb_img/len(img_list)
        max_v = np.nanmax(rgb_img)
        rgb_img = rgb_img/max_v

    if plot:
        if not manual_colours and colorbar is not None:
            hsv = np.ones((len(hue_list), 3))
            hsv[:, 0] = hue_list
            hsv[:, 1] = s
            rgb = hsv_to_rgb(hsv)
            stack_cmap = matplotlib.colors.ListedColormap(rgb, name='stack_cmap')
            show_image(rgb_img, show=False, cmap=stack_cmap, colorbar=False)
            cmin = min(colorbar)
            cmax = max(colorbar)
            if len(colorbar) != len(hue_list):
                print('WARNING: different number of color values to number of frames')
            else:
                # Centre colorbar segments on the correct wavelength values
                dc = 0.5*(cmax-cmin)/(len(colorbar)-1)
                cmax += dc
                cmin -= dc
            plt.clim(cmin, cmax)
            if len(colorbar) < 10:
                plt.colorbar(label=colorbar_label, ticks=colorbar)
            else:
                plt.colorbar(label=colorbar_label)
        elif legend:
            show_image(rgb_img, show=False, colorbar=False)
            hsv = np.ones((len(hue_list), 3))
            hsv[:, 0] = hue_list
            hsv[:, 1] = s
            rgb = hsv_to_rgb(hsv)
            for c, l in zip(rgb, legend):
                plt.scatter(-10, -10, color=c, label=l, marker='s')
            plt.xlim(0)
            plt.ylim(0)
            plt.legend()

        else:
            show_image(rgb_img, show=False, colorbar=False)

    if return_hues:
        return rgb_img, hue_list
    else:
        return rgb_img


def ratio_images(img1, img2, ratio_type='both', plot_graph=True, normalise=True, clim=1,
                 align=True, **kwargs):
    """Calculate and plot ratio of two images.

    Arguments
    ---------
    img1, img2 : images to plot.

    ratio_type : type of ratio to perform on images.
        'both' produces scale from -1 to 1 with 0 for no difference.
        1 or 2 to define imge in numerator of division.

    plot_graph : toggle automatic graph plotting.

    normalise : toggle normalisation of flux for two images.

    clim : limits on colour axis in plot.

    align : toggle image alignment before plot.

    Additional arguments passed to `align_images()`

    Returns
    -------
    Ratio image.
    """
    if align:
        img1, img2 = align_images(img1, img2, **kwargs)

    if normalise:
        img1 = img1/img1.max()
        img2 = img2/img2.max()

    with np.errstate(divide='ignore', invalid='ignore'):
        if ratio_type == 1:
            ratio_img = img1/img2
        elif ratio_type == 2:
            ratio_img = img2/img1
        else:
            ratio_img = img1.copy()
            ratio_img[:] = 0
            ratio_img[np.where(img1 < img2)] = -(1-img1/img2)[np.where(img1 < img2)]
            ratio_img[np.where(img1 > img2)] = (1-img2/img1)[np.where(img1 > img2)]

    if plot_graph:
        plt.clf()
        if clim is None:
            plt.imshow(ratio_img, cmap='RdBu', origin='bottom')
        else:
            plt.imshow(ratio_img, cmap='RdBu', origin='bottom', vmin=-clim, vmax=clim)
        if clim is None or (clim == 1 and ratio_type == 'both'):
            plt.colorbar(label='Frame ratio')
        else:
            plt.colorbar(extend='both', label='Frame ratio')
        plt.show()

    return ratio_img


def grid_images(*images, w=None, h=None, ratio=1.4, colorbar=True, auto_clim=True,
                number_plots=True, title_plots=False, **kwargs):
    """Plots grid of images using `show_image()`

    Arguments
    ---------
    *images : Series of image arrays to plot as image grid.

    w, h, ratio : Define either width, height or w/h ratio of image grid used in plotting.

    colorbar : Toggle if common colorbar is provided for all plots. If `colorbar` is a dict, it is
        passed as as argument to `plt.colorbar(..., **colorbar)`.

    auto_clim : Toggle if colour limits are automatically chosen from data limits. Overriden by
        `vmin` and `vmax` in `**kwargs`.

    number_plots : Toggle if plots are numbered in bottom left corner. If `number_plots` is a dict,
        it is passed as an argument to `plt.annotate(..., **number_plots)`. If `number_plots` is
        not iterable, uses each value for label.

    title_plots : Titles for plots.

    Additional keywords are passed to `show_image()`.
    """
    images = np.array(images)
    if len(images) == 1 and images.ndim == 4:
        # Deal with list provided rather than seperate image arrays
        images = images[0]

    n_images = len(images)

    if auto_clim:
        if 'vmin' not in kwargs:
            kwargs['vmin'] = np.nanmin([np.nanmin(img) for img in images])
        if 'vmax' not in kwargs:
            kwargs['vmax'] = np.nanmin([np.nanmax(img) for img in images])

    if w:
        h = np.ceil(n_images/w)
    elif h:
        w = np.ceil(n_images/h)
    else:
        w = np.sqrt(n_images)*ratio
        h = np.ceil(n_images/w)

    w, h = int(w), int(h)

    plt.clf()
    for idx, img in enumerate(images):
        plt.subplot(h, w, idx+1)
        show_image(img, show=False, colorbar=False, **kwargs)
        plt.xticks([])
        plt.yticks([])
        if title_plots:
            plt.title(title_plots[idx])
        if number_plots:
            if isinstance(number_plots, dict):
                plt.annotate(str(idx), (5, 5), **number_plots)
            else:
                try:
                    value = str(number_plots[idx])
                except TypeError:
                    value = str(idx)
                plt.annotate(value, (5, 5))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    if colorbar:
        if not isinstance(colorbar, dict):
            colorbar = {}
        fig = plt.gcf()
        fig.subplots_adjust(right=0.90)
        cbar_ax = fig.add_axes([0.905, 0.05, 0.01, 0.9])
        plt.colorbar(cax=cbar_ax, **colorbar)


def add_fig_metadata(msg=''):
    """
    Adds useful information to corner of figure.

    Arguments
    ---------
    msg : custom message to add to datetime in metadata.
    """
    msg += str(datetime.now())
    plt.figtext(0.995, 0.995, msg, alpha=0.5, size='x-small', family='monospace',
                horizontalalignment='right', verticalalignment='top')


#%% IMAGE CLEANING --------------------------------------------------------------------------------

def replace_bad_pixels(raw_img, bpm_img=None, fill_bad_regions=False, print_progress=False):
    """
    Clean image by replacing bad pixels in image.

    Bad pixels are replaced by median of surrounding pixels. NaN values are assumed to be bad
    pixels. Images with large bad pixel regions can be cleaned with `fill_bad_regions=True`. This
    repeatedly calls `replace_bad_pixels()` so that the pixels on the edge of the bad region are
    progressively fixed each call.

    Arguments
    ---------
    raw_img : image to clean.

    bpm_img : map of bad pixels in image, bad pixels indicated by True/1, good by False/0.

    fill_bad_regions : toggle if agressive replacement is used to fill whole regions of bad pixels.
        Set to a number to limit number of iterations of replacement.

    print_progress : toggle printing of progress bar.

    Returns
    -------
    img : image with bad pixels replaced by median value of surrounding pixels.
    """
    img = np.array(raw_img.copy())
    if bpm_img is not None:
        img[np.where(bpm_img)] = np.nan

    if np.all(np.isnan(img)):
        print('WARNING: No good pixels found in `replace_bad_pixels()`')
        print('   Image cleaning not possible, returning original image')
        return raw_img

    nan_locations = np.argwhere(np.isnan(img))
    start_bp = np.sum(np.isnan(img))
    orig_img = img.copy() # Want to get values from uncleaned image, not partially cleaned one

    for idx in range(len(nan_locations)):
        if print_progress and (idx % 5e4 == 0 or idx == len(nan_locations)-1):
            progress_bar(idx/(len(nan_locations)-1),
                         label=f'Replacing {len(nan_locations)} bad px')
        x_idx = nan_locations[idx][0]
        y_idx = nan_locations[idx][1]
        x_start = x_idx - 1
        x_end = x_idx + 2
        y_start = y_idx - 1
        y_end = y_idx + 2

        # Deal with pixels on edge of image
        if x_start < 0:
            x_start = 0
        if y_start < 0:
            y_start = 0

        values = orig_img[x_start:x_end, y_start:y_end]
        if not np.all(np.isnan(values)):
            replacement_value = np.nanmedian(values)
            img[x_idx, y_idx] = replacement_value

    if np.any(np.isnan(img)):
        if fill_bad_regions:
            if fill_bad_regions is not True:
                fill_bad_regions -= 1
                if fill_bad_regions < 0:
                    fill_bad_regions = 0
            return replace_bad_pixels(img, fill_bad_regions=fill_bad_regions,
                                      print_progress=print_progress)
        else:
            end_bp = np.sum(np.isnan(img))
            print(f'WARNING: {end_bp} bad pixels remaining in image after replace_bad_pixels()), '
                  f'{start_bp} at start')
    return img


def exp_despike(raw_image, nsigma=5, replace_nan=True):
    """
    Removes bad pixels by calculating median of surrounding pixels and SD of image.

    Code based on original IDL script at `/data/nemesis/drm/lib/exp_despike.pro`. Bad pixels
    identified by being `nsigma` from median of surrounding pixels and are replaced by that
    median value.

    Simular to `sigma_filter`.

    Arguments
    ---------
    raw_image : image to despike.

    nsigma : number of standard deviations to define bad pixel.

    replace_nan : toggle to replace nan values.

    Returns
    -------
    Image with bad pixel values replaced.
    """
    image = raw_image.copy()

    # Replace nan values with median of nearby non-nan values
    if replace_nan:
        image = replace_bad_pixels(image)

    # Determine noise level in image by subtracting median image & finding SD of all pixels
    med_image = median_filter(image, 3, mode='reflect')
    diff_image = image - med_image
    sigma = np.nanstd(diff_image)
    bad_pixels = np.where(abs(diff_image) > nsigma*sigma)

    # Replace bad pixels with median value
    for idx in range(len(bad_pixels[0])):
        x_idx = bad_pixels[0][idx]
        y_idx = bad_pixels[1][idx]
        image[x_idx, y_idx] = med_image[x_idx, y_idx]
    return image


def sigma_filter(raw_img, nsigma=5, radius=1, iterate=False, print_summary=False):
    """
    Removes bad pixels by calculating mean and variance of surrounding pixels.

    Based on IDL code at https://idlastro.gsfc.nasa.gov/ftp/pro/image/sigma_filter.pro. Calculates
    mean and variance of pixels surrounding each pixel, and replaces pixels which have a larger
    `variance*nsigma` than their surrounding pixels. Pixels are replaced with the mean value of the
    surrounding pixels.

    Similar to `exp_despike`.

    Arguments
    ---------
    raw_img : image to filter.

    nsigma : threshold number of standard deviations to use when determining if bad pixel.

    radius : size of side of box to calculate averages.

    iterate : toggle recursive cleaning until no pixels are changed. Set to True for default max.
        of 20 iterations, or set to number for manual limit of iterations.

    print_summary : toggle printing of filter result.

    Returns
    -------
    img : filtered image.
    """
    if iterate is True:
        iterate = 20
    if iterate:
        iterate = int(iterate)
    img = np.array(raw_img.copy())
    shape = img.shape
    if len(shape) > 2:
        return [sigma_filter(frame, nsigma=nsigma, radius=radius,
                             iterate=iterate, print_summary=print_summary) for frame in img]
    box_width = 2*radius + 1
    box_size = box_width**2
    avg_img = (uniform_filter(img, box_width, mode='reflect')*box_size - img)/(box_size - 1)
    dev_img = (img - avg_img)**2
    fact = nsigma**2/(box_size-2)
    var_img = fact*(uniform_filter(dev_img, box_width, mode='reflect')*box_size - dev_img)
    replace_idx = np.where(dev_img > var_img)
    img[replace_idx] = avg_img[replace_idx]

    n_repl = len(replace_idx[0])
    if print_summary:
        n_pix = img.size
        pct = n_repl/n_pix
        if iterate is not False:
            itr = ', max {} loops left'.format(int(iterate))
        else:
            itr = ''
        print(f'{pct:.2%} of px replaced by sigma_filter{itr} (replaced {n_repl:d})')
    if iterate and n_repl:
        iterate -= 1
        return sigma_filter(img, nsigma=nsigma, radius=radius, iterate=iterate,
                            print_summary=print_summary)
    return img


def cosmetics_clean_bp(img, flat_img, bad_img, bp_threshold, **kwargs):
    """
    Perfoms cosmetic cleaning of bad pixels in image.

    Based on nacoadicube_cosmetics.pro IDL code.

    Arguments
    ---------
    img : input image cube.

    flat_img : flat file.

    bad_img : badpixel map.

    bp_threshold : threshold to flag bad pixels from the flat.

    Returns
    -------
    cubecln : cleaned image cube.
    """
    img = np.array(img)
    flat_img = np.array(flat_img)
    bad_img = np.array(bad_img)
    cube = img/flat_img
    bad_img[np.where((flat_img <= (1-bp_threshold)) | (flat_img >= (1+bp_threshold)))] = 0
    imtemp = cube.copy()
    nimtemp = maskinterp(imtemp, bad_img, 1, 6, gpix=10, gpoints=5, cdis=2)
    cubecln = sigma_filter(nimtemp, radius=1, nsigma=3, iterate=True)
    return cubecln


def maskinterp(rawdata, m, apstep=1, maxap=6, gpix=20, gpoints=4, cdis=3):
    """
    Replace bad pixels in image with interpolated values.

    Based on maskinterp.pro IDL code.
    Each bad pixel is fixed using only data from neighboring
    good pixels.  A circular aperture centered on the bad
    pixel determines the neighboring pixels to be used in the
    interpolation.  The aperture radius starts at zero and
    increases on successive iterations by an aperture step up
    to a maximum radius.  Since the number and the locations
    of the interpolating pixels vary, we set certain conditions
    for fixing the bad pixel.  The pixel is fixed only if (1)
    there is a sufficient number of good pixels,(2) the good
    pixels are not weighted to one side of the aperture, and
    (3) the conditions in the interpolating function are
    satisfied.  The method of determining the second condition
    is the calculation of the center of mass.  A good pixel is
    weighted by 1 and the bad pixel is weighted by 0.  The user
    specifies the maximum allowed distance between the center
    of mass and the center of the aperture, the minimum number
    of good pixels, and the minimum percentage of good pixels
    within the aperture.

    Arguments
    ---------
    rawdata : data array.

    m : mask indicating good (1) and bad (0) pixels in rawdata.

    apstep : aperture step size when gradually increading aperture radius.

    maxap : maximum aperture radius allowed.

    gpix : minimum percentage of good pixels allowed in aperture to allow interpolation.

    gpoints : minimum number of good pixels allowed in aperture to allow interpolation.

    cdis : maximum distance of centre of mass of good pixels from centre of aperture.

    Returns
    -------
    im : image with cleaned bad pixels.
    """
    rawdata = np.array(rawdata)
    m = np.array(m)
    info = rawdata.shape
    xdim = info[0]
    ydim = info[1]

    # Create a new mask which is the old mask padded with 0s on the sides
    minfo = m.shape
    mxdim = minfo[0] + 2*maxap
    mydim = minfo[1] + 2*maxap
    mask = np.zeros((mxdim, mydim))
    mask[maxap:-maxap, maxap:-maxap] = m

    # Create a new data matrix which is rawdata padded with 0s on the sides
    im0 = np.zeros((mxdim, mydim))
    im0[maxap:-maxap, maxap:-maxap] = rawdata
    fmask = np.zeros((xdim, ydim))
    for ap in range(apstep, maxap+apstep, apstep):
        # Create aperture mask
        rad = int(np.ceil(ap))
        apmask = create_dist_circle(1+2*rad)
        apmask = apmask <= ap
        # Find number of good pixels in aperture mask
        appts = np.sum(apmask)
        bad_px = np.where((m == 0) & (fmask == 0))
        for idx1, idx2 in zip(bad_px[0]+maxap, bad_px[1]+maxap):
            umask = mask[idx1-rad:idx1+rad+1, idx2-rad:idx2+rad+1]*apmask
            if disc_conditions(umask, appts, gpix, gpoints, cdis):
                # Only clean if disc conditions are met
                im_in = im0[idx1-rad:idx1+rad+1, idx2-rad:idx2+rad+1]
                out = csplinterp(im_in, umask)
                im0[idx1, idx2] = out[0]
                fmask[idx1-maxap, idx2-maxap] = out[1]
        if np.sum(fmask[np.where(m == 0)]) == np.sum(m == 0):
            break
    im = im0[maxap:-maxap, maxap:-maxap]
    return im


def create_dist_circle(n, xcen=None, ycen=None):
    """
    Forms a square array where each value is the distance to the given centre.

    Based on dist_circle.pro IDL code

    Arguments
    ---------
    n : size of output (scalar or list like to specify x and y dimensions).

    xcen, ycen : centres of circle, `None` uses centre of array.

    Returns
    -------
    im : image with values of distance to centre of circle.
    """
    if np.isscalar(n):
        nx = ny = n
    else:
        nx = n[0]
        ny = n[1]
    nx = int(nx)
    ny = int(ny)
    if xcen is None:
        xcen = (nx - 1)/2
    if ycen is None:
        ycen = (ny - 1)/2
    x2 = (np.arange(nx) - xcen)**2
    y2 = (np.arange(ny) - ycen)**2
    im = np.zeros((nx, ny))
    for idx in range(ny):
        im[idx, :] = np.sqrt(x2 + y2[idx])
    return im


def disc_conditions(umask, appts, goodpix, goodpoints, cdis):
    """
    Determine if region of raw datai is a good candidate for interpolation.

    Based on disc.pro IDL code.

    Arguments
    ---------
    umask : matrix of mask values, 1=good, 0=bad.

    appts : number of good points in aperture mask.

    goodpix : minimum percentage of good pixels.

    goodpoints : miniumum number of good pixels.

    cdis : maximum distance from centre of mass of good pixels to aperture centre.

    Returns
    -------
    boolean indicating if disc is good candidate for interpolation
    """
    pos = np.array(np.where(umask == 1))
    points = len(pos[0])
    if points < goodpoints or points == 0:
        return False # Short circuit to avoid 0/0 errors later
    percent = 100*points/appts
    if percent < goodpix:
        return False
    info = np.array(umask).shape
    xdim = info[0]
    ydim = info[1]
    xo = np.floor(xdim/2)
    yo = np.floor(ydim/2)
    # center of mass
    xcm = np.mean(pos[0]) - xo
    ycm = np.mean(pos[1]) - yo
    dist = np.sqrt(xcm**2 + ycm**2)
    return dist <= cdis


def csplinterp(raw, mask):
    """
    Use spline interpolation to replace bad pixel value in image.

    Based on csplinterp.pro IDL code.

    Arguments
    ---------
    raw : raw image to interpolate.

    mask : mask of good and bad pixels, 1=bad, 0=good.

    Returns
    -------
    value, success : value of central pixel and boolean indicating if it has been successfully
        interpolated.
    """
    raw = np.array(raw)
    # Determine position of centered bad pixel
    info = raw.shape
    xdim = info[0]
    ydim = info[1]
    xo = int(xdim/2)
    yo = int(ydim/2)
    # Require minimum number of points for interpolation
    row = np.sum(mask, axis=0)
    col = np.sum(mask, axis=1)
    if row[xo] >= 3 and col[yo] >= 3:
        # Interpolate row then column
        if min(row[xo], col[yo]) <= 3:
            k = 1 # Use more basic spline for smaller aperture
        else:
            k = 3
        x = np.squeeze(np.where(mask[:, yo] == 1))
        z = np.squeeze(raw[x, yo])
        tck = scipy.interpolate.splrep(x, z, k=k)
        val1 = scipy.interpolate.splev(xo, tck)
        y = np.squeeze(np.where(mask[xo, :] == 1))
        z = np.squeeze(raw[xo, y])
        tck = scipy.interpolate.splrep(y, z, k=k)
        val2 = scipy.interpolate.splev(yo, tck)
        return (val1+val2)/2, True
    else:
        return raw[xo, yo], False


def fourier_mask(img, mask='threshold', mask_mode='blank', mask_val=0, fftshift=True,
                 abs_output=True, **kwargs):
    """
    Applies mask to image fourier transform and converts image back to real space.

    Arguments
    ---------
    img : image to apply fourier mask to.

    mask : mask value to apply to fourier transformed image.

    mask_mode : type of masking to perform.

    mask_val : value to use to mask FFT.

    fftshift : toggle to choose if fftshift is applied to image before masking.

    abs_output : toggle if output is converted from complex numbers.

    Additional arguments used in `threshold` masking calculations.

    Returns
    -------
    img : cleaned image.
    """
    if len(img.shape) == 3:
        # Deal with image cubes
        output = []
        for idx in range(len(img)):
            output.append(fourier_mask(img[idx], mask=mask, mask_mode=mask_mode,
                          fftshift=fftshift, abs_output=abs_output, **kwargs))
        return output

    if isinstance(mask, str):
        # Deal with two argument case
        mask_mode = mask

    img = img.copy()
    img = np.fft.fft2(img)
    if fftshift:
        img = np.fft.fftshift(img)

    if mask_mode == 'multiply':
        # Multiply the FT and mask together
        img *= mask
    else:
        if mask_mode == 'threshold':
            # Use threshold calculation to identify regions needed to blank
            params = {'threshold_val': 95,
                      'blur_radius': 1,
                      'safe_radius': 25}
            params.update(kwargs)
            c = np.array((np.array(img.shape)/2), dtype=int) # Image centre coordinates
            abs_img = np.abs(img)
            ref_img = scipy.ndimage.filters.gaussian_filter(abs_img, params['blur_radius'])
            threshold = np.percentile(ref_img, params['threshold_val'])
            mask_pos = np.where(ref_img > threshold)
            masks = [[], []]
            radii = np.sqrt((mask_pos[0] - c[0])**2 + (mask_pos[1] - c[1])**2)
            for ii in range(len(mask_pos[0])):
                mask = mask_pos[0][ii], mask_pos[1][ii]
                if radii[ii] <= params['safe_radius']:
                    continue # Don't blank pixels in centre of image
                masks[0].append(mask[0])
                masks[1].append(mask[1])
            mask = masks

        if np.array(mask).shape == img.shape:
            mask = np.where(mask) # Change from 2D mask of image to array of indices

        if len(mask[0]) > 0:
            if mask_val in ['interpolate', 'interp']:
                # Replace blanked values with interpolated values
                img[mask[0], mask[1]] = np.nan
                x = np.arange(0, img.shape[1])
                y = np.arange(0, img.shape[0])
                array = np.ma.masked_invalid(img)
                xx, yy = np.meshgrid(x, y)
                x1 = xx[~array.mask]
                y1 = yy[~array.mask]
                newarr = array[~array.mask]
                img = scipy.interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy),
                                                 method='linear')
            else:
                img[mask[0], mask[1]] = mask_val
    if fftshift:
        img = np.fft.ifftshift(img)
    img = np.fft.ifft2(img)

    if abs_output:
        img = np.abs(img)
    return img


def create_mask(shape, coords, reflect=True, blur=False, fractional_coords=False):
    """
    Creates mask from given coordinates and radii to use in fourier masking.

    Arguments
    ---------
    shape : shape of image to mask.

    coords : list of coordinates to use in mask. Coordinates are of form `[x,y]` or `[x,y,r]` where
        `x` and `y` are the x and y indices of the location to blank. `r` is the radius around the
        location to blank. If `r` is not given, radius of 1 pixel is assumed. All values can be
        integers or floats.

    reflect : toggle if coordinates are reflected about centre of image. Best for removing bands
        from fourier transform and keeping output of masked FT (close to) real values.

    blur : optionally apply gaussian blur to mask.

    fractional_coords : give coords as fraction of image size, radius is scaled by mean of x & y
        size.

    Returns
    -------
    mask : np array in shape of image with values ranging from 0 (fully blanked) to 1 (not blanked)
    """
    mask = np.ones(shape)
    if len(coords) == 0:
        return mask
    x = np.arange(0, shape[1])
    y = np.arange(0, shape[0])
    xx, yy = np.meshgrid(x, y)

    if fractional_coords:
        coords = np.array(coords)
        coords[:, 0] *= shape[1]
        coords[:, 1] *= shape[0]
        coords[:, 2] *= np.mean(shape)
    for c in coords:
        if len(c) == 2:
            c.append(1)
        c_mask = (np.sqrt((xx-c[0])**2 + (yy-c[1])**2) - abs(c[2]))*np.sign(c[2])
        # Anti-aliase edge of circle for smoother transition between blanked/non-blanked regions
        aa_offset = 0.5
        c_mask[np.where(c_mask > aa_offset)] = aa_offset
        c_mask[np.where(c_mask < -aa_offset)] = -aa_offset
        c_mask = (c_mask + aa_offset)/(2*aa_offset)
        mask = np.minimum(c_mask, mask)

    if reflect:
        mask = np.minimum(mask, np.flip(mask))

    if blur:
        mask = scipy.ndimage.filters.gaussian_filter(mask, blur)

    return mask


#%% ANALYSIS --------------------------------------------------------------------------------------

def get_radial_dependence(img, centroid=None, return_r_list=True, return_areas=False,
                          annuli=True):
    """
    Calculates radial brightness function around given centroid in image.

    Arguments
    ---------
    img : image to calculate brightness from.

    centroid : coordinates of centre of radial function.

    return_r_list : toggle returning of r list as well as value list.

    return_areas : toggle returning of areas of annuli.

    annuli : toggle finding annuli or circle brightness.

    Returns
    -------
    val_list : list of brightness densities at different radii.
    """
    img = np.array(img)
    if centroid is None:
        img_masked = img.copy()
        img_masked[np.isnan(img_masked)] = np.nanmin(img)
        img_masked -= np.nanmin(img)
        centroid = photutils.centroids.centroid_2dg(img_masked)

    centroid = np.array(centroid)
    r_ceil = int(min(*centroid, *(img.shape-centroid)))
    r_list = np.array(range(1, r_ceil+1))
    if annuli:
        apertures = [photutils.aperture.CircularAnnulus(centroid, r-1, r) for r in r_list]
        apertures[0] = photutils.aperture.CircularAperture(centroid, r_list[0]) # Avoid /0 error
    else:
        apertures = [photutils.aperture.CircularAperture(centroid, r) for r in r_list]

    val_list = []
    for aperture in apertures:
        table = photutils.aperture.aperture_photometry(img, aperture)
        aperture_sum = float(table['aperture_sum'])
        val_list.append(aperture_sum/aperture.area())
    val_list = np.array(val_list)
    area_list = np.array([ap.area() for ap in apertures])
    if return_r_list and return_areas:
        return r_list, val_list, area_list
    elif return_r_list:
        return r_list, val_list
    elif return_areas:
        return val_list, area_list
    else:
        return val_list


def sum_calstar(cube, r_max=None, visualise=False, wavelengths=False, **kwargs):
    """
    Calculate calibration star spectrum using `star_aperture()` on each frame.

    Identifies star centroid from collapsed image of a sum of all frames and then sums the
    brightness (and background etc.) independently for each frame.

    Arguments
    ---------
    cube : image cube to process.

    r_max : maximum radius (px) to pass to `star_aperture()`.

    visualise : toggle visualisation of result.

    wavelengths : toggle wavelength dependent r_max

    **kwargs
        Additional arguments passed to star_aperture().

    Returns
    -------
    sum_values : list of values giving spectrum of calibration star.
    """
    sum_img = np.sum(cube, axis=0)
    with warnings.catch_warnings():
        # Supress warning about masked data
        warnings.filterwarnings('ignore',
                                message='Input data contains input values (e.g. NaNs or infs)*',
                                module='photutils.centroids.core'
                                )
        centroid = photutils.centroids.centroid_2dg(sum_img)
    sum_values = []
    bg_values = []
    if wavelengths is False:
        wavelengths = np.ones(len(cube))
    for img, wl in zip(cube, wavelengths):
        r = r_max*wl/wavelengths[0]
        sum_value, bg_value = star_aperture(img, centroid=centroid, r_max=r, **kwargs)
        sum_values.append(sum_value)
        bg_values.append(bg_value)

    if visualise:
        idx_list = range(len(sum_values))
        plt.clf()
        plt.subplot(1, 2, 1)
        show_image(sum_img, show=False)
        kwargs = {'color': 'r', 'linewidth': 0.5}
        plt.axvline(x=centroid[0], **kwargs)
        plt.axhline(y=centroid[1], **kwargs)
        plt.title('Collapsed image with centroid marked by crosshairs')
        plt.subplot(1, 2, 2)
        plt.plot(idx_list, sum_values)
        plt.scatter(idx_list, sum_values, marker='.')
        plt.ylim(bottom=0)
        plt.title('Calculated spectrum')
        plt.xlabel('Frame index')
        plt.ylabel('Brightness')
        plt.suptitle(f'Ratio between brightest/darkest part of spectrum = '
                     f'{max(sum_values)/min(sum_values):.3f} with r_max = {r_max}px')
        plt.show(block=False)
    return np.array(sum_values), np.array(bg_values)


def star_aperture(img, centroid=None, r_max=None, r_bg=None, visualise=False, fix_r=False,
                  nan_check=None):
    """
    Sum brightness of star image after removing background.

    Based on aper.pro IDL code. Sums brightness in circular apertures around centroid and
    identifies brightest aperture to calculate total brightness from star. Background identified by
    summing brightness from annulus around star.

    Arguments
    ---------
    img : image to sum.

    centroid : location of star, `None` automatically locates centroid in image.

    r_max : maximum radius to use for aperture.

    r_bg : list/tuple of radii defining annulus to calculate background value.

    visualise : toggle plotting of result visualisation.

    fix_r : bool
        Toggle fixing of radius such that the annulus used for the calculation is r_max.

    nan_check : bool
        Toggle checking for NaN values in image. This step is slow, so if NaNs
        are already removed, can set nan_check=False for faster code.

    Returns
    -------
    aperture_sum : value of total brightness in aperture (note this is a float as the aperture is
          usually composed of a non-integer number of pixels).
    """
    img = np.array(img)
    if nan_check is None:
        nan_check = r_max is None

    if centroid is None:
        centroid = photutils.centroids.centroid_2dg(img)
    centroid = np.array(centroid)
    # Max allowed radius from centroid to edge
    r_ceil = int(min(*centroid, *(img.shape-centroid)))

    if nan_check:
        # Prevent issues from nan values in aperture calculation, this value has no effect on final
        # output as r_ceil prevents any of these pixels being included in any calculation. This
        # step can be very slow, so skip it for cases where nan locations are known and are daelt
        # with otherwise.
        if np.any(np.isnan(img)):
            img_mask = np.isnan(img)
            for r in range(1, r_ceil+1):
                aperture = photutils.aperture.CircularAperture(centroid, r)
                table = photutils.aperture.aperture_photometry(img_mask, aperture)
                aperture_sum = float(table['aperture_sum'])
                if aperture_sum:
                    r_ceil = r - 1
                    break
            img[np.isnan(img)] = -999

    if isinstance(r_max, (list, tuple, np.ndarray)):
        r_list = r_max
    else:
        if r_max is None:
            r_max = r_ceil
        if not fix_r:
            r_list = range(1, r_max + 1)
    if fix_r:
        r_list = [r_max]

    if max(r_list) > r_ceil:
        print(f'Only including radii <= {r_ceil}')
        r_list = [r for r in r_list if r <= r_ceil] # Ensure values aren't off edge of image
    if r_bg is None:
        if fix_r:
            r_bg = [r_max, r_max+10]
        else:
            r_bg = [r_list[int(len(r_list)*2/3)], r_list[-1]]
    bg_aperture = photutils.aperture.CircularAnnulus(centroid, *r_bg)
    bg_table = photutils.aperture.aperture_photometry(img, bg_aperture)
    bg_val = float(bg_table['aperture_sum']/bg_aperture.area())
    img -= bg_val
    star_apertures = [photutils.aperture.CircularAperture(centroid, r) for r in r_list]

    phot_list = []
    for aperture in star_apertures:
        table = photutils.aperture.aperture_photometry(img, aperture)
        phot_list.append(float(table['aperture_sum']))
    aperture_sum = max(phot_list)
    r_sum = r_list[phot_list.index(aperture_sum)]

    if visualise:
        plt.clf()
        ax = plt.subplot(1, 2, 1)
        img += bg_val
        img[np.where(img == -999)] = np.nan # Put nan values back in for plotting purposes
        show_image(img, show=False)
        for r in r_list:
            if r == r_sum:
                circle = plt.Circle(centroid, r, color='r', fill=False, linewidth=1)
            else:
                circle = plt.Circle(centroid, r, color='w', fill=False, linewidth=0.25)
            ax.add_artist(circle)
        for r in r_bg:
            circle = plt.Circle(centroid, r, color='c', fill=False, linestyle='--')
            ax.add_artist(circle)
        plt.subplot(1, 2, 2)
        plt.plot(r_list, phot_list, marker='.', c='k', label='Brightness within circular aperture')
        plt.scatter(r_sum, aperture_sum, c='r')

        plt.axhline(y=aperture_sum, color='r', zorder=0, label='Brightest circular aperture')
        plt.axvline(x=r_sum, color='r', zorder=0)
        for r in r_bg:
            if r == r_bg[0]:
                ll = 'Background annulus'
            else:
                ll = None
            plt.axvline(x=r, color='c', linestyle='--', label=ll)
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.legend(loc='lower center')
        plt.xlabel('Radius (px)')
        plt.ylabel('Brightness')
        plt.suptitle(f'Maximum brightness found at r={r_sum}px with count of {aperture_sum:.1f} '
                     f'and background={bg_val:.1f}')
        plt.show(block=False)
    return aperture_sum, bg_val


#%% MANIPULATION ----------------------------------------------------------------------------------

def interp_image(img, value, method='linear'):
    """
    Interpolate image to higher resolution using spline interpolation.

    Simular to `rebin()`.

    Arguments
    ---------
    img : image to interpolate. If img is a cube, interps each frame
        separately.

    value : scaling value to increase image size.

    method : interpolation method.

    Returns
    -------
    img : image scaled to higher resolution.
    """
    if not value or value == 1:
        return img
    nan_img = np.any(np.isnan(img))
    if nan_img:
        img = np.array(img)
        mask = np.isnan(img) # Deal with nans in image
        img[mask] = 0
    shape = img.shape
    if len(shape) > 2:
        return np.array([interp_image(frame, value, method) for frame in img])
    shp_y, shp_x = shape
    x1 = np.linspace(0, 1, shp_x)
    y1 = np.linspace(0, 1, shp_y)
    x2 = np.linspace(0, 1, round(shp_x*value))
    y2 = np.linspace(0, 1, round(shp_y*value))
    img = interp2d(x1, y1, img, kind=method)(x2, y2)
    if nan_img:
        mask = interp_image(mask, value, method)
        mask = np.ceil(mask).astype(bool)
        img[mask] = np.nan
    return img


def rebin(image, factor, function=None):
    """
    Function to rebin image to higher resolution.

    Scaling factors must be integers and no interpolation is performed.
    Similar to `interp()`.

    Arguments
    ---------
    image : image to scale up

    factor : factor to scale image

    function : function to call to rebin image when downscaling. None uses a
        combination of np.mean and np.nanmean as appropriate.

    Returns
    -------
    Image scaled to higher resolution
    """
    if factor == 1:
        return image
    image = np.array(image)
    new_shape = [int(s) for s in factor*(np.array(image.shape))]
    new_image = np.zeros(new_shape)

    x_arr = range(new_shape[0])
    x1_arr = np.array([int(x/factor) for x in x_arr])
    x2_arr = x1_arr + int(np.ceil(1/factor))
    y_arr = range(new_shape[1])
    y1_arr = np.array([int(y/factor) for y in y_arr])
    y2_arr = y1_arr + int(np.ceil(1/factor))

    for x_idx, x1, x2 in zip(x_arr, x1_arr, x2_arr):
        for y_idx, y1, y2 in zip(y_arr, y1_arr, y2_arr):
            if factor > 1:
                new_image[x_idx, y_idx] = image[x1, y1]
            elif function is None:
                # np.nanmean is slow, so try using faster np.mean function first.
                new_image[x_idx, y_idx] = np.mean(image[x1:x2, y1:y2])
                if np.isnan(new_image[x_idx, y_idx]) and np.sum(1 - np.isnan(image[x1:x2, y1:y2])):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        new_image[x_idx, y_idx] = np.nanmean(image[x1:x2, y1:y2])
            else:
                new_image[x_idx, y_idx] = function(image[x1:x2, y1:y2])

    return new_image


def align_images(*args, pad_value=0, **kwargs):
    """
    Aligns multiple images so that their centres of mass are in the same position.

    Images are padded as necessary so that both images are the same shape and the centre of the
    images are aligned.
    Similar to `align_cubes()`.

    Arguments
    ---------
    *args : images to align.

    pad_value : value to use when extending image.

    Additional arguments passed to `centre_image()`.

    Returns
    -------
    Tuple of images which are now aligned.
    """
    img_list = []
    shape_list = []
    args = np.array(args)
    if len(args) == 1 and len(args[0].shape) > 2:
        args = args[0]

    for img in args:
        img = center_image(img, pad_value=pad_value, **kwargs)
        img_list.append(img)
        shape_list.append(np.array(img.shape))

    shape_final = [max([s[0] for s in shape_list]),
                   max([s[1] for s in shape_list])]

    # Find needed offset for each image to align
    offset_list = []
    for shape in shape_list:
        offset_list.append(((shape_final - shape)/2).astype(int))

    output_list = []
    for idx in range(len(img_list)):
        img_aligned = np.zeros(shape_final, dtype=img_list[idx].dtype)
        img_aligned[:] = pad_value
        img_aligned[offset_list[idx][0]:offset_list[idx][0]+shape_list[idx][0],
                    offset_list[idx][1]:offset_list[idx][1]+shape_list[idx][1]] = img_list[idx]
        output_list.append(img_aligned)

    return tuple(output_list)


def align_cubes(cube_list, pad_value=0, interp=False):
    """
    Align multiple image cubes so the centre of mass of the cubes are in the same position.

    Images are padded as necessary with `pad_value` and scaled up using `interp`.
    Similar to `align_images()`.

    Arguments
    ---------
    cube_list : list of image cubes to align.

    pad_value : value to use when extending image.

    interp : set interpolation scale used when increasing image size.

    Returns
    -------
    Tuple of cubes which are now aligned.
    """
    offset_list = []
    if interp:
        cube_list = interp_image(cube_list, interp)
    for cube in cube_list:
        img = np.sum(cube, axis=0)
        threshold_img = img.copy()
        threshold = 0.5*sum([np.percentile(threshold_img, 10),
                             np.percentile(threshold_img, 90)])
        threshold_img[np.where(threshold_img <= threshold)] = 0
        threshold_img[np.where(threshold_img > threshold)] = 1

        com = np.array(center_of_mass(threshold_img))
        offset = np.array(img.shape)/2 - com
        offset = np.around(offset).astype(int)
        offset_list.append(offset)

    offset_list = np.array(offset_list)
    offset_list -= np.min(offset_list, axis=0)
    difference_list = np.abs(np.max(offset_list, axis=0) - np.min(offset_list, axis=0))
    if max(difference_list) == 0:
        return cube_list
    shape_final = img.shape + difference_list

    cube_aligned_list = []
    for cube, offset in zip(cube_list, offset_list):
        cube = np.array(cube)
        cube_aligned = np.zeros((len(cube), *shape_final), dtype=cube.dtype)
        cube_aligned[:] = pad_value
        shape = cube[0].shape
        cube_aligned[:, offset[0]:offset[0]+shape[0], offset[1]:offset[1]+shape[1]] = cube
        cube_aligned_list.append(cube_aligned)

    return tuple(cube_aligned_list)


def center_image(img, pad_value=0, threshold='auto', interp=10):
    """
    Shift image so `centre of mass` of image is at centre.

    Centre of mass found after applying threshold to identify target and background. Image is
    padded with `pad_value` to extend image such that the centre of the padded image is the
    centre of mass of the image.

    Arguments
    ---------
    img : image to centre

    pad_value : value to use to pad image

    threshold : value to use to calculate threshold for finding centre of mass.
        'auto' automatically chooses threshold to be average of 10th and 90th percentile values
        from the image.
        None applies no threshold value to image and calculates centre of mass of image values.

    interp : factor to scale image by to improve accuracy of centering.
        Image scaling uses 2D interpolation.

    Returns
    -------
    Image padded with `pad_value` and optionally scaled up with centre of mass of image in centre
    of image.
    """
    img = interp_image(img, interp)

    if threshold is None:
        com = np.array(center_of_mass(img))
    else:
        threshold_img = img.copy()
        if threshold == 'auto':
            threshold = 0.5*sum([np.percentile(threshold_img, 10),
                                 np.percentile(threshold_img, 90)])

        threshold_img[np.where(threshold_img <= threshold)] = 0
        threshold_img[np.where(threshold_img > threshold)] = 1
        com = np.array(center_of_mass(threshold_img))

    shape = np.array(img.shape)
    offset = shape/2 - com
    offset = np.around(offset).astype(int)
    padded_img = np.zeros(shape + 2*abs(offset), dtype=img.dtype)
    padded_img[:] = pad_value
    inset_shift = offset + abs(offset)

    padded_img[inset_shift[0]:inset_shift[0]+shape[0],
               inset_shift[1]:inset_shift[1]+shape[1]] = img

    return padded_img
