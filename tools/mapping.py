"""
Module for mapping tools.

Contains methods to generate, photometrically correct and process maps.
"""
import gzip
import pickle
from datetime import datetime, timedelta
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
import scipy
import scipy.interpolate
from PIL import Image
from astroquery.jplhorizons import Horizons, conf
from scipy.ndimage.measurements import center_of_mass

import tools


def get_ephemerides(hdr=None, return_eph=True, loc=None, epoch=None, target=None,
                    cache=False, **kwargs):
    """
    Loads ephemeris data from JPL horizons for provided FITS header object.

    Telescope location, target and observation datetime can be found from
    header object. See astroquery documentation for more details.

    Parameters
    ----------
    hdr
        FITS header object for observation requiring ephemeris data. Set to
        None to use data from loc, epoch and target instead.

    return_eph : bool
        Toggle returning full ephemeris object or returning default mapping
        parameters.

    loc : dict
        Location of observer.

    epoch : dict
        Observation timings.

    target : str
        Observation target.

    cache : bool
        Toggle saving/using locally cached results.

    Returns
    -------
    ephemerides
    """
    # Get info from FITS header
    if target is None:
        target = hdr['OBJECT'].casefold()
    target_dict = {
            'io': 501,
            'europa': 502,
            'jupiter': 599,
            }
    if target in target_dict:
        target = target_dict[target] # Correct names to unambiguous ID if necessary
    if epoch is None:
        epoch = hdr['MJD-OBS']
    if loc is None:
        loc = {'lon': hdr['ESO TEL GEOLON'],
               'lat': hdr['ESO TEL GEOLAT'],
               'elevation': hdr['ESO TEL GEOELEV']/1e3  # m -> km
               }
    # Load ephemeris data
    conf.horizons_server = 'https://ssd.jpl.nasa.gov/horizons_batch.cgi'
    obj = Horizons(id=target, location=loc, id_type='majorbody', epochs=epoch)
    # Astroquery automatically caches query result locally, so should still work if no internet
    # connection is available and exact query has been performed before on the same machine.

    if cache:
        # Use cached results from previous runs etc. to avout bottleneck
        if cache is True:
            cache = 'rw'
        cache_path = tools.path.code('cache', 'ephemerides.pickle.gz')
        try:
            with gzip.open(cache_path, 'rb') as f:
                cache_db = pickle.load(f)
        except FileNotFoundError:
            cache_db = {}

        key = str(obj)
        if key in cache_db and 'r' in cache:
            eph = cache_db[key]
        else:
            eph = obj.ephemerides(**kwargs)

        if 'w' in cache and key not in cache_db:
            cache_db[key] = eph
            tools.file.check_path(cache_path)
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump(cache_db, f)
    else:
        eph = obj.ephemerides(**kwargs)

    if return_eph:
        return eph
    # Return data relevant for mapping
    obs_long = -float(eph['PDObsLon'])
    obs_lat = float(eph['PDObsLat'])
    sun_long = -float(eph['PDSunLon'])
    sun_lat = float(eph['PDSunLat'])
    np_ang = float(eph['NPole_ang'])
    return obs_long, obs_lat, sun_long, sun_lat, np_ang


def correct_angle(hdr, angle=0):
    """
    Correct position angle for SPHERE IFS detector.

    Uses values from FITS header, ephemeris (`np_ang`) and from ESO SPHERE
    manual at
    https://www.eso.org/sci/facilities/paranal/instruments/sphere/doc/VLT-MAN-SPH-14690-0430_P102_phase2_zwa.pdf
    Calculation of angle uses formulae from the ESO SPHERE manual too.

    Parameters
    ----------
    hdr
        FITS header object for observation.

    angle : float
        Angle to correct.

    Returns
    -------
    rotation : float
        Rotation to align coordinate system to actual north orientation.
    """
    # Get relevant setup info from header
    date_obs = datetime.strptime(hdr['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f')
    instrument = hdr['ESO OCS DET1 IMGNAME'].split('_')[2]
    stabilization = hdr['ESO INS4 COMB ROT']

    # Define angles
    tn = -1.75
    parang = hdr['ESO TEL PARANG START']
    posang = hdr['ESO INS4 DROT2 POSANG']
    tel_alt = hdr['ESO TEL ALT']
    ins4_drot2_begin = hdr['ESO INS4 DROT2 BEGIN']

    if stabilization == 'FIELD':
        pupil_offset = 0
    else:
        pupil_offset = -135.99

    if instrument == 'IFS':
        ifs_offset = -100.48
    else:
        ifs_offset = 0

    # Calculate correction factor for old observations
    if stabilization == 'FIELD' and date_obs < datetime(2016, 7, 13):
        tn += np.rad2deg(np.arctan(np.tan(np.deg2rad(tel_alt - parang - 2*ins4_drot2_begin))))
    elif stabilization == 'PUPIL' and date_obs < datetime(2016, 7, 13):
        parang += np.rad2deg(np.arctan(np.tan(np.deg2rad(tel_alt - 2*ins4_drot2_begin))))
    if stabilization == 'FIELD':
        parang_posang = posang
    else:
        parang_posang = parang
    angle += parang_posang + tn + pupil_offset + ifs_offset
    angle %= 360  # Put value in nice range
    return angle


def get_disc(img, com=None, r0=None):
    """
    Identifies moon disc in image.

    Parameters
    ----------
    img : array
        Image to identify moon from.

    com : list
        Optional x0, y0 coordinates to manually fix.

    r0 : float
        Optional r0 coordinate to manually fix.

    Returns
    -------
    x0, y0, r0
        Coordinates of centre and radius of disc.
    """
    img = np.array(img)
    mask_img = np.isnan(img)
    img[mask_img] = np.nanmin(img)  # Mask nan values for com calculation etc.
    if com is None:
        # Identify moon centre
        threshold_img = img.copy()
        threshold = 0.5*sum([np.percentile(threshold_img, 10),
                             np.percentile(threshold_img, 90)])
        threshold_img[np.where(threshold_img <= threshold)] = 0
        threshold_img[np.where(threshold_img > threshold)] = 1
        com = np.array(center_of_mass(threshold_img))[::-1]
    x0, y0 = com

    if r0 is None:
        # Identify moon radius
        r_list, val_list = tools.image.get_radial_dependence(img, com)
        r_list = r_list[1:] - 0.5*(r_list[1] - r_list[0])  # Get radii corresponding to dv
        dv_list = np.diff(val_list)
        r0 = r_list[dv_list.argmin()]
    x0, y0, r0 = float(x0), float(y0), float(r0)

    return x0, y0, r0

def get_solar_angles(hdr, **kwargs):
    """
    Get solar angles from ephemerides for observation.

    Parameters
    ----------
    hdr
        FITS header

    Returns
    -------
    theta_i, phi_i
    """
    eph = get_ephemerides(hdr, **kwargs)
    theta_i = -float(eph['alpha'])
    ss_ang = correct_angle(hdr, -float(eph['SubSol_ang']))
    phi_i = -(90 + ss_ang)
    theta_i %= 360
    phi_i %= 360
    return theta_i, phi_i


def map_observation(img, x0, y0, r0, n_angle, obs_long, obs_lat, **kwargs):
    """
    Projects observed planetary disc into an equirectangular map.

    Assumes planetary disc is observed from an infinite distance and is
    therefore an orthographic coordinate projection:
    https://en.wikipedia.org/wiki/Orthographic_projection_in_cartography

    The returned map is an equirectangular/cylindrical projection with rows of
    constant latitude and columns of constant longitude.
    https://en.wikipedia.org/wiki/Equirectangular_projection

    All angles must be given in degrees.

    Parameters
    ----------
    img : array
        Observed image. Must be a 2D image or 3D data cube (where the first
        axis is the spectral dimension).

    x0, y0 : float
        Pixel coordinates of centre of planetary disc in observed image.

    r0 : float
        Pixel radius of planetary disc in observed image.

    n_angle : float
        Angle in degrees of planet's north pole from the top of the observed
        image.

    obs_long, obs_lat : float
        Sub-observer longitude and latitude in degrees. These are the
        geographic coordinates of point in the centre of the observed disc
        given by pixel coordinates (x0, y0).

    **kwargs
        Additional keyword arguments are passed to longlat_to_map()

    Returns
    -------
    map_img : array
        Equirectangular projected mapped image. Unmapped areas have a value of
        NaN.

    long_arr : array
        Arrays of longitude coordinates in degrees corresponding to columns of
        map_img.

    lat_arr : array
        Arrays of latitude coordinates in degrees corresponding to rows of
        map_img.
    """
    shape = img.shape
    if len(shape) == 3:
        shape = shape[1:]
    long_img, lat_img = img_to_longlat(shape, x0, y0, r0, n_angle, obs_long, obs_lat)
    return longlat_to_map(img, long_img, lat_img, obs_long=obs_long, **kwargs)


def img_to_xyz(shape, x0, y0, r0):
    """
    Create coordinates images for an observed planetary disc.

    Parameters
    ----------
    shape
        Shape of observed image.

    x0, y0 : float
        Pixel coordinates of centre of planetary disc in observed image.

    r0 : float
        Pixel radius of planetary disc in observed image.

    Returns
    -------
    x_img, y_img, z_img: array
        Images of (x, y, z) coordinates of observed planetary disc. Points
        outside the disc are given as NaN. Coordinates are given as fractions
        of the planetary radius.
    """
    x_img, y_img = np.meshgrid(np.arange(shape[1]) - x0, np.arange(shape[0]) - y0)
    r_img = np.sqrt(x_img**2 + y_img**2)
    x_img /= r0
    y_img /= r0
    z_img = 1 - r_img/r0
    x_img[np.where(r_img > r0)] = np.nan
    y_img[np.where(r_img > r0)] = np.nan
    z_img[np.where(r_img > r0)] = np.nan
    return x_img, y_img, z_img


def img_to_longlat(shape, x0, y0, r0, n_angle, obs_long, obs_lat):
    """
    Create geographic coordinate images for an observed planetary disc.

    All angles must be given in degrees.

    Parameters
    ----------
    shape
        Shape of observed image.

    x0, y0 : float
        Pixel coordinates of centre of planetary disc in observed image.

    r0 : float
        Pixel radius of planetary disc in observed image.

    n_angle : float
        Angle in degrees of planet's north pole from the top of the observed
        image.

    obs_long, obs_lat : float
        Sub-observer longitude and latitude in degrees. These are the
        geographic coordinates of point in the centre of the observed disc
        given by pixel coordinates (x0, y0).

    Returns
    -------
    long_img, lat_img : array
        Images of geographic coordinates of observed planetary disc. Points
        outside the disc are given as NaN.
    """
    x_img, y_img, z_img = img_to_xyz(shape, x0, y0, r0)
    return xyz_to_longlat(x_img, y_img, z_img, n_angle, obs_long, obs_lat)


def longlat_to_map(img, long_img, lat_img, method='linear', ppd=4, visualise=False, obs_long=None,
                   print_progress=True):
    """
    Projects image to map using associated coordinates.

    Parameters
    ----------
    img : array
        Image to project into a map. If cube, projects each part separately.

    long_img, lat_img : array
        Image of long/lat values of each pixel in `img`.

    method : str
        Interpolation method used in `scipy.interpolate.griddata()`.

    ppd : int
        Resolution of output map, pixels per degree.

    visualise : bool
        Toggle plotting of map fit.

    obs_long : float
        Longitude of centre of observation.

    print_progress : bool
        Toggle printing of progress.

    Returns
    -------
    map_img, long_arr, lat_arr
        Image projected to map and arrays of coordinates.
    """
    # Mask values so only interpolate non-nan coordinates
    lat_arr_in = lat_img[np.where(~np.isnan(lat_img) & ~np.isnan(long_img))]
    long_arr_in = long_img[np.where(~np.isnan(lat_img) & ~np.isnan(long_img))]
    long_arr_in = long_arr_in % 360  # Ensure consistent longitude range

    # Interpolate to map image
    lat_arr = np.arange(-90, 90, 1/ppd)
    long_arr = np.arange(0, 360, 1/ppd)

    if obs_long is not None:
        # Shift coordinate system so that map part of image is in centre to avoid wraparound
        # effects.
        obs_long = int(obs_long + 180) % 360
        long_arr = (long_arr - obs_long) % 360
        long_arr_in = (long_arr_in - obs_long) % 360

    points = [(long, lat) for long, lat in zip(long_arr_in, lat_arr_in)]
    long_grd, lat_grd = np.meshgrid(long_arr, lat_arr)

    # Do image manipulation, assume cube by default
    img = np.array(img)
    is_cube = (len(img.shape) == 3)
    if is_cube:
        cube = img
    else:
        cube = [img]
    cube_arrs_in = [f[np.where(~np.isnan(lat_img) & ~np.isnan(long_img))] for f in cube]

    # Following lines are bottleneck for mapping procedure
    map_cube = np.zeros((len(cube_arrs_in), len(lat_arr), len(long_arr)))
    for idx, f in enumerate(cube_arrs_in):
        map_cube[idx] = scipy.interpolate.griddata(points, f, (long_grd, lat_grd), method=method)
        if print_progress:
            print('.', end='', flush=True)
    if print_progress:
        print()

    if visualise:
        plt.clf()
        if obs_long is not None:
            long_arr = (long_arr + obs_long)
            long_arr_in = (long_arr_in + obs_long)
        plt.pcolormesh(long_arr, lat_arr, map_cube[0])
        plt.colorbar()
        plt.scatter(long_arr_in, lat_arr_in, c='k', s=1, marker='.')
        plt.title('Dots represent the projection of individual pixels from the original image')
        plt.pause(1e-3)

    if obs_long is not None:
        # Reset coordinates
        long_arr = np.arange(0, 360, 1/ppd)

    # Convert back to correct format for return
    if is_cube:
        map_img = map_cube
    else:
        map_img = map_cube[0]
    return map_img, long_arr, lat_arr


def xyz_to_longlat(x_img, y_img, z_img, n_angle, obs_long, obs_lat, degrees=True):
    """
    Convert images of xyz coordinates to images of long/lat coordinates.

    Uses formulae to produce orthographic coordinate projection from
    https://en.wikipedia.org/wiki/Orthographic_projection_in_cartography.

    Parameters
    ----------
    x_img, y_img, z_img : array
        Images of xyz coordinates (in units of moon radii).

    n_angle : float
        Angle of north pole (rotation applied to image before orthographic
        projection).

    obs_long, obs_lat : float
        Sub-observer long/lat (i.e. centre of final projection).

    degrees : bool
        Toggle between input and output angles being degrees or radians.

    Returns
    -------
    long_img, lat_img :
        Iimages of long and lat coordinates. `long_img` values run from 0 to
        360 (degrees) or 2*pi (radians).
    """
    if degrees:
        n_angle = np.deg2rad(n_angle)
        obs_long = np.deg2rad(obs_long)
        obs_lat = np.deg2rad(obs_lat)

    # Rotate initial axes so that north is in the correct location
    coords = [c for c in zip(x_img.ravel(), y_img.ravel(), z_img.ravel())]
    M = tools.maths.rotation_matrix(n_angle, 'z', degrees=False)
    coords = np.array([np.dot(M, c) for c in coords])
    x_img = coords[:, 0].reshape(x_img.shape)
    y_img = coords[:, 1].reshape(y_img.shape)

    # Carry out transformation to orthographic projection using formulae from Wikipedia
    with np.errstate(divide='ignore', invalid='ignore'):
        r_img = np.sqrt(x_img**2 + y_img**2)
        r_img[np.where(r_img > 1)] = 1  # Avoid floating point errors
        c_img = np.arcsin(r_img)
        lat_img = np.arcsin(np.cos(c_img)*np.sin(obs_lat)
                            + y_img*np.sin(c_img)*np.cos(obs_lat)/r_img)
        if 0 in r_img:
            # Deal with pixels where 0/0 causes a value of NaN
            lat_img[np.where(r_img == 0)] = np.arcsin(
                np.cos(c_img)*np.sin(obs_lat))[np.where(r_img == 0)]

        long_img = obs_long + np.arctan2(x_img*np.sin(c_img),
                                         (r_img*np.cos(c_img)*np.cos(obs_lat)
                                             - y_img*np.sin(c_img)*np.sin(obs_lat)))
    long_img = long_img % (2*np.pi)  # ensure values are in expected range
    if degrees:
        lat_img = np.rad2deg(lat_img)
        long_img = np.rad2deg(long_img)
    return long_img, lat_img


def xyz_to_thetaphi(x_img, y_img, z_img, theta=0, phi=0, degrees=True):
    """
    Convert xyz coordinates to polar coordinates.

    Parameters
    ----------
    x_img, y_img, z_img : array
        Images of xyz coordinates (in units of moon radii).

    theta, phi : float
        Rotation of coordinate system.

    degrees : bool
        Toggle units.

    Returns
    -------
    theta_img, phi_img
    """
    if degrees:
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
    coords = [c for c in zip(x_img.ravel(), y_img.ravel(), z_img.ravel())]
    M_theta = tools.maths.rotation_matrix(theta, 'y', degrees=False)
    M_phi = tools.maths.rotation_matrix(-phi, 'z', degrees=False)
    M_phi = M_theta@M_phi@np.linalg.inv(M_theta)
    M = M_phi@M_theta
    coords = np.array([np.dot(M, c) for c in coords])

    x_img = coords[:, 0].reshape(x_img.shape)
    y_img = coords[:, 1].reshape(y_img.shape)
    z_img = coords[:, 2].reshape(z_img.shape)
    z_img = np.clip(z_img, None, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        theta_img = np.arccos(z_img)
        phi_img = np.arctan2(y_img, x_img)
        phi_img += phi
        phi_img = np.mod(phi_img, 2*np.pi)
    if degrees:
        theta_img = np.rad2deg(theta_img)
        phi_img = np.rad2deg(phi_img)
    return theta_img, phi_img


def get_jpl_simulation(hdr, size=None, n_rot=None, color=False):
    """
    Loads JPL Space Simulator simulation of observation of target.

    Assumes observer is in centre of Earth, and manually corrects light travel
    time effect according to `lighttime` from JPL Horizons ephemeris data.
    Loads image from JPL website, so requires active internet connection to
    run. JPL Space Simulator is at https://space.jpl.nasa.gov.

    Parameters
    ----------
    hdr
        FITS header object to specify target and observation datetime.

    size : list
        Required image size. Set to `None` to return image loaded fromJPL.
        Otherwise crops and rescales image to meet specified axis ratio and
        image size.

    n_rot : float
        Rotation of north pole of image, applied to loaded image. If set to
        `None` gets `n_rot` from ephemeris data.

    color : boll
        Toggle conversion to greyscale.

    Returns
    -------
    img
        JPL simulated image of moon observed from Earth at specified time.
    """
    target = hdr['OBJECT']
    target_dict = {
            'Io': 501,
            'Europa': 502,
            'Ganymede': 503,
            'Callisto': 504
            }
    target = target_dict[target]
    obs_date = datetime.strptime(hdr['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f')
    eph = get_ephemerides(hdr)
    lighttime = eph['lighttime']
    if lighttime.unit != 'min':
        print('WARNING: unexpected unit for lighttime')
    obs_date -= timedelta(minutes=float(lighttime))
    year = obs_date.year
    month = obs_date.month
    day = obs_date.day
    hour = obs_date.hour
    minute = obs_date.minute
    # Construct URL to request image from JPL
    url = (f'https://space.jpl.nasa.gov/cgi-bin/wspace?tbody={target}&vbody=399&'
           f'month={month}&day={day}&year={year}&hour={hour}&minute={minute}'
           f'&rfov=0.2&fovmul=-1&bfov=50&showac=1')
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = np.flipud(img)
    if not color:
        img = (1/3)*np.sum(img, axis=2)
    if size is None:
        return img
    size_in = img.shape
    desired_width = int((size[1]/size[0])*size_in[0])
    crop_dist = (size_in[1] - desired_width)//2
    img = img[:, crop_dist:crop_dist+desired_width]
    img = scipy.misc.imresize(img, size)
    if n_rot is None:
        n_rot = calculate_rotation_to_north(hdr, eph['NPole_ang'])
    img = scipy.misc.imrotate(img, angle=n_rot)
    return img
