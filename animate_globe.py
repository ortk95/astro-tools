#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates animation of rotating globe using projected mapped data.

Example in python:
>>> from animate_globe import animate_globe
>>> img = np.random.rand(18, 36) # Create some random data to plot
>>> animate_globe(img, lat=30, equator=dict(color='w'), meridian=True)

Example from command line:
$ ./animate_globe.py map_image.png animated_globe.mp4

See docstring of animate_globe() for list of function arguments.

Requirements
------------
* numpy
* matplotlib
* cartopy
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mpanim
import matplotlib.image as mpimg
import matplotlib.ticker as mticker
import cartopy.crs as ccrs


def animate_globe(img, movie_path='globe.mp4',
                  long_step=1, lat=0, coord_arr=None,
                  positive_west=True, title=True,
                  background_colour='k', figsize=(5, 5), dpi=200, interval=50,
                  central_longitude=180,
                  colorbar=False, colorbar_label='',
                  meridian=False, equator=False, grid=False, grid_kw=None,
                  fn=None,
                  **kwargs):
    """
    Creates animation of rotating globe using projected mapped data.

    The image to project onto the globe can be provided as an array of values
    or as a path to an image file. The plotting uses matplotlib's imshow(). The
    only required argument is `img`, which defines the mapped data to project
    onto the globe. All other arguments are optional and allow the output to be
    customised.

    Parameters
    ----------
    img : array or str
        Image to plot on globe. Can be array of scalar values, array of rgb
        pixel values or a string giving the path of an image to load. Input
        images are assumed to be a cylindrical projection with North at the top
        and a central longitude given by central_longitude (default=180).

    movie_path : str
        Path to save the output animation.

    long_step : number
        Longitude step between each frame of animation. Has no effect if
        coord_arr is defined.

    lat : number
        Latitude of sub-observer point. Has no effect if coord_arr is defined.

    coord_arr : array
        Array of (long, lat) values for each frame of animation. If defined,
        this overrides the long_step and lat values.

    positive_west : bool
        Toggle between positive east and positive west longitudes.

    title : bool or str
        Toggle title for animation (showing current longitude). Set to string
        to add a label to the title.

    background_colour
        Matplotlib compatible colour for background of animation.

    figsize : array
        Size of output animation (in inches).

    dpi : number
        Resolution of output animation (dots per inch).

    interval : int
        Duration of each frame of animation (ms).

    central_longitude : number
        Longitude of central meridian of input cylindically projected image.

    colorbar : bool
        Toggle to enable colorbar on plot.

    colorbar_label : str
        Label of colorbar.

    meridian, equator : bool or dict
        Toggle plotting of lines for the prime meridian and the equator
        respectively. Set as dict of to have custom formatting
        e.g. meridian=dict(color='r') for a red medridian line.

    grid : bool or number
        Toggle plotting of coordinate grid. Set to number to define interval
        between grid lines (default=45).

    grid_kw : dict
        Set custom formatting keywords for gridlines.

    fn : function
        Function to call when generating each frame. The function should take
        the current (long, lat) coordinate tuple as a single argument.

    **kwargs
        Additional arguments are passed to Matplotlib's imshow(). e.g. this can
        be used to choose a specific colormap with cmap='colormap name'.
    """
    # Process inputs
    if isinstance(img, str):
        img = np.flipud(mpimg.imread(img))

    # Set up image properties
    if 'vmin' not in kwargs:
        kwargs['vmin'] = np.nanmin(img)
    if 'vmax' not in kwargs:
        kwargs['vmax'] = np.nanmax(img)
    text_colour = 'k' if background_colour == 'w' else 'w'
    if grid_kw is None:
        grid_kw = {}
    if equator is True:
        equator = dict(color='grey')
    if meridian is True:
        meridian = dict(color='grey')
    if grid is True:
        grid = 45

    # Define coordinates for each frame
    if coord_arr is None:
        long_arr = np.arange(0, 360, abs(long_step))
        if long_step < 1:
            long_arr = long_arr[::-1]
        coord_arr = [(long, lat) for long in long_arr]

    # Define function for plotting each frame
    def animate_frame(coords):
        """Function to create animation frame-by-frame"""
        # Print progress
        fraction_done = (coord_arr.index(coords) + 1)/len(coord_arr)
        msg = f'Animating globe {fraction_done:.1%}  '
        print('\r'+msg, end='', flush=True)

        # Set up map projection and plot image
        long, lat = coords
        plt.clf()
        ax = plt.axes(projection=ccrs.Orthographic(long, lat))
        im = ax.imshow(img, transform=ccrs.PlateCarree(central_longitude=central_longitude),
                       **kwargs)
        plt.box(False)

        # Custom formatting options
        if grid:
            gl = ax.gridlines(crs=ccrs.PlateCarree(central_longitude=central_longitude), **grid_kw)
            gl.xlocator = mticker.FixedLocator(np.arange(0, 360+grid/2, grid))
            gl.ylocator = mticker.FixedLocator(np.arange(-90, 90+grid/2, grid))
        if equator:
            plt.plot(np.linspace(0, 360, 361), np.full(361, 0), transform=ccrs.Geodetic(),
                     **equator)
        if meridian:
            plt.plot(np.full(181, 0), np.linspace(-90, 90, 181), transform=ccrs.Geodetic(),
                     **meridian)

        if title:
            if isinstance(title, str):
                title_str = title
            else:
                title_str = ''
            if positive_west:
                long_real = (-long) % 360
                plt.title(f'{title_str}{long_real:}°W', color=text_colour)
            else:
                long_real = long
                plt.title(f'{title_str}{long_real:}°E', color=text_colour)

        if colorbar:
            cb = plt.colorbar(im)
            if colorbar_label:
                cb.set_label(colorbar_label, color=text_colour)
            cb.ax.yaxis.set_tick_params(color=text_colour)
            cb.outline.set_edgecolor(text_colour)
            plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=text_colour)

        if fn:
            fn(coords)

        return im,

    # Do actual animation
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=background_colour)
    ani = mpanim.FuncAnimation(fig, animate_frame, coord_arr, interval=interval, blit=True)
    ani.save(movie_path, savefig_kwargs={'facecolor': background_colour})
    print(f'\nAnimation saved as "{movie_path}"')
    plt.close(fig)


if __name__ == '__main__':
    animate_globe(*sys.argv[1:])
