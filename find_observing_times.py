#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module to identify good observing times from VLT.
"""

__version__ = '1.0'
__date__ = '2019-04-08'
__author__ = 'Oliver King'

import sys
import tools
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def main(*args):
    """
    Execute with specified arguments, or use default case as example.
    """
    if args:
        find_observing_times(*args)
    else:
        print('Using example case of Jupiter for spring 2020')
        find_observing_times('jupiter', datetime(2020, 1, 1), datetime(2020, 4, 1), step='15m')


def find_observing_times(target, start_time, end_time, step='15m', elev_cutoff=0,
                         airmass_cutoff=2, twilight_cutoff='*', print_summary=True,
                         plot_summary=True):
    """
    Find good observing times for a specified target from VLT.

    Uses astroquery/JPL Horizons to load ephemeris at specified time step and
    calculate the observing conditions over the specified time interval. The
    initial call to JPL Horizons is slow, so the result is cached allowing
    future calls to execute much faster.

    Observing time is calculated by counting the number of time steps that the
    ephemeris meets the elevation, airmass and twilight cutoffs. Therefore, the
    resolution of the observing time is equal to the value of `step` (e.g.
    would not be useful for a time step of '1d'). Smaller time steps are
    therefore preferable (but slower).

    Parameters
    ----------
    target : str
        Name of target (ultimately passed to JPL Horizons).

    start_time, end time : datetime or str
        Start and end of observing period, if str, use format '%Y-%m-%d'.

    step : str
        Ephemeris time resolution, unit can be 'm'=minutes, 'h'=hours,
        'd'=days. E.g. '10m' = 10 minutes.

    elev_cutoff : float
        Minimum elevation allowed for valid observing time.

    airmass_cutoff : float
        Maximum airmass allowed for observing time.

    twilight_cutoff : str
        Twilight conditions not allowed for observing time, can be any
        combination of '*CNA'. See JPL Horizons for more detail. '*' will only
        allow observing time when sun is below horizon, '' will allow observing
        time in any twilight condition.

    print_summary : bool
        Toggle printing list of dates.

    plot_summary : bool
        Toggle plotting of summary.
    """
    print('Reading data...')
    if isinstance(start_time, str):
        start_time = datetime.strptime(start_time, '%Y-%m-%d')
    if isinstance(start_time, str):
        start_time = datetime.strptime(start_time, '%Y-%m-%d')

    epoch = {'start': datetime.strftime(start_time, '%Y-%m-%d'),
             'stop': datetime.strftime(end_time, '%Y-%m-%d'),
             'step': step}
    loc = {'lon': -70.4045, 'lat': -24.6268, 'elevation': 2.648}
    eph = tools.mapping.get_ephemerides(epoch=epoch, loc=loc, target=target, cache=True)

    print('Processing data...')
    if twilight_cutoff:
        eph_idxs = [idx for idx, sp in enumerate(eph['solar_presence'])
                    if all(x not in sp for x in twilight_cutoff)]
        eph = eph[eph_idxs]
    if airmass_cutoff:
        eph_idxs = [idx for idx, am in enumerate(eph['airmass'])
                    if am < airmass_cutoff]
        eph = eph[eph_idxs]
    if elev_cutoff:
        eph_idxs = [idx for idx, elev in enumerate(eph['EL'])
                    if elev > elev_cutoff]
        eph = eph[eph_idxs]

    dt = timedelta(hours=-3)
    times = [datetime.strptime(s, '%Y-%b-%d %H:%M') + dt for s in eph['datetime_str']]
    dates = [datetime.date(t) for t in times]
    dates_unique = [datetime.date(start_time + timedelta(days=d))
                    for d in range((end_time - start_time).days)]
    t_step = float(step[:-1])
    if step[-1] == 'm':
        t_step /= 60
    if step[-1] == 'd':
        t_step *= 24
    num_dates = [dates.count(d)*t_step for d in dates_unique]
    key_list = ['EL', 'elong', 'lunar_elong', 'airmass', 'RA', 'DEC', 'ang_width']
    if print_summary:
        ok_bg = 'g'
        moon_bg = 'c'
        limited_tot_bg = 'b'
        short_tot_bg = 'm'
        bad_bg = 'r'
        print()
        print('Minimum and Maximum values given for each parameter')
        tools.script.cprint('No observing time    ', bg=bad_bg)
        tools.script.cprint('Lunar elong <30˚     ', bg=moon_bg)
        tools.script.cprint('Observing time <30min', bg=short_tot_bg)
        tools.script.cprint('Observing time <1h   ', bg=limited_tot_bg)
        tools.script.cprint('Good                 ', bg=ok_bg)

        print()
        skipped_days = 0
        title = '   Date    | ToT '
        for key in key_list:
            title += f' | {key:^11s}'
        print(title)

        for idx, date in enumerate(dates_unique):
            if num_dates[idx] == 0:
                skipped_days += 1
            else:
                if skipped_days:
                    msg = f'>>> skipped {skipped_days} dates with no observing time <<<'
                    tools.script.cprint(f'{msg:^{len(title)}s}', bg=bad_bg)
                    skipped_days = 0
                fg = None
                bg = ok_bg
                msg = f'{date} | {num_dates[idx]:.1f}h'
                if num_dates[idx] < 1:
                    bg = limited_tot_bg
                if num_dates[idx] < 0.5:
                    bg = short_tot_bg
                idx_list = np.where(np.array(dates) == date)
                if len(idx_list[0]) == 0:
                    continue
                for key in key_list:
                    vals = eph[key][idx_list]
                    if key == 'lunar_elong' and min(vals) < 30:
                        bg = moon_bg
                    msg += f' | {min(vals):5.1f} {max(vals):5.1f}'
                tools.script.cprint(msg, fg=fg, bg=bg)
        if skipped_days:
                    msg = f'>>> skipped {skipped_days} dates with no observing time <<<'
                    tools.script.cprint(f'{msg:^{len(title)}s}', bg=bad_bg)

    if plot_summary:
        plt.clf()
        plt.grid(alpha=0.33)
        for idx, key in enumerate(key_list):
            plt.scatter(times, eph[key], label=key, s=10)
            plt.xlim(start_time, end_time)
        plt.ylabel('Angle, deg')
        plt.legend()
        plt.axhline(0, color='k', alpha=0.5, linewidth=1)
        plt.ylim(0)
        plt.xlabel('Chilean time')
        plt.title(f'{target.capitalize()} visibility from VLT (elevation cutoff = {elev_cutoff}˚, '
                  f'twilight cutoff = {twilight_cutoff}, airmass cutoff = {airmass_cutoff}, '
                  f' time resolution = {step})')
        plt.twinx()
        plt.plot(dates_unique, num_dates, label='Observable time', color='k')
        plt.xlim(start_time, end_time)
        plt.ylim(0)
        plt.ylabel(f'Approx. observable time, h (black line)')
        plt.show()


if __name__ == '__main__':
    main(*sys.argv[1:])
