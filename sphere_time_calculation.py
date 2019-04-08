#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate execution time for SPHERE IFS/IRDIS instrument on VLT.

Equations taken from page 95 of
https://www.eso.org/sci/facilities/paranal/instruments/sphere/doc/VLT-MAN-SPH-14690-0430_P104.pdf
"""

__version__ = '1.0'
__date__ = '2019-04-08'
__author__ = 'Oliver King'

import tools


# Define observing parameters here...
NDIT_IFS = 15
DIT_IFS = 4
NEXP_IFS = 5

n_obs = 4
n_target = 2

NDIT_IRD = NDIT_IFS
DIT_IRD = DIT_IFS
NEXP_IRD = NEXP_IFS

NDITHER = 1
NPATTERN = 1

calstar = True

coronagraph_centering = False
star_center = False
flux = False


# Display inputs
print()
print(f'\tIFS\tIRD')
fmt = '3.0f'
for label, v_ifs, v_ird in zip(['DIT', 'NDIT', 'NEXP'],
                               [DIT_IFS, NDIT_IFS, NEXP_IFS],
                               [DIT_IRD, NDIT_IRD, NEXP_IRD]):
    print(f'{label}\t{v_ifs:{fmt}}\t{v_ird:{fmt}}')
print()
print(f'# obs/target\t{n_obs}')
print(f'# targets  \t{n_target}')
print(f'calstar  \t{calstar}')
print(f'coronagraph\t{coronagraph_centering}')
if star_center:
    print(f'star center\t{star_center}')
if flux:
    print(f'flux    \t{flux}')

print()

# Detector constants and equations
O_START_IFS = 0.3
O_START_IRD = 0.3
DIT_DELAY_IFS = 0.2
DIT_DELAY_IRD = 0.1
ROT_IFS = 1.65
ROT_IRD = 0.838
O_EXP_IFS = 1.1
O_EXP_IRD = 1.1
O_DITH_IRD = 1.77

time_IFS = (O_START_IFS + NDIT_IFS*(DIT_DELAY_IFS + DIT_IFS + ROT_IFS) + O_EXP_IFS)*NEXP_IFS
time_IRD = ((O_START_IRD + NDIT_IRD*(DIT_DELAY_IRD + DIT_IRD + ROT_IRD)+ O_EXP_IRD)*NEXP_IRD
             + O_DITH_IRD)*NDITHER*NPATTERN
time_IFS /= 60
time_IRD /= 60
time_obs = max(time_IFS, time_IRD)

acquisition = 10
coronagraph_centering *= 5
star_center *= 3.5
flux *= 3.5
calstar = 2 if calstar else 1 # Double time for calstar obs
time_ovh = acquisition + coronagraph_centering + star_center + flux
time_tot = time_ovh + time_obs
time_moon = time_tot*n_obs*calstar
time_all = time_moon*n_target

# Display output
fmt = '5.1f'
tools.script.cprint(f'Overhead     {time_ovh:{fmt}} min', bg='b')
tools.script.cprint(f'IFS          {time_IFS:{fmt}} min', bg='c', fg='k')
tools.script.cprint(f'IRDIS        {time_IRD:{fmt}} min', bg='c', fg='k')
tools.script.cprint(f'Observation  {time_obs:{fmt}} min', bg='b')
tools.script.cprint(f'Total/obs    {time_tot:{fmt}} min', bg='g')
tools.script.cprint(f'Total/target {time_moon:{fmt}} min', bg='y')
tools.script.cprint(f'            = {time_moon/60:.2f} h  ', bg='y')
tools.script.cprint(f'Total all    {time_all:{fmt}} min', bg='r')
tools.script.cprint(f'            = {time_all/60:.2f} h  ', bg='r')
