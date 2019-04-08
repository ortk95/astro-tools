#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for timing code snippets.

Specify the snippets in the SETUP section below, then run the script to
calculate the timings. Adjust the value of repeat and number as appropriate to
perform the timings in a reasonable amount of time. The snippets timed to see
how long they take to execute `number` times, and the fastest time from
`repeat` repeats is taken as the optimal time for that snippet.
"""

__version__ = '1.0'
__date__ = '2019-04-08'
__author__ = 'Oliver King'

import timeit
import math
import tools


#%% SETUP -----------------------------------------------------------------------------------------
# Adjust repeat & number as needed to ensure the timing doesn't take too long (e.g. if the snippets
# take ~0.1s to execute, you'll want to reduce repeat from 10000). Higher values will take longer
# but will give more reliable results.
repeat = 10000 # Number of times to repeat timing loop
number = 100 # Number of times statement is called in each timing loop

# Define any variables, module imports etc. to use in the snippets here...
x = 123.456

# Define code snippets as a list of strings to execute here...
statements = [
'out = x**2',
'out = x*x',
]


#%% PERFORM TIMINGS AND ANALYSIS ------------------------------------------------------------------
# Nothing below here should generally need modifying

global_vals = locals() # Get copy of local variables for use in function statements

print(f'Testing code timings with repeats={repeat} and number={number}')
print()

check_output = all(('out = ' in s or 'out=' in s) for s in statements)
if check_output:
    print(f'{"TIME (s)":^12s}|{"COMMAND":^{max(len(s) for s in statements)+2}s}| OUTPUT')
else:
    print(f'{"TIME (s)":^12s}|{"COMMAND":^{max(len(s) for s in statements)+2}s}')

times = []
out_old = None
for s in statements:
    t = min(timeit.repeat(s, repeat=repeat, number=number, globals=global_vals))/number
    times.append(t)

    if check_output:
        gv = global_vals.copy()
        exec(s, gv)
        out = gv['out']
        print(f'{t:.5e} | {s:{max(len(s) for s in statements)}s} | ', end='')
        if out_old is None:
            fg = None
        elif out == out_old or out is out_old:
            fg = 'g'
        else:
            fg = 'r'
            try:
                if math.isclose(out, out_old):
                    fg = 'c'
            except TypeError:
                pass
        out_old = out
        tools.script.cprint(f'{repr(out)}', fg=fg, style='b', flush=True)
    else:
        print(f'{t:.5e} | {s}', flush=True)

print()
tools.script.print_bar_chart(statements, times, sort=True)
