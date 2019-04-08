"""
Module for script operations.

Contains methods to deal with running scripts, displaying output to user etc.
"""
import collections
import copy
import hashlib
import os
import random
import re
import subprocess
import traceback
from datetime import datetime, timedelta
from multiprocessing import Pool

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tools


def test_if_ipython():
    """Detect if script is running in IPython console"""
    try:
        return __IPYTHON__
    except NameError:
        return False


def checksum(item):
    """
    Produces a checksum of any python object.

    Hashes the string representation produced by obj_to_str(). This produces
    the same string for equivalent values, so the checksum is independent of
    factors such as dictionary key order, list vs. numpy array etc - see
    obj_to_str() for more details.

    Hash is calculated using SHA-256.

    Parameters
    ----------
    item
        Python object to calculate checksum of

    Returns
    -------
    str
        Hexdigest of object's string representation.

    """
    return hashlib.sha256(obj_to_str(item).encode('utf-8')).hexdigest()


def obj_to_str(obj):
    """
    Convert generic object into a standardised string representation.

    Similar to calling repr() on an object, but this function standardises
    certain objects so different objects with the same meaning produce the same
    output string.

    dict: String representations of dictionaries are sorted by string first so
    output is independent of key order:

    >>> obj_to_str({'a':1, 'b':2}) == obj_to_str({'b':2, 'a':1})

    int: Integers are converted to floats:

    >>> obj_to_str(1) == obj_to_str(1.0)

    iterable: Iterables (e.g. list, tuple, numpy array, range) are all
    converted to lists:

    >>> obj_to_str([1,2,3]) == obj_to_str((1,2,3)) == obj_to_str(range(1,4))

    set: Sets are sorted then converted to lists:

    >>> obj_to_str(set(1,2)) == obj_to_str(set(2,1)) == obj_to_str([1,2])
    >>> obj_to_str(set(2,1)) != obj_to_str([2,1])

    Note obj_to_str() is called recursively on objects within other objects:

    >>> obj_to_str([1,2]) == obj_to_str([obj_to_str(1), obj_to_str(2)])

    All other objects just return the result of calling repr() on the object.

    Parameters
    ----------
    obj

    Returns
    -------
    str
    """
    try:
        # Automatically return repr() for a certain types to avoid going through the whole function
        # and then returning the repr() right at the end.
        repr_types = (str, astropy.io.fits.header.Header)
        # Ensure FITS headers are just treated as string representations rather than iterables
    except NameError:
        repr_types = str
    if isinstance(obj, repr_types):
        # Default cases
        return repr(obj)

    try:
        # Convert to give same string for 1, 1.0, 1.0000 etc.
        return repr(float(obj))
    except (TypeError, ValueError):
        pass

    if isinstance(obj, dict):
        # Normalise dict order by sorting
        obj = sorted([obj_to_str(k) + ': ' + obj_to_str(v) for k, v in obj.items()])
        return '{' + ', '.join(obj) + '}'

    if isinstance(obj, set):
        # Normalise set order by sorting, then treat as standard iterable
        obj = list(sorted(obj))

    try:
        try:
            # Try short circuit for all numeric case for performance reasons (this doesn't change
            # the output string, but can avoid repeated calling of obj_to_str() for e.g. numpy
            # arrays).
            return repr(list(map(float, obj)))
        except (TypeError, ValueError):
            pass
        # Treat all iterables the same and normalise contents
        return '[' + ', '.join([obj_to_str(x) for x in obj]) + ']'

    except TypeError:
        pass

    # Fallback to "official" string representation of object
    return repr(obj)


def shell(command_str, print_command=True, **kwargs):
    """
    Prints and calls system command.

    Parameters
    ----------
    command_str : str
        String which will be executed at system shell.

    print_command : bool
        Toggle printing shell command to terminal.

    **kwargs
        Additional commands are passed to `subprocess.Popen()` or
        `subprocess.run()` depending on if IPython. `shell=True` for both.
    """
    if print_command:
        cprint('> ' + command_str, fg='c', style='b')

    if test_if_ipython():
        # Deal with IPython explicitly so that stdout is printed to console
        process = subprocess.Popen(command_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   shell=True, **kwargs)
        while process.poll() is None:
            line = process.stdout.readline()
            print(line.decode('utf-8'), end='')
        line = process.stdout.read()
        print(line.decode('utf-8'), end='')
        if process.poll():
            retcode = process.poll()
            print(process.stderr.read().decode('utf-8'), end='')
            raise subprocess.CalledProcessError(retcode, process.args, output=process.stdout,
                                                stderr=process.stderr)
    else:
        subprocess.run(command_str, check=True, shell=True, **kwargs)


def cprint(msg='', fg=None, bg=None, style=None, return_str=False, **kwargs):
    """
    Prints coloured and formatted text.

    Parameters
    ----------
    msg
        Message to print.

    fg, bg : {'k', 'r', 'g', 'b', 'y', 'm', 'c', 'w'}
        Foreground and background colours (see code for options).

    style : {'b', 'f', 'i', 'u', 'x', 'y', 'r', 'h', 's'}
        Formatting style to apply to text. Can be multiple values, e.g. 'bi'
        for bold and italic style.

    return_str : bool
        Toggle to disable printing output and instead return formatted string
        (for use in other print statement).

    **kwargs
        Additional arguments are passed to `print()`

    Returns
    -------
    str
        Optional toggle `return_str` returns formatted string for use printing
        elsewhere.
    """
    colcode = {
        'k': 0,  # black
        'r': 1,  # red
        'g': 2,  # green
        'y': 3,  # yellow
        'b': 4,  # blue
        'm': 5,  # magenta
        'c': 6,  # cyan
        'w': 7,  # white
    }

    fmtcode = {
        'b': 1,  # bold
        'f': 2,  # faint
        'i': 3,  # italic
        'u': 4,  # underline
        'x': 5,  # blinking
        'y': 6,  # fast blinking
        'r': 7,  # reverse
        'h': 8,  # hide
        's': 9,  # strikethrough
    }

    # Properties
    props = []
    if isinstance(style, str):
        props = [fmtcode[s] for s in style]
    if isinstance(fg, str):
        props.append(30 + colcode[fg])
    if isinstance(bg, str):
        props.append(40 + colcode[bg])

    # Display
    props = ';'.join([str(x) for x in props])
    if return_str:
        if props:
            return '\x1b[%sm%s\x1b[0m'%(props, msg)
        else:
            return msg
    else:
        if props:
            print('\x1b[%sm%s\x1b[0m'%(props, msg), **kwargs)
        else:
            print(msg, **kwargs)


def cprint_table():
    """
    Prints summary table of cprint options.
    """
    colors = [None, 'k', 'r', 'g', 'y', 'b', 'm', 'c', 'w']
    styles = [None, 'b', 'f', 'i', 'u', 'x', 'y', 'r', 'h', 's']
    for style in styles:
        print(style)
        for fg in colors:
            for bg in colors:
                msg = ''
                for x in [fg, bg]:
                    if x is None:
                        msg += '-'
                    else:
                        msg += x
                cprint(' ' + msg + ' ', fg=fg, bg=bg, style=style, end='')
            print()
        print()


def print_message_with_title(title, *message, c1=None, c2=None, **kwargs):
    """
    Prints message with a title block and formatted with colours for emphasis.

    Parameters
    ----------
    title : str
        Title of message (e.g. 'WARNING')

    *message : str
        Message(s) for printing. Supply multiple arguments for multiple lines.

    c1, c2 : str
        Colours to format the title/message, using format in `cprint()`. `c1`
        is the main message text colour and `c2` is the main message background
        colour (reversed for title).

    **kwargs
        Additional arguments passed to cprint()
    """
    cprint(' ' + title + ' ', fg=c2, bg=c1, end='', **kwargs)
    first_line = True
    for string in message:
        if not first_line:
            cprint(' '*(len(title) + 2), fg=c1, bg=c2, end='', **kwargs)
        first_line = False
        extra_width = max([len(s) for s in message]) - len(string) + 1
        cprint(' ' + str(string) + ' '*extra_width, fg=c1, bg=c2, **kwargs)


def todo(*args):
    """
    Prints formatted message to command prompt when script run.

    Parameters
    ----------
    *args :
        Message to print. Additional arguments are printed indented under
        original message.
    """
    print_message_with_title('TODO', *args, c1='b', c2='c')


def print_warning(*args):
    """
    Prints formatted message to command prompt when script run.

    Parameters
    ----------
    *args :
        Message to print. Additional arguments are printed indented under
        original message.
    """
    print_message_with_title('WARNING', *args, c1='y', c2='k', style='b')


def print_error(*args):
    """
    Prints formatted message to command prompt when script run.

    Parameters
    ----------
    *args :
        Message to print. Additional arguments are printed indented under
        original message.
    """
    print_message_with_title('ERROR', *args, c1='r', c2=None, style='b')


def print_progress(annotation=None, c1='g', c2='k', style='b'):
    """
    Print progress summary of current code.

    Prints summary of code location and execution time for use in optimising and monitoring code.
    Uses traceback to identify the call stack and prints tree-like diagram of stacks where this
    function was called. The call stack is relative to the first time this function is called as it
    uses properties of print_progress to communicate between calls.

    Printed output contains:
        - Seconds elapsed since last call of print_progress.
        - Current time.
        - Traceback for location print_progress is called from, relative to first location
          print_progress was called. This includes file names, function names and line numbers.
        - Optional annotation provided to explain what current line is.

    Arguments
    ---------
    annotation : str
        Optionally provide annotation about current line of code.

    c1, c2, style : str
        Formatting options to pass to cprint()
    """
    now = datetime.now()  # Ignore duration of current code

    # Get timings
    if print_progress.last_dtm is None:
        title = ' seconds'
    else:
        td = now - print_progress.last_dtm
        title = f'{td.total_seconds():8.2f}'
    title += ' @ '
    title += now.strftime('%H:%M:%S')
    title += ' '

    # Get stack
    stack = traceback.extract_stack()[:-1]
    if print_progress.first_stack is None:
        print_progress.first_stack = stack
    first_stack = print_progress.first_stack
    split_idx = len([None for a, b in zip(stack[:-1], first_stack) if a == b])
    stack = stack[split_idx:]
    stack_text = [f'{s[2]} (line {s[1]} in {os.path.split(s[0])[1]})' for s in stack]
    last_stack = print_progress.last_stack
    if last_stack is not None:
        last_idx = len([None for a, b in zip(stack[:-1], last_stack) if a == b])
        stack_text = stack_text[last_idx:]
    else:
        last_idx = 0
    print_progress.last_stack = stack

    # Do actual printing
    for st in stack_text:
        msg = cprint(' ' + '|  '*last_idx + st + ' ', fg=c1, bg=c2, style=style, return_str=True)
        if st == stack_text[-1]:
            tt = title
            if annotation:
                msg += cprint(f'{annotation} ', fg=c1, bg=c2, style=style + 'i', return_str=True)
        else:
            tt = ' '*len(title)
        tt = cprint(tt, fg=c2, bg=c1, style=style, return_str=True)
        print(tt + msg, flush=True)
        last_idx += 1
    print_progress.last_dtm = datetime.now()  # Ignore duration of this code


print_progress.last_dtm = None
print_progress.last_stack = None
print_progress.first_stack = None


def wrap_call(*args, return_idx=0):
    """
    Convenience function to allow separate functions to be called from one place.

    The value of one of the arguments (specified by return_idx) is returned as-is.

    Usage
    -----
    >>> a = [wrap_call(x**2, print(x, end=' ')) for x in range(10)]
    0 1 2 3 4 5 6 7 8 9
    >>> a
    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]


    Arguments
    ---------
    *args
        Arguments which can be any function.

    return_idx : int
        Index of value to return.

    Returns
    -------
    args[return_idx]
    """
    return args[return_idx]


def get_console_width(fallback=75):
    """
    Attempts to find console width, otherwise uses fallback provided.

    Parameters
    ----------
    fallback : int
        Default width value if `stty size` fails.

    Returns
    -------
    width : int
        Console width.
    """
    if test_if_ipython():
        return fallback
    try:
        _, width = subprocess.check_output(['stty', 'size'], stderr=subprocess.PIPE).split()
    except:
        width = fallback
    width = int(width)
    return width


def print_heading(msg, level=1, width=None, add_spaces=True, upper=True, box=False,
                  buffer='before', align='^', **kwargs):
    """
    Prints message in heading format for emphasis and code consistency.

    Parameters
    ----------
    msg : str
        String to print as message.

    level : int or str
        Heading level (1=highest) for default heading formats. Otherwise,
        define custom spacer here.

    width : int
        Width of heading output, `None` uses the result of
        `get_console_width()`.

    add_spaces : bool
        Toggle adding spaces around msg.

    upper : bool
        Toggle transforming to upper case.

    box : bool
        Toggle printing line of spacers before and after msg line.

    buffer : {'before', 'after', 'both'}
        Print blank lines 'before'/'after'/'both'.

    align : {'^', '<', '>'}
        Define alignment string format for heading.

    **kwargs
        Passed to `cprint()` when printing heading.
    """
    # Default heading levels
    if isinstance(level, int):
        spacer_list = ['█', '░', '━', '-', '.', ' ']
        spacer_idx = np.clip(level, 1, len(spacer_list)) - 1
        spacer = spacer_list[spacer_idx]
    else:
        spacer = str(level)
    if not np.isscalar(msg):
        if buffer == 'both' or buffer == 'before':
            print()
        if box:
            print(spacer*width)
        for m in msg:
            print_heading(m, spacer, width, add_spaces, upper, box=False, buffer=False)
        if box:
            print(spacer*width)
        if buffer == 'both' or buffer == 'after':
            print()
        return

    if upper:
        msg = msg.upper()
    if add_spaces:
        msg = ' ' + msg.strip() + ' '
    if width is None:
        width = get_console_width()
    if buffer is True or buffer == 'before':
        print()
    if box:
        cprint(spacer*width, **kwargs)
    cprint(f'{msg:{spacer}{align}{width}}', **kwargs)
    if box:
        cprint(spacer*width, **kwargs)
    if buffer is True or buffer == 'after':
        print()


def print_box(*msg, width='console', align='^', start=True, end=True, **kwargs):
    """
    Prints message surrounded by a box.

    Parameters
    ----------
    *msg : str
        Message to print, provide multiple messages for multiple lines.

    width : str or int
        Box width, set to 'console' to use get_console_width(), 'content' to
        use content width or a value to use a specific width.

    align : {'^', '<', '>'}
        Define alignment string format.

    start, end : bool
        Toggle printing start/end of box (useful for separating printing box
        over multiple statements).

    **kwargs
        Passed to cprint() when printing box.
    """
    if width == 'content':
        width = max([len(m) for m in msg]) + 2
    elif width == 'console':
        width = get_console_width()
    if start:
        cprint('┏' + '━'*(width - 2) + '┓', **kwargs)
    for m in msg:
        cprint(f'┃{m:{align}{width - 2}}┃', **kwargs)
    if end:
        cprint('┗' + '━'*(width - 2) + '┛', **kwargs)
    return


def print_table(table, separator=' ', **kwargs):
    """
    Print table of data.

    Parameters
    ----------
    table : array
        2D list like object containing table rows data.

    separator : str
        String used to separate table columns

    **kwargs
        Passed to cprint() when printing table.
    """
    num_cols = max([len(row) for row in table])
    for idx, _ in enumerate(table):
        while len(table[idx]) < num_cols:
            table[idx].append('')
    widths = [max([len(str(cell)) for cell in [row[col_idx] for row in table]])
              for col_idx in range(len(table[0]))]
    for row in table:
        msg = ''
        for idx, cell in enumerate(row):
            msg += f'{cell:{widths[idx]}}{separator}'
        cprint(msg, **kwargs)


def progress_bar(progress, label='', full_width=None, max_width=160, new_line_complete=True,
                 trim_label=True, bg1='w', bg2='k', bg3=None, fg1='k', fg2='w', fg3=None, ):
    """
    Prints progress bar to nicely show the progress of code.

    >>> for x in range(101):
    >>>     # Some slow code
    >>>     progress_bar(x/100, 'Calculating...')

    The progress bar will continually overwrite itself, so ideally there should
    not be any other print statements in the loop where the progress bar is
    called.

    Parameters
    ----------
    progress : float
        Fractional progress

    label : str
        Message to put at start of progress bar

    full_width : int
        Width of progress bar. If None, uses result of get_console_width()-1.

    max_width : int
        Maximum width of progress bar. If full_width is None, sets progress bar
        width to min(get_console_width()-1, max_width).

    new_line_complete : bool
        Print a new line once the progress bar is complete

    trim_label : bool
        Toggle trimming of label if it is too long.

    bg1, bg2, bg3, fg1, fg2, fg3 : str
        Set colours for various parts of the progress bar, passed to cprint().
        1 = colour of done (left) part, 2 = colour of not done (right) part,
        3 = colour of bar once it is 100% complete.
    """
    now = datetime.now()

    if full_width is None:
        full_width = min(get_console_width() - 1, max_width)

    pct_len = 4
    percent_str = f'{progress:{pct_len}.0%}'
    time_str_len = full_width - len(label) - pct_len

    if progress < progress_bar.last_progress or progress_bar.start_dtm is None:
        progress_bar.start_dtm = now
        time_str = ' '*time_str_len
    else:
        total_seconds = (now - progress_bar.start_dtm).total_seconds()
        if progress < 1:
            total_seconds *= (1 - progress)/progress
        else:
            percent_str = ''
            time_str_len += pct_len

        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        end_dtm = now + timedelta(seconds=total_seconds)
        end_dtm = datetime.strftime(end_dtm, '%H:%M:%S')
        time_str = ''
        if hours:
            time_str += f'{hours:.0f}h {minutes:02.0f}m {seconds:20.0f}s'
        elif minutes:
            time_str += f'{minutes:.0f}m {seconds:02.0f}s'
        else:
            time_str += f'{seconds:.1f}s'

        if progress < 1:
            time_str += f' remaining ({end_dtm})'
        else:
            time_str = 'Done in ' + time_str
        if len(time_str) + 1 > time_str_len:
            time_str = time_str.replace(' remaining', '')
            time_str = time_str.replace('in ', '')
            if len(time_str) + 1 > time_str_len:
                time_str = time_str.split('(')[0].strip()
                time_str = time_str.replace('Done ', '').strip()
                if len(time_str) + 1 > time_str_len:
                    time_str = time_str.replace(' ', '')
                    if len(time_str) + 1 > time_str_len:
                        time_str = ' ' + time_str
                        time_str_len = len(time_str)
                        if trim_label:
                            if progress < 1:
                                label_len = full_width - time_str_len - pct_len - 1
                            else:
                                label_len = full_width - time_str_len - 1
                            label = label[:label_len] + '…'
        if label:
            if len(time_str) < 0.66*time_str_len:
                # Center in space
                time_str = f'{time_str:>{time_str_len}s}'
            else:
                time_str = f'{time_str:>{time_str_len}s}'
        else:
            time_str = f'{time_str:<{time_str_len}s}'

    msg = f'{label}{time_str}{percent_str}'
    idx = int(progress*full_width)
    if progress >= 1:
        bg1 = bg3
        fg1 = fg3
    msg = ('\r'
           + cprint(msg[:idx], bg=bg1, fg=fg1, return_str=True)
           + cprint(msg[idx:], bg=bg2, fg=fg2, return_str=True))
    print(msg, flush=True, end='')
    progress_bar.last_progress = progress
    if progress >= 1 and new_line_complete:
        print()


progress_bar.start_dtm = None
progress_bar.last_progress = float('inf')


def progress_bar_basic(progress, start='', end='', percentage=True, numbers=False,
                       max_width=None, complete='█', empty=' ', bar_width=None):
    """
    Print progress bar which can be updated.

    Parameters
    ----------
    progress : float
        Fractional progress between 0 & 1.

    start, end : str
        String to prepend/append to progress bar.

    percentage : bool
        Toggle showing percentage complete at end of progress bar.

    numbers : bool
        Toggle showing numbers for fractional completion within a box.

    max_width : int
        Maximum allowed width of progress bar, set to `None` to use
        `get_console_width()`.

    complete : str
        Character to use for completed part of progress bar.

    empty : str
        Character to use for empty part of progress bar.

    bar_width : int
        Fixed width for bar (overrides other values).

    See Also
    --------
    progress_bar() - more fancy version with more useful info.
    """
    auto_bar_width = bar_width is None
    if auto_bar_width:
        if max_width is None:
            max_width = get_console_width()
        bar_width = max_width - 2 - len(start) - len(end)
    if percentage:
        percentage = '{:2d}%'.format(int(100*progress))
        bar_width -= 3
    else:
        percentage = ''
    steps_done = int(bar_width*progress)
    steps_left = bar_width - steps_done
    if numbers:
        numbers = str(int(10*((progress*bar_width) % 1)))
        steps_left -= 1
    else:
        numbers = ''
    if progress >= 1:
        numbers = ''
        if percentage and auto_bar_width:
            steps_done = bar_width - 1
    msg = ''.join(['\r', start,
                   '|', complete*steps_done, numbers, empty*steps_left, '|',
                   percentage, end])
    print(msg, end='', flush=True)
    if progress >= 1:
        print()


def print_bar_chart(labels, bars=None, formats=None, print_values=True, max_label_length=None,
                    sort=False, **kwargs):
    """
    Print bar chart of data

    Parameters
    ----------
    labels : array
        Labels of bars, or bar values if `bars is None`.

    bars : array
        List of bar lengths.

    formats : array
        List of bar formats to be passed to `cprint()`.

    print_values : bool
        Toggle printing bar lengths.

    max_label_length : int
        Set length to trim labels, None for no trimming.

    sort : bool
        Toggle sorting of bars by size.

    **kwargs
        Arguments passed to `cprint()` for every bar.
    """
    if bars is None:
        bars = labels.copy()
        labels = ['' for _ in bars]
    if max_label_length is None:
        max_label_length = max([len(l) for l in labels])
    else:
        labels = clip_string_list(labels, max_label_length)
    labels = [f'{l:{max_label_length}s}' for l in labels]
    if sort:
        bars, labels = zip(*sorted(zip(bars, labels)))
    if print_values:
        fmt = '.2e'
        if isinstance(print_values, str):
            fmt = print_values
        labels = [f'{l}|{v:{fmt}}' for l, v in zip(labels, bars)]
    max_label_length = max([len(l) for l in labels])
    max_length = get_console_width() - max_label_length - 2
    for idx, label in enumerate(labels):
        kw = {**kwargs}
        if formats:
            if formats == 'auto':
                if bars[idx]/sum(bars) > 0.5:
                    kw.update(fg='y', style='b')
                elif bars[idx]/sum(bars) > 0.1:
                    kw.update(fg='g', style='b')
                elif bars[idx]/sum(bars) > 0.01:
                    kw.update(fg='b', style='b')
                else:
                    kw.update(fg='w', style='f')
            elif formats == 'extreme':
                if bars[idx] == max(bars):
                    kw.update(fg='g', style='b')
                elif bars[idx] == min(bars):
                    kw.update(fg='r', style='b')
                else:
                    kw.update(fg='b', style='b')
            else:
                kw.update(formats[idx])
        bar = '█'*int(max_length*bars[idx]/max(bars))
        cprint(f'{label:{max_label_length}s}|{bar}', **kw)


def compare_strings(s1, s2):
    """
    Compares two strings and prints fancy output.

    Parameters
    ----------
    s1, s2 : str
        Strings to compare
    """
    print(s1)
    for idx, c in enumerate(s2):
        if s1[idx] == c:
            cprint(c, fg='c', end='')
        else:
            cprint(c, bg='r', style='b', end='')
    print()


def match_start(msg, options, fallback=None):
    """
    Tries to match a string with a set of strings by matching the start.

    >>> tools.script.match_start('a', ['abc', 'def', 'ghi'])
    'abc' # One clear choice
    >>> tools.script.match_start('a', ['abc', 'aaa'])
    None # No unambiguous choice

    Parameters
    ----------
    msg : str
        String to try to match.

    options : list of str
        List of strings to try to match.

    fallback
        Return value if no certain match found.

    Returns
    -------
    str
        Returns option chosen, or fallback if none have been founds

    """
    # Check if actual option
    if not msg:
        return fallback
    if msg in options:
        return msg
    matches = [o for o in options if o.startswith(msg)]
    if len(matches) == 1:
        return matches[0]
    matches = [o for o in options if o.casefold().startswith(msg.casefold())]
    if len(matches) == 1:
        return matches[0]
    return match_start(msg[:-1], options, fallback)


def get_colormap(values, cmap='jet', colorbar=True, log=False):
    """
    Gets a Matplotlib colormap and returns colour values.

    Useful for use in plotting multiple lines with different colours, e.g.:

    >>> colors = get_colormap(z)
    >>> for y, c in zip(y_list, colors):
    >>>     plt.plot(x, y, color=c)

    Parameters
    ----------
    values : list or int
        List of values for colorbar or int of length of required colour values.

    cmap : str
        Matplotib colormap name.

    colorbar : bool or str
        Toggle plotting of colorbar, provide str for the colorbar label.

    log : bool
        Toggle logarithmic color scale.

    Returns
    -------
    colors : array
        List of colors.
    """
    by_value = not isinstance(values, int)
    if not by_value:
        colorbar = False
    if by_value:
        cmap = plt.get_cmap(cmap)
        if log:
            colors = [cmap(x) for x in tools.maths.normalise(np.log10(values))]
        else:
            colors = [cmap(x) for x in tools.maths.normalise(np.array(values))]
    else:
        cmap = plt.get_cmap(cmap, values)
        colors = [cmap(x) for x in range(values)]
    if colorbar:
        if log:
            norm = mpl.colors.LogNorm(vmin=min(values), vmax=max(values))
        else:
            norm = mpl.colors.Normalize(vmin=min(values), vmax=max(values))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        if isinstance(colorbar, str):
            label = colorbar
        else:
            label = ''
        plt.colorbar(sm, label=label)
    return colors


def truncate_colormap(cmap, minval=0, maxval=1, n=256):
    """
    Truncates matplotlib colormap to remove start/end values.

    Useful to remove e.g. white parts of a colormap for plotting on a white
    background.

    Parameters
    ----------
    cmap : str
        Name of colormap.

    minval : float
        Fraction to trim start of colormap (0 = no start trim).

    maxval : float
        Fraction to trim end of colormap (1 = no end trim).

    n : int
        Number of colormap points to produce

    Returns
    -------
    new_cmap
        Truncated colormap
    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def extract_key_value_pairs(string, joiner='=', separator=','):
    """
    Convert a string of key, value pairs into a dict.

    >>> extract_key_value_pairs('a:1& b:2', joiner=':', separator='&')
    {'a': '1', 'b': '2'}

    Parameters
    ----------
    string : str
        Input string containing data.

    joiner : str
        String between key and value.

    separator : str
        String between one key, value pair and the next.

    Returns
    -------
    dict
    """
    return dict([x.strip() for x in s.split(joiner, 1)] for s in string.split(separator))


def shuffle_list(a):
    """
    Shuffles iterable into random order.

    Parameters
    ----------
    a : iterable or int
        Iterable to shuffle. Provide an int to shuffle range(a).

    Returns
    -------
    Shuffled list.
    """
    if isinstance(a, int):
        a = range(a)
    a = copy.copy(a)
    try:
        random.shuffle(a)
    except TypeError:
        a = list(a)
        random.shuffle(a)
    return a


def distribute_list(a):
    """
    Shuffle list into a well distributed order.

    The values will be chosen so that subsequent values are separated by
    (golden ratio)*len(a) in the original list, i.e. values near each other in
    the original list are far from each other in the final list.

    This can be useful to provide a good representation of values during a
    a calculation where the intermediate results are displayed to the user by
    increasing the resolution of a calculation rather than the range of it.

    >>> for x in distribute_list(range(100)):
    >>>     plt.scatter(x, fn(x))
    >>>     plt.show(block=False)

    See https://en.wikipedia.org/wiki/Golden_ratio for more.

    Parameters
    ----------
    a : iterable or int
        Iterable to shuffle. Provide an int to shuffle range(a).

    Returns
    -------
    Distributed list.
    """
    if isinstance(a, int):
        a = range(a)
    out_idxs = []
    ratio = (0.5*(1 + np.sqrt(5)) - 1)*len(a)  # Golden fraction
    idx = -ratio
    while len(out_idxs) < len(a):
        idx += ratio
        idx %= len(a)
        idx_int = int(idx)
        while idx_int in out_idxs:
            idx_int += 1
            idx_int %= len(a)
        out_idxs.append(idx_int)
    out = copy.copy(a)
    try:
        for idx_out, idx_a in enumerate(out_idxs):
            out[idx_out] = a[idx_a]
    except TypeError:
        out = list(copy.copy(a))
        for idx_out, idx_a in enumerate(out_idxs):
            out[idx_out] = a[idx_a]

    return out


def clip_string_list(a, max_len, continue_str='…'):
    """
    Takes a list of strings and clips them to a certain length if needed.

    Parameters
    ----------
    a : list of str

    max_len : int
        Maximum length of strings allowed

    continue_str : str
        String to indicate clipping.

    Returns
    -------
    clipped list
    """
    return [x if len(x) <= max_len else x[:max_len - len(continue_str)] + '…' for x in a]


def split_string_at_numbers(string):
    """
    Splits string into text and numerical contents.

    Attempts to correct units, e.g. 1mm -> 0.001.

    Parameters
    ----------
    string : str

    Returns
    -------
    list
        List of strings and numbers
    """
    split_list = re.compile(r'(\d+)').split(string)
    filtered_list = []
    skip_next_loops = 0
    for i in range(len(split_list)):
        if skip_next_loops > 0:
            skip_next_loops -= 1
            continue
        part = split_list[i]
        if part.isdigit() or (part == '.' and i < len(split_list) - 1 and split_list[i + 1].isdigit()):
            # Some kind of number
            if part == '.':
                # number of format '.###' (start of string)
                part += split_list[i + 1]
                skip_next_loops = 1
            elif i < len(split_list) - 2 and split_list[i + 1] == '.' and split_list[i + 2].isdigit():
                # number of format '###.###'
                part += split_list[i + 1] + split_list[i + 2]
                skip_next_loops = 2
            elif (i > 0 and len(filtered_list) and len(filtered_list[-1]) and
                  filtered_list[-1][-1] == '.'):
                # number of format '.###' (within string)
                filtered_list[-1] = filtered_list[-1][:-1]
                part = '.' + part
            # otherwise just number of format '###'
            factor = 1
            if i < len(split_list) - 1:
                # check next part for unit information
                msg = split_list[i + 1].strip()
                msg = msg.lstrip('_([{')
                msg = re.split('[^a-zA-Zµ]', msg)[0]
                if msg:
                    for unit in tools.science.UNIT_SYMBOLS:
                        if msg.endswith(unit):
                            msg = msg[:-len(unit)]
                            break
                    if len(msg) == 1:
                        factor = 10**tools.science.SI_PREFIXES.get(msg[0], 0)
            filtered_list.append(float(part)*factor)
        else:
            # Actual string
            filtered_list.append(part)
    return filtered_list


def sort_mixed(iterable):
    """
    Sort iterable containing strings with numerical values.

    More intelligent than bare sort() function

    >>> sorted(['2m', '10m', '30m', '5mm']) # Sorts by character
    ['10m', '2m', '30m', '5mm']
    >>> sort_mixed(['2m', '10m', '30m', '5mm']) # Uses actual values
    ['5mm', '2m', '10m', '30m']

    Parameters
    ----------
    iterable : list

    Returns
    -------
    sorted list
    """
    return sorted(iterable, key=lambda x: split_string_at_numbers(x))


def flatten_dict(d, sep=' ', parent_key=''):
    """
    Flattens nested dictionary into single dictionary with no nesting.

    >>> flatten_dict({'a':{'b':1, 'c':2}, 'b':3}, sep='-')
    {'a-b': 1, 'a-c': 2, 'b': 3}

    Parameters
    ----------
    d : dict
        Dictionary to flatten

    sep : str
        Separator to create flattened dictionary keys

    parent_key : str
        String to prepend to keys (for recursive use within this function).

    Returns
    -------
    dict
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, sep=sep, parent_key=new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def parallel(fn, args=None, kwargs=None, loop_kw=None, spare_cpu=1, num_processes=None,
             do_parallel=True):
    """
    Perform mapping in parallel to increase execution speed.

    Uses Pool().map(...) under the hood to split mapping over multiple
    processes. Generally best for CPU bound tasks. This function handles
    both constant and changing arguments of mapping to extend standard map()
    procedure.

    Uses _parallel_wrapper() to wrap function calls for each loop and handle
    args and kwargs correctly.

    Parameters
    ----------
    fn : function
        Function to call on each loop of iteration.

    args : list
        Tuple (or other iterable) containing arguments that will be passed to
        fn(). These are assumed to all be varying, i.e. each element of args
        should have a length equal to the number of iterations.

    kwargs : dict
        Key word arguments to be passed to fn(). These are assumed to be
        constant (i.e. have single values unchanging for different iterations)
        unless specified to be varying in loop_kw.

    loop_kw : str or list
        Key words in kwargs which are varying in different iterations.

    Returns
    -------
    list
        List of results of loop.

    Other Parameters
    ----------------
    spare_cpu : int
        Number of CPUs to keep free (if num_processes = None) when number of
        processes is calculated as num_processes = os.cpu_count() - spare_cpu.

    num_processes : int
        Number of processes to use in parallel loop, overrides spare_cpu if
        set.

    do_parallel : bool
        Toggle is Pool.map() or map() should be used. Mainly for testing
        purposes, but also possibly for other cases where easy switching
        between parallel and serial execution is desirable.

    See Also
    --------
    _parallel_wrapper()
        Function which is called for each iteration to handle arg and kwarg
        wrapping.

    Examples
    --------
    Equivalent serial (list comprehension) and parallel cases:

    >>> [pow(a, b) for a, b in zip([1,2,3], [1,2,3])]
    >>> parallel(pow, args=([1, 2, 3], [1, 2, 3]))

    >>> [f(x, a=5) for x in range(10)]
    >>> parallel(f, args=(range(10),), kwargs=dict(a=5))

    >>> [f(x=x, a=5) for x in range(10)]
    >>> parallel(f, kwargs=dict(x=range(10), a=5), loop_kw='x')
    """
    if num_processes is None:
        num_processes = os.cpu_count() - spare_cpu  # Default to defining a number of spare CPUs
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    # Set up wrapper for arguments for each iteration of the loop
    try:
        za = zip(*args)
    except TypeError:
        args = (args,)
        za = zip(*args)
    wrapper_arg = [(fn, a, kwargs.copy()) for a in za]
    if loop_kw is not None:
        # Deal with kwargs with different parameters for each loop
        if isinstance(loop_kw, str):
            loop_kw = [loop_kw]
        if len(kwargs[loop_kw[0]]) > len(wrapper_arg):
            # Args are non existent
            wrapper_arg = [(fn, (), kwargs.copy()) for _ in range(len(kwargs[loop_kw[0]]))]
        for idx, wa in enumerate(wrapper_arg):
            wa = list(wa)
            for kw in loop_kw:
                wa[2][kw] = kwargs[kw][idx]
            wrapper_arg[idx] = tuple(wa)

    # Perform actual loop
    do_parallel = do_parallel and num_processes > 1
    if do_parallel:
        with Pool(num_processes) as pool:
            return pool.map(_parallel_wrapper, wrapper_arg)
    else:
        return list(map(_parallel_wrapper, wrapper_arg))

def _parallel_wrapper(wrapper_arg):
    fn, args, kwargs = wrapper_arg
    return fn(*args, **kwargs)
