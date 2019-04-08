"""
Module for file operations.

Contains methods to deal with file reading, writing, analysing and processing.
"""
import copy
import fnmatch
import glob
import io
import os
import pathlib
import re
import shutil
import subprocess
import warnings
from datetime import datetime

import astropy
import astropy.io.fits as fits
import matplotlib.pyplot as plt

import tools


checksum = tools.script.checksum
cprint = tools.script.cprint
shell = tools.script.shell


class ReadError(Exception):
    """
    Error raised when reading in data files.
    """
    def __init__(self, value):
        self.parameter = value

    def __str__(self):
        return repr(self.parameter)


def check_path(path):
    """
    Checks if file path's directory tree exists, and creates it if necessary.

    Assumes path is to a file if `os.path.split(path)[1]` contains '.',
    otherwise assumes path is to a directory.

    Parameters
    ----------
    path : str
        Path to directory to check.
    """
    if os.path.isdir(path):
        return
    if '.' in os.path.split(path)[1]:
        check_path(os.path.split(path)[0])
        return
    if path == '':
        return
    print('Creating directory path "{}"'.format(path))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def compress_file(path, compression_type='gz', keep_input=False, force_output=True,
                  print_command=False):
    """
    Compresses file using specified compression algorithm.

    File compressed by calling e.g. `gzip ...` in the system shell with
    appropriate flags.

    Parameters
    ----------
    path : str
        Path to file which is to be compressed.

    compression_type : {'gz', 'z'}
        Specify compression type.

    keep_input : bool
        Toggle if uncompressed file is kept (only relevant for gzip).

    force_output : bool
        Toggle if compressed file will overwrite existing files.

    print_command : bool
        Toggle if shell command is printed to user.
    """
    compression_type = compression_type.lower()
    if compression_type in ['z', 'compress']:
        command = 'compress '
        keep_input = False
    else:
        command = 'gzip '

    command += path

    if keep_input:
        command += ' -k'
    if force_output:
        command += ' -f'

    shell(command, print_command=print_command)


def soft_delete(path):
    """
    Moves file to local trash directory so 'deletion' can be undone by user.

    Useful as standard `os.remove()` deletes file immediately (i.e. file is not
    sent to Trash/Recycle Bin). File name in `trash/` is same as original file.
    If a duplicate exists in `trash/`, file name is appended with the current
    datetime.

    Parameters
    ----------
    path : str
        Path to file that should be deleted.
    """
    check_path(tools.path.TRASH)
    file_name = os.path.split(path)[-1]
    trash_list = os.listdir(tools.path.TRASH)
    if file_name in trash_list:
        file_name += ' ' + str(datetime.now()) + '.' + file_name.split('.')[-1]
    path_end = tools.path.trash(file_name)

    shutil.move(path, path_end)

    metadata_path = path_end + '.trash'
    with open(metadata_path, 'w') as f:
        f.write(f'DELETION CWD:  {os.getcwd()}\n')
        f.write(f'ORIGINAL PATH: {path}\n')
        f.write(f'TRASH PATH:    {path_end}\n')
        f.write(f'DELETION DTM:  {str(datetime.now())}\n')
        f.write(f'METADATA PATH: {metadata_path}\n')


#%% FITS ------------------------------------------------------------------------------------------

def read_fits(item, output='both', **kwargs):
    """
    Convenience function to read FITS image.

    If `fits` is a string, it is passed to `read_compressed_fits` as the path
    of a FITS file to read. Otherwise passes `fits` straight through. Useful
    for functions where FITS file may possibly be already loaded, but may not
    be.

    Parameters
    ----------
    item
        Image or path to FITS file.

    output : str
        Type of data to output, see `read_compressed_fits()`.

    **kwargs
        Additional arguments passed to `read_compressed_fits()`.

    Returns
    -------
    fits
        FITS data/header etc.
    """
    if isinstance(item, str):
        item = read_compressed_fits(item, output=output, **kwargs)

    return item


def read_compressed_fits(filename, output='both', compression_type=None, ignore_missing_end=True,
                         try_extensions=False, **kwargs):
    """
    Read FITS file, uncompressing file if necessary.

    Capable of reading `.fits`, `.fits.gz` and `.fits.Z` files. `.fits.Z` files
    are uncompressed using `uncompress -c <filename>` shell command. Function
    will attempt to read file with all compression types supported, and raise
    error otherwise.

    Parameters
    ----------
    filename : str
        Relative path to file to open.

    output : {'both', 'header', 'data'/'img'}, optional
        Choose output of function.

    compression_type : str
        Option to explicitly define type of compression used to read file.

    ignore_missing_end : bool
        Passed to `fits.getdata()`.

    try_extensions : bool
        Toggle if should try to find file with '.fits', '.fits.Z' or '.fits.gz'
        extension.

    **kwargs
        Additional arguments passed to `fits.getdata()`.

    Returns
    -------
    image, header
        Tuple of image and header data from FITS file or output specified.
    """
    if try_extensions:
        bare_fn = filename.replace('.fits.Z', '').replace('.fits.gz', '').replace('.fits', '')
        fn_list = [filename] + [bare_fn + ext for ext in ['.fits', '.fits.Z', '.fits.gz']]
        for fn in fn_list:
            if os.path.isfile(fn):
                return read_compressed_fits(fn,
                                            output=output,
                                            compression_type=compression_type,
                                            ignore_missing_end=ignore_missing_end,
                                            **kwargs)
        else:
            # Run for final time to raise error
            return read_compressed_fits(filename,
                                        output=output,
                                        compression_type=compression_type,
                                        ignore_missing_end=ignore_missing_end,
                                        **kwargs)
    if compression_type is None:
        # If compression not specified, intelligently cycle through options to try to read file
        file_extension = filename.split('.')[-1].lower()
        compression_list = ['fits', 'open', 'z']
        if file_extension == 'z':
            compression_list = ['z', 'open', 'fits']
        for compression_type in compression_list:
            try:
                return read_compressed_fits(filename,
                                            output=output,
                                            compression_type=compression_type,
                                            ignore_missing_end=ignore_missing_end,
                                            **kwargs)
            except subprocess.CalledProcessError:
                pass
            except OSError as e:
                if str(e) != 'Empty or corrupt FITS file':
                    raise
        else:
            # Run for final time to raise error
            return read_compressed_fits(filename,
                                        output=output,
                                        compression_type='fits',
                                        ignore_missing_end=ignore_missing_end,
                                        **kwargs)

    # Use specified compression to try to read file
    compression_type = compression_type.lower()
    if compression_type == 'z':
        process = subprocess.run(['uncompress', '-c', filename],
                                 stdout=subprocess.PIPE, check=True)
        file = io.BytesIO(process.stdout)
    elif compression_type == 'open':
        with open(filename, mode='rb') as f:
            file = f.read()
            file = io.BytesIO(file)
    else:
        # Read file directly
        file = filename

    if output in ['header', 'hdr']:
        return fits.getheader(file, ignore_missing_end=ignore_missing_end, **kwargs)
    elif output in ['data', 'image', 'img']:
        return fits.getdata(file, ignore_missing_end=ignore_missing_end, **kwargs)
    else:
        return fits.getdata(file, ignore_missing_end=ignore_missing_end, header=True, **kwargs)


def write_fits(file_path, image_data, header_data, compression=False):
    """
    Write data to FITS file.

    Parameters
    ----------
    file_path : str
        File path where file will be written.

    image_data
        Image data for FITS file.

    header_data
        Header data for FITS file.

    compression : bool or str
        Optionally compress FITS file, define compression type for
        `compress_file()`.
    """
    check_path(file_path)
    with warnings.catch_warnings():
        # Suppress warning about comments being truncated
        warnings.filterwarnings('ignore',
                                message='Card is too long, comment will be truncated.',
                                module='astropy.io.fits.card')
        fits.writeto(file_path, image_data, header=header_data, overwrite=True, checksum=True)
    if compression:
        compress_file(file_path, compression)


def add_header_reduction(header, description=None, script_file=None, source_file=None,
                         target_file=None, date='now', directory='cwd', sep=True, **kwargs):
    """
    Adds standard reduction header information to FITS header object.

    Header keys will be prefaced with `find_reduction_number()` string. Use
    `__file__` from calling script to set `script_file` to usual desired value.

    Parameters
    ----------
    header
        Header object to update.

    date, description, directory, script_file, source_file, target_file : str
        Header items to add to header object. Items set to `None` are not
        included in header. `date` defaults to the current datetime ('now'),
        and will be formatted using `datetime.isoformat()` if is datetime.
        `cwd` defaults to `os.getcwd()`. All others default to `None`. Paths
        are split using `os.path.split()`.

    sep : bool
        Toggle separator at start of reduction info for ease of reading
        header.

    **kwargs
        Specify additional arguments in dictionary. Values can be strings or
        tuple of value and comment. Keys are transformed to upper case and
        underscores replaced with spaces. Value can also be dict of key-value
        pairs for `add_header_item()`.

    Returns
    -------
    header
        Object with updated metadata info.
    """
    header = copy.deepcopy(header)
    re_str = find_reduction_number(header)

    if sep:
        msg = '*'*80
        add_header_item(header, re_str+'SEP', msg, overflow_str='')
    if date:
        if date == 'now':
            date = datetime.now()
        if isinstance(date, datetime):
            date = datetime.isoformat(date)
        add_header_item(header, re_str+'DATE', date, 'Date reduction completed')
    if description:
        add_header_item(header, re_str+'TYPE', description, 'Type of reduction performed')
    if directory:
        if directory == 'cwd':
            directory = os.getcwd()
        add_header_item(header, re_str+'DIR', directory, 'Working directory of reduction',
                        keep_start=False)
    if script_file:
        p, f = os.path.split(script_file)
        add_header_item(header, re_str+'SCR PATH', p, 'Path to script location used in reduction',
                        keep_start=False)
        add_header_item(header, re_str+'SCR FILE', f, 'Filename of script used in reduction')
    if source_file:
        p, f = os.path.split(source_file)
        add_header_item(header, re_str+'SOU PATH', p, 'Path to source FITS file location',
                        keep_start=False)
        add_header_item(header, re_str+'SOU FILE', f, 'Filename of source FITS file')
    if target_file:
        p, f = os.path.split(target_file)
        add_header_item(header, re_str+'TAR PATH', p, 'Path to reduced FITS file location',
                        keep_start=False)
        add_header_item(header, re_str+'TAR FILE', f, 'Filename of reduced FITS file')
    for key in kwargs:
        key_str = key.upper().replace('_', ' ')
        value = kwargs[key]

        if isinstance(value, dict):
            add_header_item(header, re_str+key_str, **value)
        else:
            add_header_item(header, re_str+key_str, value)
    return header


def add_header_item(header, key, value, comment=None, overflow_str='...', keep_start=True,
                    copy_header=False, ispath=False, islist=False, pop=True):
    """
    Adds item to FITS header, checking correcting value length issues.

    If value is a string, automatically checks it is not too long. If string is
    too long, the start or end of the string (depending on `keep_start`) will
    be replaced by `overflow_str` such that the shortened value string now fits
    in the 80 char limit for FITS headers. If `copy_header=False`, header is
    updated in place. If `copy_header=True`, the header is deep copied before
    being updated, and the new header is returned.

    Parameters
    ----------
    header
        Header object to add item to.

    key : str
        Header key.

    value : str
        Header value. If `comment=None`, this can be a tuple of form
        `(value, comment)`.

    comment : str
        Header comment.

    overflow_str : str
        String used to indicate value has been shortened.

    keep_start : bool
        Toggle between removing start/end of long value strings.

    copy_header : bool
        Toggle between updating header in place or returning copy of header.

    ispath : bool
        Toggle treating input as path (uses default options).

    islist : bool
        Toggle treating input as list (splits over multiple header lines).

    pop : bool
        Completely replace existing value (so position and comment aren't
        retained).

    Returns
    -------
    header
        If `copy_header=True`, returns updated header object.
    """
    if ispath:
        if comment is None:
            comment = ''
        p, f = os.path.split(value)
        header = add_header_item(header, key+' PATH', p, 'Path to '+comment+' location',
                                 keep_start=False, pop=pop, copy_header=copy_header)
        header = add_header_item(header, key+' FILE', f, 'Filename of '+comment, pop=pop,
                                 copy_header=copy_header)
        return header
    if islist:
        header = add_header_item(header, key, f'<list>', comment, overflow_str=overflow_str,
                                 copy_header=copy_header, pop=pop)
        for idx, v in enumerate(value):
            k = key + str(idx)
            c = '<list element>'
            header = add_header_item(header, k, v, c, overflow_str=overflow_str,
                                     copy_header=copy_header, pop=pop)
        return header
    if copy_header:
        header = copy.deepcopy(header)
    if comment is None:
        if isinstance(value, tuple) or isinstance(value, list):
            comment = value[1]
            value = value[0]
    if isinstance(value, str):
        key = key.strip()
        max_val_len = 80 - len(key) - 4  # 4 accounts for equals, space and two quote marks
        max_val_len -= value.count("'") # Account for conversion quotes in FITS (' -> '')
        slice_len = max_val_len - len(overflow_str)
        if len(value) > max_val_len:
            if keep_start:
                value = value[:slice_len] + overflow_str
            else:
                value = overflow_str + value[-slice_len:]
    if comment is None:
        header[key] = value
    else:
        with warnings.catch_warnings():
            # Suppress warning about comments being truncated
            warnings.filterwarnings('ignore',
                                    message='Card is too long, comment will be truncated.',
                                    module='astropy.io.fits.card')
            if pop:
                header.pop(key, None) # Completely replace existing value
            header[key] = (value, comment)

    return header


def summarise_fits_in_dir(root_path='.', header_keys=None, required_filename=None,
                          required_value=None, custom_widths=None, check_data=True,
                          error_files='show', read_subdirectories=False, show_img=False):
    """
    Prints summary of FITS files in directory.

    Required values use unix filename matching with *=wildcard. ! at the start
    of a string negates it.

    Parameters
    ----------
    root_path : str or list
        Path to directory which will be summarised.

    header_keys : str or list
        Keys to read from FITS file header to display in output table. Can be
        `None` or `'ifs_calibration_keys'` for various defaults.

    required_filename : str
        String to filter files displayed in summary. Uses unix filename
        matching.

    required_value : dict
        Dict of key, value pairs that must/must not exist in header for it to
        be printed. Uses unix filename matching.

    custom_widths : iterable
        Widths of each column of summary.

    check_data : bool
        Try loading data for FITS files as well as header.

    error_files : bool
        Options for printing error file info.

    read_subdirectories : bool
        Toggle to read all subdirectories in given root_path.

    show_img : bool
        Display example image from FITS file.
    """
    if required_value is None:
        required_value = {}
    if custom_widths is None:
        custom_widths = {}
    # Deal with recursive call modes
    if not isinstance(root_path, str):
        for path in root_path:
            summarise_fits_in_dir(path,
                                  header_keys=header_keys,
                                  required_filename=required_filename,
                                  required_value=required_value,
                                  custom_widths=custom_widths,
                                  check_data=check_data,
                                  error_files=error_files,
                                  read_subdirectories=read_subdirectories,
                                  show_img=show_img)
        return

    # Find FITS files to read and set up for current directory
    root_path = os.path.abspath(root_path)
    cprint('\nSummarising FITS files in {}'.format(root_path), end='', fg='w', style='i')

    if read_subdirectories:
        file_list = glob.glob(os.path.join(root_path, '**'), recursive=True)
        cprint(' and subdirectories', fg='w', style='i')
    else:
        file_list = os.listdir(root_path)
        print()
    file_list = [n for n in file_list if (n.lower()[-5:] == '.fits'
                                          or n.lower()[-7:] == '.fits.z'
                                          or n.lower()[-8:] == '.fits.gz')]
    file_list.sort()
    cprint('  {} FITS files found'.format(len(file_list)), fg='w', style='i')

    if required_filename:
        if required_filename[0] != '!':
            file_list = fnmatch.filter(file_list, required_filename)
        else:
            file_list = [n for n in file_list if not fnmatch.fnmatch(n, required_filename[1:])]
        cprint('  {} with filename containing "{}"'.format(len(file_list), required_filename),
               fg='w', style='i')
    if len(file_list) == 0:
        return

    column_widths = {'OBJECT': 16}
    column_widths.update(custom_widths)
    if header_keys is None:
        header_keys = ['NAXIS1', 'NAXIS2', 'NAXIS3', 'CTYPE3', 'EXPTIME', 'OBJECT',
                       'ESO OBS TARG NAME', 'ORIGFILE']
    elif header_keys == 'ifs_calibration_keys':
        header_keys = ['EXPTIME', 'OBJECT', 'ESO INS2 COMB IFS', 'ESO INS2 FILT NAME',
                       'ESO INS2 OPTI2 ID', 'ESO INS2 CAL', 'ORIGFILE']
    elif isinstance(header_keys, str):
        header_keys = [header_keys]
    for key in header_keys:
        # Default column widths to 1 + key name length
        if key not in column_widths:
            column_widths[key] = len(key) + 1

    max_file_len = max([len(n) for n in file_list]) + 3
    if max_file_len <= len('Filename'):
        max_file_len = len('Filename') + 1
    header_str = cprint('Filename' + ' '*(max_file_len - 9),
                        fg='k', bg='w', return_str=True) + ' '
    for key in header_keys:
        header_str_local = key
        w = column_widths[key]
        header_str_local += ' '*(w - 1 - len(key))
        header_str_local = cprint(header_str_local, fg='k', bg='w', return_str=True)
        header_str += header_str_local + ' '
    print()
    print(header_str)

    skipped_files = 0
    for filename in file_list:
        print(filename + ' '*(max_file_len-len(filename)), end='')
        skip_file = False

        # Load data
        path = os.path.join(root_path, filename)
        img, header = [], []
        try:
            if check_data or show_img:
                img, header = read_compressed_fits(path, output='both')
            else:
                header = read_compressed_fits(path, output='header')

            if error_files == 'only':
                skip_file = True
        except Exception as e:
            if error_files == 'hide':
                skip_file = True
            else:
                cprint('ERROR: ' + str(e), fg='r', style='b')
                continue

        # Skip unwanted files
        for key in required_value:
            if key not in header:
                skip_file = True
                break
            elif (required_value[key] and required_value[key][0] != '!' and not
                  fnmatch.fnmatch(str(header[key]), required_value[key])):
                skip_file = True
                break
            elif (required_value[key] and required_value[key][0] == '!' and
                  fnmatch.fnmatch(str(header[key]), required_value[key][1:])):
                skip_file = True
                break

        if skip_file:
            print('\r' + ' '*max_file_len + '\r', end='')
            skipped_files += 1
            continue

        # Print useful information
        for key in header_keys:
            w = column_widths[key]
            overflow = False

            try:
                value_str = header[key]
                if key != header_keys[-1]:
                    # Format strings appropriately
                    if isinstance(value_str, str):
                        value_str = value_str.strip()
                    elif isinstance(value_str, int):
                        value_str = '{:<{w}g}'.format(value_str, w=w-1)
                        if int(value_str) != header[key]:
                            value_str = '{:<{w}g}'.format(header[key], w=w-2)
                            overflow = True
                    elif isinstance(value_str, float):
                        value_str = '{:<{w}g}'.format(value_str, w=w-1)
                        if float(value_str) != header[key]:
                            value_str = '{:<{w}g}'.format(header[key], w=w-2)
                            overflow = True
                    else:
                        value_str = str(value_str)

                    if len(value_str) > w-1:
                        # Catch long strings and long numbers
                        value_str = value_str[:w-2]
                        overflow = True

                if overflow:
                    # Show overflow explicitly
                    print(value_str, end='')
                    cprint('… ', fg='g', style='b', end='')
                else:
                    print(value_str + ' '*(w-len(value_str)), end='')
            except KeyError:
                cprint('[none]' + ' '*(w-6), sfg='b', end='', style='b')
        if show_img or show_img == 0:
            while len(img.shape) > 2:
                img = img[0]
            if show_img is True:
                percentile = 5
            else:
                percentile = show_img
            tools.image.show_image(img, title=filename, colorbar=True, percentile=percentile)
        print()
    print(header_str)

    if required_value or skipped_files:
        cprint('\n{} / {} files ({}%) skipped due to value and error conditions:'.format(
                    skipped_files, len(file_list), int(100*skipped_files/len(file_list))),
               fg='w', style='i')
        for key in required_value:
            cprint('  {} : {}'.format(key, required_value[key]), fg='w', style='i')
        cprint('  error_files = {}'.format(error_files), fg='w', style='i')
    print()


def get_header_lists(hdr, return_ignore_keys=False):
    """
    Gets lists (created by add_header_item()) from header object.

    Parameters
    ----------
    hdr : FITS header

    return_ignore_keys : bool
        Toggle returning of keys which contain list info.

    Returns
    -------
    hdr_lists : dict of lists
        Dictionary of header lists.
    """
    keys = [k for k in hdr.keys() if len(k) > 0]
    hdr_lists = {}
    ignore_keys = []
    for idx, key in enumerate(keys):
        if hdr[key] == '<list>':
            list_values = {k: hdr[k] for k in tools.script.sort_mixed(keys)
                           if (k.startswith(key) and k[len(key):].isdigit())}
            if len(list_values) - 1 != int(list(list_values.keys())[-1][len(key):]):
                # Don't have whole list or some other strange thing
                break
            hdr_lists[key] = [v for v in list_values.values()]
            ignore_keys.extend(list_values.keys())
    if return_ignore_keys:
        return hdr_lists, ignore_keys
    return hdr_lists


def print_header(hdr, show_comments=True, ignore_keys=True, parse_lists=True):
    """
    Print FITS file header

    Parameters
    ----------
    hdr
        Fits file path or header object

    show_comments : bool
        Toggle printing comments.

    ignore_keys : bool or list of str
        List of key prefixes in header to ignore, True uses default list.

    parse_lists : bool
        Toggle parsing of lists created by add_header_item() by
        get_header_lists().
    """
    if not isinstance(fits, astropy.io.fits.header.Header):
        _, hdr = read_fits(hdr)
    keys = [k for k in hdr.keys() if len(k) > 0]
    if parse_lists:
        hdr_lists, list_ignore_keys = get_header_lists(hdr, return_ignore_keys=True)
        keys = [k for k in keys if k not in list_ignore_keys]
    else:
        hdr_lists = {}
    if ignore_keys:
        if ignore_keys is True:
            ignore_keys = ['ESO INS1 TEMP',
                           'ESO INS1 SENS',
                           'ESO INS1 PAC XTEMP',
                           'ESO INS1 PAC YTEMP',
                           'ESO INS2 TEMP',
                           'ESO INS2 SENS',
                           'ESO INS4',
                           'ESO DET CLDC1',
                           'ESO PRO REC1',
                           'ESO DRS']
        if isinstance(ignore_keys, str):
            ignore_keys = [ignore_keys]

        for idx, key in enumerate(keys):
            for ignore_key in ignore_keys:
                if key.startswith(ignore_key):
                    keys[idx] = None
                    break
        keys = [k for k in keys if k is not None]

    for k in keys:
        cprint(k, end='', fg='b', style='b')
        cprint(' = ', end='', fg='w')
        cprint(hdr[k], end='')
        if show_comments:
            if hdr.comments[k]:
                cprint(' / ' + hdr.comments[k], fg='w', style='i', end='')
        print()
        if k in hdr_lists:
            print(hdr_lists[k])

    if ignore_keys:
        print()
        print('Ignored following header keys:')
        [print('  ' + k + '*') for k in ignore_keys]


def compare_fits(path1, path2, show_common=True, show_img=True, show_comments=True,
                 ignore_keys=None):
    """
    Prints summary comparison of two FITS file headers.

    Parameters
    ----------
    path1, path2 : str
        Paths of two FITS files to compare.

    show_common : bool
        Toggle printing of values common to both files.

    show_img : bool
        Plot example images from FITS files.

    show_comments : bool
        Print comments.

    ignore_keys : list
        List of keys to ignore, set to `True` for default keys.
    """
    print('Comparing FITS files:')
    c1 = 'r'
    c2 = 'm'
    cprint('1: ' + path1, fg=c1, style='b')
    cprint('2: ' + path2, fg=c2, style='b')
    img1, hdr1 = read_compressed_fits(path1)
    img2, hdr2 = read_compressed_fits(path2)

    if show_img or show_img == 0:
        plt.clf()
        idx = 0
        for img, c in zip([img1, img2], [c1, c2]):
            idx += 1
            plt.subplot(1, 2, idx)
            while len(img.shape) > 2:
                img = img[0]
            if show_img is True:
                percentile = 1
            else:
                percentile = show_img
            tools.image.show_image(img, colorbar=True, percentile=percentile, show=False)
            plt.title(f'Example frame from file {idx}', color=c)
        plt.pause(1e-3)
    print()

    keys1, keys2 = list(hdr1.keys()), list(hdr2.keys())
    keys1 = [k for k in keys1 if len(k) > 0]
    keys2 = [k for k in keys2 if len(k) > 0]

    if ignore_keys:
        if ignore_keys is True:
            ignore_keys = ['ESO INS1 TEMP',
                           'ESO INS1 SENS',
                           'ESO INS1 PAC XTEMP',
                           'ESO INS1 PAC YTEMP',
                           'ESO INS2 TEMP',
                           'ESO INS2 SENS',
                           'ESO INS4',
                           'ESO DET CLDC1',
                           'ESO PRO REC1',
                           'ESO DRS']
        if isinstance(ignore_keys, str):
            ignore_keys = [ignore_keys]

        for header_idx in [1, 2]:
            if header_idx == 1:
                key_set = keys1
            else:
                key_set = keys2
            for idx, key in enumerate(key_set):
                for ignore_key in ignore_keys:
                    if key.startswith(ignore_key):
                        key_set[idx] = None
                        break
            key_set = [k for k in key_set if k is not None]
            if header_idx == 1:
                keys1 = key_set
            else:
                keys2 = key_set

    common_keys = [k for k in keys1 if k in keys2]
    unique_keys1 = [k for k in keys1 if k not in keys2]
    unique_keys2 = [k for k in keys2 if k not in keys1]

    cprint(f'{len(common_keys)} common keys:', fg='k', bg='w', style='b')
    max_len = max([len(k) for k in common_keys])
    common_val = 0
    for k in common_keys:
        # Print info for keys in both files
        v1 = hdr1[k]
        v2 = hdr2[k]
        cmt1 = hdr1.comments[k]
        if v1 == v2:
            # Values equal
            common_val += 1
            if not show_common:
                continue
            cprint(k + ' '*(max_len-len(k)), fg='g', end=' ')
            cprint(v1, bg='g', end=' ')
            if show_comments:
                cprint(cmt1, fg='w', style='i', end=' ')
        else:
            # Values different
            print(k + ' '*(max_len-len(k)), end=' ')
            cprint(v1, bg=c1, end=' ')
            if show_comments:
                cprint(cmt1, fg='w', style='i', end=' ')
            print()
            cprint(' '*(1+max_len), end='', style='u', fg='w')
            cprint(v2, bg=c2, end='')
            if show_comments:
                cl = len(cmt1) + 1 - len(str(v2)) + len(str(v1))
                cprint(' '*cl, fg='w', style='u', end=' ')
        print()
    cprint(f'^ {common_val}/{len(common_keys)} common values', fg='k', bg='w')

    print()
    cprint(f'{len(unique_keys1)} keys unique to 1:', fg='k', bg='w', style='b')
    for k in unique_keys1:
        max_len = max([len(k) for k in unique_keys1])
        print(k + ' '*(max_len-len(k)), end=' ')
        cprint(hdr1[k], bg=c1, end=' ')
        if show_comments:
            cprint(hdr1.comments[k], fg='w', style='i', end=' ')
        print()

    cprint(f'{len(unique_keys2)} keys unique to 2:', fg='k', bg='w', style='b')
    for k in unique_keys2:
        max_len = max([len(k) for k in unique_keys2])
        print(k + ' '*(max_len-len(k)), end=' ')
        cprint(hdr2[k], bg=c2, end=' ')
        if show_comments:
            cprint(hdr2.comments[k], fg='w', style='i', end=' ')
        print()

    if ignore_keys:
        print()
        print('Ignored following header keys:')
        [print('  ' + k + '*') for k in ignore_keys]


def find_reduction_number(header, output='string', break_num=999):
    """
    Finds next reduction number for header documentation.

    Scans through FITS header object and identifies if any reduction metadata
    header keys exist. Used to provide sequential reduction keys ('RE1', 'RE2'
    etc.).

    Parameters
    ----------
    header
        Header object to scan.

    output : {'string', 'number', 'both'}
        Switch between outputing header string ('string') or reduction number
        ('number') or tuple of both (number, string).

    break_num : int
        Maximum reduction number to scan for (prevents infinite loop in edge
        cases).

    Returns
    -------
    str or int or tuple
        Either string of format 'HIERARCH RE1' or reduction number depending on
        output toggle or both.
    """
    num = 1
    while any('RE{} '.format(num) in e for e in header.keys()):
        num += 1
        if num and num >= break_num:
            print('WARNING: reduction number = {}'.format(num))
            print('  Stopping trying to find larger reduction number')

    msg = 'HIERARCH RE{} '.format(num)
    if output == 'number':
        return num
    elif output == 'both':
        return num, msg
    else:
        return msg


def get_reduction_info(header, re_idx=None):
    """
    Gets reduction information from FITS header.

    Reduction information has keys of form `HIERARCH RE1` (i.e. have been added
    by this module). If specific reduction index is specified, dict of that
    reduction info is returned. Otherwise list of dicts of every reduction is
    returned.

    Note `re_idx` uses Python zero based indexing, so is one less than the one
    based indexing used in `HIERARCH RE1`, `HIERARCH RE2` etc.

    Parameters
    ----------
    header
        FITS header object to return information about.

    re_idx : int or None
        Reduction index to return information about. Set to `None` to return
        list of information about all reductions

    Returns
    -------
    list or dict
        List or dict of information about reductions depending on value of
        `re_idx`.
    """
    keys = header.keys()
    regex = r'^RE\d\d*$'  # Only get relevant keys
    keys = [k for k in keys if re.match(regex, k.split()[0])]
    reductions_hdr = {k: header[k] for k in keys}
    reduction_idxs = [int(k.split()[0][2:])-1 for k in reductions_hdr]
    reduction_list = [None]*(max(reduction_idxs) + 1)
    for key, reduction_idx in zip(reductions_hdr, reduction_idxs):
        if reduction_list[reduction_idx] is None:
            reduction_list[reduction_idx] = {}
        new_key = ' '.join(key.split()[1:])
        reduction_list[reduction_idx][new_key] = reductions_hdr[key]
    if re_idx is None:
        return reduction_list
    else:
        return reduction_list[re_idx]
