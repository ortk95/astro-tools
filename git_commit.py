#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module to simplify process of creating git commits and pushing to GitHub.

Example usage in Python:
>>> from git_commit import git_commit
>>> git_commit('First commit')

Example usage from command line:
$ ./git_commit.py "First commit"

If the commit message is not given, the code will prompt for it.

The remote name for the git directory should be specified below.
"""

__version__ = '1.0'
__date__ = '2019-04-08'
__author__ = 'Oliver King'

import os
import sys
import subprocess
from subprocess import CalledProcessError


# Change REMOTE_NAME to the name of your remote branch in git (often 'origin').
# This needs to be set up using git command line with `git remote add ...` e.g.
# $ git remote add github https://github.com/ortk95/astro-tools.git
REMOTE_NAME = 'github'


def git_commit(commit_msg=None, remote_name=None):
    """
    Adds all relevant new files to git, creates commit and pushes to GitHub.

    Use optional argument commit_msg to specify message used in commit.
    Otherwise code will prompt user for  a commit message. Remote can be
    defined using remote_name, otherwise takes default from REMOTE_NAME.

    Script will attempt to identify git root directory by progressively moving
    up the directory tree and checking if `.git` exists in the current
    directory.

    git_commit effectively runs following commands:
        $ git add . -A
        $ git commit -m commit_msg
        $ git push remote_name --all
        $ git_path remote_name --tags

    Arguments
    ---------
    commit_msg : str
        Message to annotate commit. If unspecified, code will prompt user for a
        commit message.

    remote_name : str
        Name of remote. If unspecified, code will use default module value
        defined in REMOTE_NAME.
    """
    # Find root directory of git repository by moving up tree and looking for directory containing
    # an item called `.git`. This assumes the current working directory is within the git
    # repository there aren't any `.git` directories anywhere other than the root directory.
    start_dir = os.getcwd()
    git_dir = './'
    for _ in range(len(start_dir.split(os.sep))):
        if '.git' in os.listdir(git_dir):
            break
        else:
            git_dir += '../'

    if start_dir != git_dir:
        os.chdir(git_dir)

    # Print info about last commit and changes since it
    shell('git add . -A')
    try:
        shell('git log -1 # PREVIOUS COMMIT:')
    except CalledProcessError:
        # Prevent code crashing for case where there are no previous commits
        pass
    print()
    shell('git status')

    # Create actual commit
    if commit_msg is None:
        try:
            commit_msg = input('Commit message: ')
        except KeyboardInterrupt:
            print('\n\nKeyboardInterrupt, cancelling commit')
            if start_dir != git_dir:
                os.chdir(start_dir)
            return
    if len(commit_msg) == 0:
        print('\nNo message, cancelling commit')
        if start_dir != git_dir:
            os.chdir(start_dir)
        return
    shell('git add . -A') # Double check to see if any new files in git_dir
    shell('git commit -m "%s"' % commit_msg)

    # Push changes to github
    try:
        if remote_name is None:
            remote_name = REMOTE_NAME
        shell('git push %s --all' % remote_name)
        shell('git push %s --tags' % remote_name)
    except CalledProcessError:
        print('WARNING: Failed pushing to GitHub, likely no internet connection')

    if start_dir != git_dir:
        os.chdir(start_dir)


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
        print('> ' + command_str)

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


def test_if_ipython():
    """Detect if script is running in IPython console"""
    try:
        return __IPYTHON__
    except NameError:
        return False


if __name__ == '__main__':
    git_commit(*sys.argv[1:])
