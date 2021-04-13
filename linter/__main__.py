import argparse
import glob
import os
import sys

import autopep8
from yapf.yapflib import yapf_api
from yapf_contrib.fixers import fixers_api

OPTS = {'yapf': {'yapf', 'quotes'}, 'medium': {'autopep8', 'yapf', 'quotes'}}

parser = argparse.ArgumentParser(
    description="""Lint the repo and correct mistakes that are not caught by YAPF.

Options list:
- yapf:
  - general Google look
- quotes:
  - bad-docstring-quotes (C0198)
  - inconsistent-quotes (W1405)
""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    prefix_chars='-+^')
parser.add_argument(
    '^x',
    type=str,
    dest='opts',
    metavar='OPTIONS',
    default='yapf',
    help='option sets to default to')
parser.add_argument(
    '-x',
    type=str,
    dest='noopts',
    metavar='OPTIONS',
    help='disable linting options')
parser.add_argument(
    '+x',
    type=str,
    dest='enopts',
    metavar='OPTIONS',
    help='enable linting options')
parser.add_argument(
    '--autopep8-agression',
    type=int,
    help='aggressiveness of autopep8',
    default=2)
parser.add_argument(
    'files',
    nargs='+',
    help='if a directory is specified, all Python files inside are used')
args = parser.parse_args()

files = args.files
for i, file in enumerate(files):
  if os.path.isdir(file):
    files[i] = None
    files.extend(glob.glob(os.path.join(file, '**', '*.py'), recursive=True))
  elif not os.path.exists(file):
    raise FileNotFoundError(file)

opts = set()
for optset in args.opts.split(','):
  opts.update(OPTS[optset])
if args.enopts is not None:
  opts.update(args.enopts.split(','))
if args.noopts is not None:
  opts.difference_update(args.noopts.split(','))

# The real linter starts here
if len(opts) != 0:
  for file in files:
    print(file)
    if file is None:
      continue
    changed = False
    with open(file) as src_file:
      src = src_file.read()

    if 'autopep8' in opts:
      old_src = src
      src = autopep8.fix_code(
          src,
          options={
              'aggressive': args.autopep8_agression,
              'ignore': ['E1', 'W1', 'E501']
          })
      changed |= (old_src != src)
      del old_src

    # if 'quotes' in opts:
    #   old_src = src
    #   try:
    #     src = fixers_api.Pre2to3FixerRun(src, {'fixers': ['quotes']})
    #   except:
    #     import traceback
    #     print("Error trying to fix quotes: " + traceback.format_exc())
    #   else:
    #     changed |= (old_src != src)
    #     del old_src

    if 'yapf' in opts:
      src, _changed = yapf_api.FormatCode(src, file, 'yapf')
      changed |= _changed

    if changed:
      with open(file, 'w') as src_file:
        src_file.write(src)
