import datetime
import json
import os
import pickle
import re
import typing

import joblib
import numpy as np
from tqdm.auto import tqdm

class ProgressParallel(joblib.Parallel):
  def __init__(self, use_tqdm=True, total=None, desc=None, unit='it', *args, 
               **kwargs):
    self._use_tqdm = use_tqdm
    self._total = total
    self._desc = desc
    self._unit = unit
    super().__init__(*args, **kwargs)

  def __call__(self, *args, **kwargs):
    with tqdm(disable=not self._use_tqdm, total=self._total, 
              desc=self._desc, unit=self._unit, leave=False) as self._pbar:
      return joblib.Parallel.__call__(self, *args, **kwargs)

  def print_progress(self):
    if self._total is None: 
      self._pbar.total = self.n_dispatched_tasks
    self._pbar.n = self.n_completed_tasks
    self._pbar.refresh()
    
def print_status(message: str) -> None:
  """Print a message along with the current time. Usefull for logging."""
  tqdm.write(f'{datetime.datetime.now()} {message}')
  
def load(filepath: str, verbose: bool = False):
  """Try to pickle load the file."""
  if verbose: print_status(f'Read in {filepath}')
  if not os.path.isfile(filepath):
    print_status(f'{filepath} does not exists')
    return None
  
  # Check extension
  extension = filepath.rsplit('.', maxsplit=1)[1]
  assert extension in ('pkl', 'npy', 'json', 'int', 'float', 'txt'), (
    f"{extension} not implemented"
  )
  if extension == 'npy':
    return np.load(filepath)
  elif extension == 'pkl':
    with open(filepath, 'rb') as file:
      return pickle.load(file)
  else:
    with open(filepath, 'r') as file:
      if extension == 'json': return json.load(file)
      elif extension == 'int': return int(file.read())
      elif extension == 'float': return float(file.read())
      elif extension == 'txt': return file.read()
  
def file_exists(files: typing.Union[str, list[str]], *, verbose: bool = False
                ) -> bool:
  """Check if file (or files) exists. If any exists, report and return True."""
  if isinstance(files, str):
    files = [files]
    
  for file in files:
    if os.path.isfile(file):
      if verbose: print_status(f"{file} already exists")
      return True
      
  return False           

def recursive_file_check(
  filenames: list[str],
  path: str = 'data/'
) -> None:
  """Report when in any subdirectory of path, one or more of the filenames are 
  missing."""
  entries = sorted(os.scandir(path), key= lambda x: x.name)
  for entry in entries:
    for filename in filenames:
      if not os.path.isfile(os.path.join(entry.path, filename)):
        print(f'{filename} does not exist in {entry.path}.')
        break

def recursive_file_loading(filename: str, path: str = 'data/', 
                           verbose: bool = False
                           ) -> dict[int, typing.Any]:
  """Search for files with filename in every directory in path. If they exist, 
  load them and return them in a dict, keyed on the index."""
  result = dict()
  entries = sorted(os.scandir('data'), key=lambda x: x.name)
  for entry in entries:
    filepath = os.path.join(entry.path, filename)
    result[int(entry.name)] = load(filepath, verbose=verbose)
  return result

def get_categories():
  result = {
    5: 'Coauthorship',
    7: 'Coauthorship',
    28: 'Hyperlink',
    29: 'Hyperlink',
    30: 'Communication',
  }
  entries = sorted(os.scandir('data'), key=lambda x: x.name)
  for entry in entries:
    meta_file = os.path.join(entry.path, 'meta')
    if os.path.isfile(meta_file):
      with open(meta_file) as file:
        for line in file:
          if line.startswith('category'):
            result[int(entry.name)] = line.split(':')[1].strip()
  return result