import datetime
import os
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
              desc=self._desc, unit=self._unit) as self._pbar:
      return joblib.Parallel.__call__(self, *args, **kwargs)

  def print_progress(self):
    if self._total is None: 
      self._pbar.total = self.n_dispatched_tasks
    self._pbar.n = self.n_completed_tasks
    self._pbar.refresh()
    
def print_status(message: str) -> None:
  """Print a message along with the current time. Usefull for logging."""
  tqdm.write(f'{datetime.datetime.now()} {message}')
  
def load(file: str, verbose: bool = False):
  """Try to pickle load the file."""
  if verbose: print_status(f'Read in {file}')
  assert os.path.isfile(file)
  if file.endswith('.npy'):
    return np.load(file)
  else:
    return joblib.load(file)
  
def file_exists(files: typing.Union[str, list[str]], *, verbose: bool = False
                ) -> bool:
  """Check if file (or files) exists. If any exists, return True."""
  if isinstance(files, str):
    files = [files]
    
  for file in files:
    if os.path.isfile(file):
      if verbose: print_status(f"{file} already exists")
      return True
      
  return False
        
def recursive_file_lookup(filename):
  result = dict()
  for dirpath, dirnames, files in os.walk('./data'):
    if filename in files:
      result[dirpath.split('/')[2]] = (
        joblib.load(os.path.join(dirpath, filename)))
  return dict(sorted(result.items()))
  
def recursive_delete(filename) -> None:
  """Delete all files in current working directory that are named as the
  filename argument.
  """
  for dirpath, dirnames, files in os.walk('.'):
    if filename in files:
      os.remove(os.path.join(dirpath, filename))
      
def get_labels_from_notebook_names(filepath) -> dict[str, str]:
  """Recursive lookup of jupyter notebooks. Use the names to get the labels for 
  a given id. Example: when '01 dblp_coauthor.ipynb' is found, 
  {'01': 'dblp_coauthor} is added to the resulting dict.
  """
  labels = {
    file.split()[0]: file.split()[1].split('.')[0] 
    for file in os.listdir('.')
    if file.endswith('.ipynb') and re.match(r'[0-9]{2}', file)
  }
  return dict(sorted(labels.items()))                