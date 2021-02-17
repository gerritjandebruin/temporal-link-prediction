import os
import re
import requests

import joblib
import numpy as np
from tqdm.auto import tqdm

from .features import Experiment

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
        
def recursive_file_lookup(filename):
  result = dict()
  for dirpath, dirnames, files in os.walk('./data'):
    if filename in files:
      result[dirpath.split('/')[2]] = joblib.load(os.path.join(dirpath, filename))
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
  
def recursive_feature_lookup(path: str) -> dict[Experiment, np.ndarray]:
  result = dict()
  for file in os.listdir(path):
    filepath = os.path.join(path, file)
    if os.path.isfile(filepath):
      result.update(joblib.load(filepath))
  return result
                