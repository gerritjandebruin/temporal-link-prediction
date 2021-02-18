import os
import typing

import numpy as np

from .core import file_exists

def _sample(array: np.ndarray, size: int) -> np.ndarray:
  """Take a sample (with replacement) of a given size n along the first axis of 
  array.
  """
  return array[np.random.randint(len(array), size=size)]

def balanced_sample(
  path: str, *, sample_size: typing.Optional[int], verbose: bool = False
  ) -> None:
  """Take n positive and n negative samples from the provided instances. Whether
  a instance is positive, is determined by the targets.
  The instances should be a np.ndarray of shape (n,2) provided at 
  path/instances.npy and the targets a np.ndarray of shape (n) provided at 
  path/targets.npy. The result are two files; path/instances_sampled.npy is a 
  np.ndarray with shape (size, 2) and path/targets_sampled.npy is a np.ndarray
  with shape (size).
  
  Args:
    path: str
    sample_size: Optional; Take this number of positive and this number of 
      negative samples. If None, do not sample and return all instances.
    verbose: Optional; Defaults to False.
  """
  instances_sampled_file = os.path.join(path, 'instances_sampled.npy')
  targets_sampled_file = os.path.join(path, 'instances_sampled.npy')
  if file_exists([instances_sampled_file, targets_sampled_file], 
                 verbose=verbose): 
    return
  
  instances_file = os.path.join(path, 'instances.npy')
  assert os.path.isfile(instances_file), f'{instances_file} does not exist'
  instances = np.load(instances_file)   
  
  targets_file = os.path.join(path, 'targets.npy')
  assert os.path.isfile(targets_file), f'{targets_file} does not exist'
  targets = np.load(targets_file)    
  
  if sample_size is None:
    np.save(instances_sampled_file, instances)
    np.save(targets_sampled_file, targets) 
    
  else:  
    positives = _sample(instances[targets], sample_size)
    negatives = _sample(instances[~targets], sample_size)
    
    instances_sampled = np.concatenate([negatives, positives])
    np.save(instances_sampled_file, instances_sampled)
    
    targets_sampled = np.concatenate(
      [np.zeros(sample_size, dtype=bool), np.ones(sample_size, dtype=bool)])
    np.save(targets_sampled_file, targets_sampled)