import argparse
import collections
import os

import joblib
import pandas as pd
from tqdm.auto import tqdm

import tlp

# make_undirected = {12: False}
# make_undirected = collections.defaultdict(lambda: True, make_undirected)

adjusted_intervals = {
  1: {
      't_min': pd.Timestamp('1996-01-01'), 
      't_split': pd.Timestamp('2005-01-01'),
      't_max': pd.Timestamp('2007-01-01')
    },
  16: {'t_min': pd.Timestamp('2001-01-10 00:00:00')}
}
adjusted_intervals = collections.defaultdict(lambda: dict(), adjusted_intervals)

if __name__ == "__main__":
  indices = sorted([int(path) for path in os.listdir('data')])
  
  # Check
  # print('The following indices are not yet processed:')
  # for index in indices:
  #   path = os.path.join('data', f'{index:02}', 'instances_sampled.npy')
  #   if not os.path.isfile(path): print(index)

  # Run only single index
  path = os.path.join('data', '27')
  tlp.data_preparation(path, **adjusted_intervals[17], verbose=True)

  # Single core
  # for index in tqdm(indices, mininterval=0, unit='graph'):
  #   path = os.path.join('data', f'{index:02}')
  #   tlp.data_preparation(path, **adjusted_intervals[index], verbose=True)
    
  # Multicore
  # joblib.Parallel(n_jobs=len(indices))(
  #   joblib.delayed(tlp.data_preparation)
  #   (path=os.path.join('data', f'{index:02}'), **adjusted_intervals[index])
  #   for index in indices
  # )
    
    