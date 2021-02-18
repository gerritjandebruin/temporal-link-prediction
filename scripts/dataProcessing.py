import collections
import os

import joblib
import pandas as pd
from tqdm.auto import tqdm

import tlp

make_undirected = {12: False}
make_undirected = collections.defaultdict(lambda: True, make_undirected)

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
  # Single core
  # for index in tqdm(os.listdir('data')):
  #   path = os.path.join('data', index)
  #   index = int(index)
  #   tlp.data_preparation(path, **adjusted_intervals[index], verbose=True)
    
  # Multicore
  indices = [int(path) for path in os.listdir('data')]
  tlp.ProgressParallel(n_jobs=len(indices), total=len(indices))(
    joblib.delayed(tlp.data_preparation)
    (path=os.path.join('data', str(index)), **adjusted_intervals[index])
    for index in indices
  )
    
    