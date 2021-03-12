import os
from tlp import features

import joblib
from tqdm.auto import tqdm

import tlp
import tlp.stats

if __name__ == "__main__":
  paths = sorted([entry.path for entry in os.scandir('data')])
  
  # Check
  # for path in paths:
  #   if not os.path.isfile(os.path.join(path, 'diameter.txt')):
  #     print(path)
    
  # Multicore
  paths = ['data/11', 'data/27']
  tlp.ProgressParallel(n_jobs=len(paths), total=len(paths))(
    joblib.delayed(tlp.stats.diameter)(path) for path in paths
  )