import os

import joblib
from tqdm.auto import tqdm

import tlp
import tlp.features

if __name__ == "__main__":
  paths = sorted([os.path.join('data', path) for path in os.listdir('data')])
  
  # Check
  # for path in paths:
  #   if not os.path.isfile(os.path.join(path, 'features', 'sp.pkl')):
  #     print(path)
  
  # Single index
  tlp.features.construction('data/24', verbose=True)
  
  # Single core
  # for path in tqdm(paths, mininterval=0, unit='graph'):
  #   tlp.features.construction(path, verbose=True)
    
  # Multicore
  # tlp.ProgressParallel(n_jobs=len(paths), total=len(paths))(
  #   joblib.delayed(tlp.features.construction)(path) for path in paths
  # )