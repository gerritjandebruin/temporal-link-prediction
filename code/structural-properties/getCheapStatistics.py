import os
from tlp import features

import joblib
from tqdm.auto import tqdm

import tlp
import tlp.stats

if __name__ == "__main__":
  indices = sorted(os.listdir('data'))
  
  # Check
  # for index in indices:
  #   if not os.path.isfile(os.path.join('data', index, 'stats.json')):
  #     print(index)
  
  # Single core
  indices = [15, 17, 26, 27]
  for index in tqdm(indices, mininterval=0, unit='graph'):    
    tlp.stats.cheap_statistics(index, verbose=True)
    
  # Multicore
  # tlp.ProgressParallel(n_jobs=len(indices), total=len(indices))(
  #   joblib.delayed(tlp.stats.cheap_statistics)(index, verbose=True) 
  #   for index in indices
  # )