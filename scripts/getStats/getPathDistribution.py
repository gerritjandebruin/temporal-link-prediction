import os
from tqdm.auto import tqdm

import tlp.stats

if __name__ == "__main__":
  paths = sorted([entry.path for entry in os.scandir('data')])
    
  # Check
  # for path in paths:
  #   if not os.path.isfile(os.path.join(path, 'path_distribution.npy')):
  #     print(path)
    
  for path in tqdm(paths, unit='graphs'):
    tqdm.write(path)
    tlp.stats.path_distribution(path)