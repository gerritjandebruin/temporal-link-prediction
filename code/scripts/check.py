import os
import os.path

def check_file(path: str) -> bool:
  if not os.path.isfile(path):
    print(f'{path} does not exist.')
    return False
  else:
    return True

def check_data(path: str):
  """Report whether all required files for analysis are present.

  If something is not present, it will be reported.  
  """
  files = [
    'edgelist.pkl',
    'edgelist_mature.pkl',
    'edgelist_probe.pkl',
    'instances.npy',
    'targets.npy',
    'instances_sampled.npy',
    'targets_sampled.npy',
    'stats.json',
    'diameter.int',
    'path_distribution.npy',
    'features/aa_time_agnostic.pkl',
    'features/aa_time_aware.pkl',
    'features/na.pkl',
    'features/sp.pkl',
    'scores/auc_all_features.float',
    'scores/auc_time_agnostic.float'
  ]

  for file in files:
    if not check_file(os.path.join(path, file)):
      return

def check_all_data(path: str = 'data/'):
  """Report whether all required files are present for all networks.
  
  If something is not present, it will be reported.
  """
  for dataset in sorted(os.listdir(path)):
    check_data(os.path.join(path, dataset))

if __name__ == '__main__':
  check_all_data()
