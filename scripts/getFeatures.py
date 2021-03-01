import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import pickle

import joblib
import numpy as np
from tqdm import tqdm

import tlp

implemented_features = ['aa_time_agnostic', 'aa_time_aware', 'na', 'sp']

def main():
  # Handle arguments
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--index', type=int, help="If provided, only process this index.")
  parser.add_argument(
    '--check', help='Check if result already exists.', 
    action='store_true')
  parser.add_argument(
    '--multicore', help='Run datasets in parallel.', 
    action='store_true')
  args = parser.parse_args()

  assert not (args.index and args.check), (
    "--check and --index not implemented at the same time")

  # Check
  if args.check:
    for entry in sorted(os.scandir('data/'), key=lambda x: x.name):
      for implemented_feature in implemented_features:
        filepath = os.path.join(
          entry.path, 'features', f'{implemented_feature}.pkl')
        if not os.path.isfile(filepath):
          tlp.print_status(
            f'#{entry.name} {implemented_feature} (and possibly others) does not exist')
          break
        else:
          with open(filepath, 'rb') as file:
            features = pickle.load(file)
          assert isinstance(features, dict), f'{filepath} is not a dict'
          for idx, feature in features.items():
            assert isinstance(feature, np.ndarray), (
              f'{filepath} {idx} is not np.ndarray but {type(feature)}')
            assert len(feature) == 20000, f'{filepath}: {idx} {len(feature)}'

    return
  
  # Single run
  if args.index is not None:
    path = os.path.join('data', f'{args.index:02}')
    tlp.get_features(path=path, verbose=True)
    return

  # Entries
  entries = sorted(os.scandir('data'), key=lambda x: x.name)

  # All datasets, multicore
  if args.multicore:
    tlp.ProgressParallel(n_jobs=len(entries))(
      joblib.delayed(tlp.get_features)(entry.path) for entry in entries
    )
    return

  # All datasets, singlecore
  with tqdm(entries, mininterval=0, unit='graph', miniters=0) as entry_it:
    for entry in entry_it:
      entry_it.set_postfix(dict(path=entry.path))
      tlp.get_features(entry.path, verbose=True)

if __name__ == '__main__':
  main()