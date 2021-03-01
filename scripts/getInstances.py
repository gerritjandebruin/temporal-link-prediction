import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import os
import pickle

import joblib
import numpy as np
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

def main():
  # Handle arguments
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--index', type=int, help="If provided, only process this index.")
  parser.add_argument(
    '--check', help='Check if edgelist.pkl already exists.', 
    action='store_true')
  parser.add_argument('--multicore', action='store_true')
  args = parser.parse_args()
  
  assert not (args.index and args.check), (
    "--check and --index not implemented at the same time")

  entries = sorted(os.scandir('data'), key=lambda x: x.name)

  # Check
  if args.check:
    for entry in entries:
      edgelist_mature_file = os.path.join(entry.path, 'edgelist_mature.pkl')
      edgelist_probe_file = os.path.join(entry.path, 'edgelist_probe.pkl')
      targets_sampled_file = os.path.join(entry.path, 'targets_sampled.npy')
      instances_sampled_file = os.path.join(entry.path, 'instances_sampled.npy')
      if not os.path.isfile(edgelist_mature_file):
        tlp.print_status(f'Edgelist_mature.pkl not in {entry.name}')
        continue
      if not os.path.isfile(edgelist_probe_file):
        tlp.print_status(f'Edgelist_probe.pkl not in {entry.name}')
        continue        
      if not os.path.isfile(instances_sampled_file):
        tlp.print_status(f'instances_sampled.npy not in {entry.name}')
        continue
      if not os.path.isfile(targets_sampled_file):
        tlp.print_status(f'targets_sampled.npy not in {entry.name}')
        continue
      instances_sampled = tlp.load(instances_sampled_file)
      assert isinstance(instances_sampled, np.ndarray)
      assert instances_sampled.shape == (20000,2)
      targets_sampled = tlp.load(targets_sampled_file)
      assert isinstance(targets_sampled, np.ndarray)
      assert targets_sampled.shape == (20000,)
      assert np.array_equal(np.bincount(targets_sampled), (10000, 10000)), (
        f'Expected (10000, 10000) (neg, pos), got: {np.bincount(targets_sampled)}'
      )
      # Check if all nodepairs in instances actually exist.
      with open(edgelist_mature_file, 'rb') as file:
        edgelist_mature = pickle.load(file)
      edgeset_mature = set(edgelist_mature[['source', 'target']].values.flat)
      for idx, (u, v) in enumerate(instances_sampled): #type: ignore
        assert (u in edgeset_mature) and (v in edgeset_mature), (
          f'#{entry.name} Error for idx {idx}: {targets_sampled[idx]}'
        )
        
    return

  # Run only single index
  if args.index is not None:
    path = os.path.join('data', f'{args.index:02}')
    tlp.from_edgelist_to_samples(path, **adjusted_intervals[args.index], verbose=True)
    return

  # Multicore
  entries = sorted(os.scandir('data'), key=lambda x: x.name)
  if args.multicore:
    joblib.Parallel(n_jobs=len(entries), backend='multiprocessing')(
      joblib.delayed(tlp.from_edgelist_to_samples)(
        entry.path, **adjusted_intervals[int(entry.name)], verbose=False) 
        for entry in entries
    )
    return

  # Single core
  for entry in tqdm(entries, mininterval=0, unit='graph'):
    tlp.from_edgelist_to_samples(
      entry.path, **adjusted_intervals[int(entry.name)], verbose=True)

if __name__ == "__main__":
  main()     