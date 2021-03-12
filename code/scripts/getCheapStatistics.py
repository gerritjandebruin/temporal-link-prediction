import argparse
import os

import joblib
import tlp

def main():
  # Handle arguments
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--index', type=int, help="If provided, only process this index.")
  parser.add_argument(
    '--check', help='Check if result already exists.', 
    action='store_true')
  parser.add_argument(
    '--multicore', help='Run cheap statistics in parallel.', 
    action='store_true')
  args = parser.parse_args()

  assert not (args.index and args.check), (
    "--check and --index not implemented at the same time") 

  # Check
  if args.check:
    ok = True
    for entry in sorted(os.scandir('data/'), key=lambda x: x.name):
      for type_stat in ['stats.json']:
        filepath = os.path.join(entry.path, type_stat)
        if not os.path.isfile(filepath):
          tlp.print_status(f'{filepath} does not exist.')
          ok = False
          break
    if ok: print('Everything is OK.')
    return

  # Entries
  entries = sorted(os.scandir('data'), key=lambda x: x.name)

  # Index
  if args.index is not None:
    tlp.get_cheap_statistics(os.path.join('data', f'{args.index:02}'), verbose=True)
    return

  # Multicore
  if args.multicore:
    tlp.ProgressParallel(total=len(entries), n_jobs=len(entries))(
      joblib.delayed(tlp.get_cheap_statistics)(entry.path) for entry in entries
    )
    return

  # Singlecore
  for entry in entries:
    tlp.get_cheap_statistics(entry.path, verbose=True)

if __name__ == '__main__':
  main()