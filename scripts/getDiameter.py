import argparse
import concurrent.futures
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
      for type_stat in ['diameter.int']:
        filepath = os.path.join(entry.path, type_stat)
        if not os.path.isfile(filepath):
          tlp.print_status(f'{filepath} does not exist.')
          ok = False
          break
    if ok: print('Everything is OK.')
    return

  # Entries
  entries = sorted(os.scandir('data'), key=lambda x: x.name)

  # Singlecore
  for entry in entries:
    tlp.print_status(f'#{entry.name}')
    tlp.get_diameter(entry.path)

if __name__ == '__main__':
  main()