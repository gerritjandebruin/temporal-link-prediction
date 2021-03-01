import argparse, datetime, os

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
      for type_stat in ['path_distribution.npy']:
        filepath = os.path.join(entry.path, type_stat)
        if not os.path.isfile(filepath):
          tlp.print_status(f'{filepath} does not exist.')
          ok = False
          break
    if ok: print('Everything is OK.')
    return

  # Single index
  if args.index is not None:
    tlp.get_path_distribution(os.path.join('data', f'{args.index:02}'), sample_size=1000000)
    return

  # Entries
  entries = sorted(os.scandir('data'), key=lambda x: x.name)

  # Singlecore
  time = datetime.datetime.now()
  for entry in entries:
    tlp.print_status(f'#{entry.name}')
    tlp.get_path_distribution(entry.path, sample_size=1000000)
  print(f'Done in {datetime.datetime.now() - time}.')

if __name__ == '__main__':
  main()