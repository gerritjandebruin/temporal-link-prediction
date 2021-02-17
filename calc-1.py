import sys
import tlp

idx = int(sys.argv[1])
print(idx)
tlp.analysis.cheap_statistics(edgelist_file=f'data/{idx}/edgelist.pkl', output_path=f'data/{idx}', verbose=True)