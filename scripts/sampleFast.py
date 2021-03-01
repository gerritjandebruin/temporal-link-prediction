import datetime, os, random, sys

import joblib
import networkx as nx
import numpy as np
from tqdm.auto import tqdm

cutoff=2
sample_size=10000

def sample_fast(path, position, sample_size):
  tqdm.write(path)
  # Collect instances
  edgelist_mature = joblib.load(os.path.join(path, 'edgelist_mature.pkl'))
  graph_mature=nx.from_pandas_edgelist(edgelist_mature)
  tqdm.write(f'{datetime.datetime.now()} {path} got graph_mature')
  
  # Check targets
  edgelist_probe = joblib.load(os.path.join(path, 'edgelist_probe.pkl'))
  edgeset_probe = {
    (u, v) 
    for u, v in edgelist_probe[['source', 'target']].itertuples(index=False)
  }
  tqdm.write(f'{datetime.datetime.now()} {path} got edgeset_probe')

  # Shuffle nodes
  nodes = list(graph_mature.nodes)
  random.shuffle(nodes)

  # Collect the positives and negatives
  negatives = list()
  positives = list()

  with tqdm(total=sample_size, desc=path, position=position) as pbar:
    for node in nodes:
      if len(positives) > sample_size:
        break
      for neighborhood, distance in nx.single_source_shortest_path_length(graph_mature, node, cutoff=2).items():
        if distance > 1 and node < neighborhood:
          instance = (node, neighborhood)
          if instance in edgeset_probe:
            pbar.update()
            positives.append(instance)
            break # Don't get a second positive from the same node.
          else:
            negatives.append(instance) # Negatives will be subsampled later.
    else:
      raise Exception('Did not find 10000 positives!')

  # Sample to 10,000 negatives
  negatives = random.sample(negatives, sample_size)

  # Combine instances
  instances_sampled = np.concatenate([negatives, positives])
  np.save(os.path.join(path, 'instances_sampled.npy'), instances_sampled)

  # Store targets
  targets_sampled = np.concatenate(
    [np.zeros(sample_size, dtype=bool), np.ones(sample_size, dtype=bool)])
  np.save(os.path.join(path, 'targets_sampled.npy'), targets_sampled)

def main():
  # joblib.Parallel(n_jobs=4)(
  #   joblib.delayed(sample_fast)(path=f'data/{index:02}', position=position, sample_size=sample_size)
  #   for position, index in enumerate([15, 17, 26, 27])
  # )
  index = int(sys.argv[1])
  sample_fast(path=f'data/{index:02}', position=None, sample_size=10000)

if __name__ == '__main__':
  main()
