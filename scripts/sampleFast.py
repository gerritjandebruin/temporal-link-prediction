import datetime, os, random, sys

import joblib
import networkx as nx
import numpy as np
from tqdm.auto import tqdm

cutoff=2
sample_size=10000

def sample_faster(graph_mature, edgeset_probe, nodes, cutoff, sample_size, position):
  # negatives = list()
  positives = list()

  with tqdm(total=sample_size, desc=f'#{position}', position=position) as pbar:
    for node in nodes:
      for neighborhood, distance in nx.single_source_shortest_path_length(graph_mature, node, cutoff=2).items():
        if distance > 1 and node < neighborhood:
          instance = (node, neighborhood)
          if instance in edgeset_probe:
            pbar.update()
            positives.append(instance)
            # break # Don't get a second positive from the same node.
          # else:
            # negatives.append(instance) # Negatives will be subsampled later. Maybe this should be done faster, to prevent excessive RAM usage.
    # else:
    #   print(f'Did not find {sample_size} positives!')
  # return positives, negatives
  return positives
    

def sample_fast(path, position, sample_size, chunks_number):

  assert sample_size % chunks_number == 0, 'Please take care that sample_size is divisible by chunks_number'

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
  # negatives = list()
  # positives = list()

  nodes_chunks = np.array_split(nodes, chunks_number)

  results = joblib.Parallel(n_jobs=chunks_number)(
    joblib.delayed(sample_faster)(graph_mature, edgeset_probe, chunck, cutoff, sample_size/chunks_number, position)
    for position, chunck in enumerate(nodes_chunks)
  )

  # for positives_in_chunk, negatives_in_chunk in results:
  #   positives.extend(positives_in_chunk)
  #   negatives.extend(negatives_in_chunk)

  # with tqdm(total=sample_size, desc=path, position=position) as pbar:
  #   for node in nodes:
  #     if len(positives) > sample_size:
  #       break
  #     for neighborhood, distance in nx.single_source_shortest_path_length(graph_mature, node, cutoff=2).items():
  #       if distance > 1 and node < neighborhood:
  #         instance = (node, neighborhood)
  #         if instance in edgeset_probe:
  #           pbar.update()
  #           positives.append(instance)
  #           break # Don't get a second positive from the same node.
  #         else:
  #           negatives.append(instance) # Negatives will be subsampled later.
  #   else:
  #     raise Exception('Did not find 10000 positives!')

  # Sample to 10,000 negatives
  # negatives = random.sample(negatives, sample_size)

  # Combine instances
  # instances_sampled = np.concatenate([negatives, positives])
  # np.save(os.path.join(path, 'instances_sampled_fast.npy'), instances_sampled)
  np.save(results, os.path.join(path, 'positives.npy'))

  # Store targets
  targets_sampled = np.concatenate(
    [np.zeros(sample_size, dtype=bool), np.ones(sample_size, dtype=bool)])
  np.save(os.path.join(path, 'targets_sampled_fast.npy'), targets_sampled)

def main():
  # joblib.Parallel(n_jobs=4)(
  #   joblib.delayed(sample_fast)(path=f'data/{index:02}', position=position, sample_size=sample_size)
  #   for position, index in enumerate([15, 17, 26, 27])
  # )
  index = int(sys.argv[1])
  chunks_number = int(sys.argv[2])
  sample_fast(path=f'data/{index:02}', position=None, sample_size=10000, chunks_number=chunks_number)

if __name__ == '__main__':
  main()
