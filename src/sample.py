import logging, random, os, itertools

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import typer

from .logger import logger
from .progress_parallel import ProgressParallel, delayed

app = typer.Typer()

def check_common_neighbor(G, u, v):
  for w in G[u]:
    if w in G[v] and w not in (u,v):
      return True
  else:
    return False

@app.command()
def single(network_index: int,
           nswap_perc: int,
           method: str = None,
           cutoff: int = 2,
           sample_size: int = 10_000,
           seed: int = 42,
           verbose: bool = True):
  if not verbose:
    logger.setLevel(logging.INFO)
  directory = f'data/{network_index:02}/{nswap_perc:+04.0f}{"" if method is None else method}'
  filepath_in = os.path.join(directory, 'edgelist.pkl')
  filepath_out = os.path.join(directory, 'samples.pkl')
  
  if os.path.isfile(filepath_out) or not os.path.isfile(filepath_in):
    return
  
  assert cutoff == 2, "Not implement for any other cutoff value than 2."
  
  edgelist = pd.read_pickle(filepath_in).loc[
    lambda x: x['source'] != x['target']] # Do not allow selfloops
  
  edgelist_mature = edgelist.loc[
    lambda x: (x['phase'] == 'mature'), ['source', 'target']]
  graph_mature = nx.from_pandas_edgelist(edgelist_mature)
  
  nodes_mature_list = list(graph_mature.nodes)
  nodes_mature_set = set(graph_mature.nodes)
  
  probe_iterator = (
    edgelist
    .loc[lambda x: x['phase'] == 'probe', ['source', 'target']]
    .itertuples(index=False)
  )
  probes_list = [(u,v) for u, v in probe_iterator]
  probes_set = set(probes_list)
  random.seed(seed)
  random.shuffle(probes_list)
  
  logger.debug(len(nodes_mature_set))
  
  positives = []
  with tqdm(total=sample_size, disable=not verbose, leave=True) as pbar:
    for u, v in tqdm(probes_list, position=1, disable=not verbose, leave=True):
      if len(positives) < sample_size:
        if (
            u in nodes_mature_set and 
            v in nodes_mature_set and 
            (u,v) not in graph_mature.edges() and 
            check_common_neighbor(graph_mature, u, v)
          ):
          positives.append((u,v))
          pbar.update(1)
      else:
        break
  logger.debug(f"S{filepath_out=}, {len(positives)} positives sampled")
  
  negatives = []
  with tqdm(total=len(positives), disable=not verbose, leave=True) as pbar:
    while len(negatives) < len(positives):
      u = random.choice(nodes_mature_list)
      nbs_u = list(graph_mature[u])
      nb_u = random.choice(nbs_u)
      v = random.choice(list(graph_mature[nb_u]))
      if v not in nbs_u and u < v and (u,v) not in probes_set:
        negatives.append((u,v))
        pbar.update(1)
  logger.debug(f"S{filepath_out=}, {len(negatives)} negatives sampled")
  
  result = pd.concat(
    [pd.Series(False, index=negatives), pd.Series(True, index=positives)]) #type: ignore
  
  result.to_pickle(filepath_out)
  
  
@app.command()
def all(n_jobs: int = -1, 
        verbose: bool = False, 
        shuffle: bool = True, 
        seed: int = 42,
        sample_size: int = 10_000):
    iterations = [(network, nswap_perc, None if nswap_perc == 0 else method)
                  for network in np.arange(1, 31)
                  for nswap_perc in np.arange(-100, 101, 20)
                  for method in ['a', 'b']
                  if not (nswap_perc == 0 and method == 'b')]
    iterator = list(iterations)
    if shuffle:
        random.seed(seed)
        random.shuffle(iterator)
    ProgressParallel(n_jobs=n_jobs, total=len(iterations))(
        delayed(single)(
            network_index=network_index, 
            nswap_perc=nswap_perc,
            method=method,
            sample_size=sample_size,
            verbose=verbose) 
        for network_index, nswap_perc, method in iterations
    )  
  
@app.command()
def check():
    iterator = list(
        itertools.product(
            [network for network in np.arange(1, 31) 
             if network not in [15, 17, 26, 27]], 
            np.arange(-100, 101, 20)
        )
    )
    for n, nswap_perc in iterator:
        if not os.path.isfile(f'data/{n:02}/{nswap_perc:+04.0f}/samples.pkl'):
            print(n, nswap_perc)
  
  
if __name__ == '__main__':
  app()