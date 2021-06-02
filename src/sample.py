import random

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import typer

from .logger import logger 

app = typer.Typer()

def check_common_neighbor(G, u, v):
  for w in G[u]:
    if w in G[v] and w not in (u,v):
      return True
  else:
    return False

@app.command()
def single(in_edgelist: str, 
           out_sampled_file: str, 
           cutoff: int = 2,
           sample_size: int = 10000,
           verbose: bool = True):
  positives = list()
  
  assert cutoff == 2, "Not implement for any other cutoff value than 2."
  
  edgelist = pd.read_pickle(in_edgelist).loc[
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
  random.seed(42)
  random.shuffle(probes_list)
  
  positives = []
  with tqdm(total=sample_size, disable=not verbose, leave=False) as pbar:
    for u, v in tqdm(probes_list, position=1, disable=not verbose, leave=False):
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
  logger.debug(f"S{out_sampled_file=}, {len(positives)} positives sampled")
  
  negatives = []
  with tqdm(total=len(positives), disable=not verbose, leave=False) as pbar:
    while len(negatives) < len(positives):
      u = random.choice(nodes_mature_list)
      nbs_u = list(graph_mature[u])
      nb_u = random.choice(nbs_u)
      v = random.choice(list(graph_mature[nb_u]))
      if v not in nbs_u and u < v and (u,v) not in probes_set:
        negatives.append((u,v))
        pbar.update(1)
  logger.debug(f"S{out_sampled_file=}, {len(negatives)} negatives sampled")
  
  result = pd.concat(
    [pd.Series(False, index=negatives), pd.Series(True, index=positives)]) #type: ignore
  
  result.to_pickle(out_sampled_file)
  
if __name__ == '__main__':
  app()