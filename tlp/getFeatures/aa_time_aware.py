import os
import pickle

import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from .strategies import AGGREGATION_STRATEGIES, TIME_STRATEGIES, Strategies
from .core import get_edgelist_and_instances, Experiment
from ..helpers import file_exists, print_status


def aa_time_aware(
  path: str, *, 
  aggregation_strategies: Strategies = AGGREGATION_STRATEGIES,
  time_strategies: Strategies = TIME_STRATEGIES,
  verbose: bool = False
  ) -> None:
  """Returns the time aware Adamic Adar feature for the given instances based
   on the provided edgelist. This feature is calculated for all the possible
   combinations of the aggregation_strategies and time_strategies.
  
  Args:
    path: The path should contain edgelist_mature.pkl and 
      instances_sampled.npy. Result is stored at path/aa_time_agnostic.pkl.
    aggregation_strategies: Optional; A list containing functions that can 
      aggregate multiple observed events between two nodes to a single value.
      See AGGREGATION_STRATEGIES.
    time_strategies: Optional; A list containing functions that can map the
      datetime column of the edgelist to a float. See 
      tlp.feature.TIME_STRATEGIES.
    verbose: Optional; If true, show tqdm progressbar.
  
  Stores at os.path.join(output_path, 'AA_time_aware.pkl'):
    A dict with as key a NamedTuple (Experiment) and as value a np.array
      containing the scores.
  """  
  feature_path = os.path.join(path, 'features')
  file = os.path.join(feature_path, 'aa_time_aware.pkl')
  if file_exists(file, verbose=verbose): return
  
  os.makedirs(feature_path, exist_ok=True)

  # Read in
  edgelist, instances = get_edgelist_and_instances(path, verbose=verbose)

  result = dict()

  with tqdm(time_strategies.items(), disable=not verbose, leave=False) as it:
    for time_str, time_func in it:
      it.set_postfix({'time strategy': time_str})
      edgelist['datetime_transformed'] = time_func(edgelist['datetime']) 
      G = nx.from_pandas_edgelist(
        edgelist, edge_attr=True, create_using=nx.MultiGraph)

      with tqdm(aggregation_strategies.items(), disable=not verbose, leave=False
                ) as it2:
        for agg_str, agg_func in it2:
          it2.set_postfix({'aggregation strategy': agg_str})
          # Calculation
          experiment = Experiment(
            'AA', time_aware=True, aggregation_strategy=agg_str, 
            time_strategy=time_str)
          scores = [
            sum(
              [
                agg_func(
                  [
                    edge_attributes['datetime_transformed']
                    for edge_attributes in G.get_edge_data(u, z).values()
                  ]
                ) *
                agg_func(
                  [
                  edge_attributes['datetime_transformed'] 
                    for edge_attributes in G.get_edge_data(v, z).values()
                  ]
                ) /
                np.log(len(list(G.neighbors(z))))
                for z in nx.common_neighbors(G, u, v)
              ]
            )
            for u, v in tqdm(instances, leave=False, disable=not verbose, 
                            unit='instances')
          ]
          result[experiment] = np.array(scores)
  
  # Store
  with open(file, 'wb') as file:
    pickle.dump(result, file)
