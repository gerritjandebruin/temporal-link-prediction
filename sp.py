import collections
import concurrent.futures
import os
import typing

import joblib
import networkx as nx
from tqdm.auto import tqdm

import tlp

edgelist_mature_dict = tlp.recursive_file_lookup('edgelist_mature.pkl')
instances_dict = tlp.recursive_file_lookup( 'instances_sampled.pkl')

joblib.Parallel(n_jobs=len(instances_dict))(
  joblib.delayed(tlp.features.sp)(
    edgelist_mature_dict[idx], instances, output_path=f'data/{idx}/features', 
    index=int(idx)) 
  for idx, instances in instances_dict.items())