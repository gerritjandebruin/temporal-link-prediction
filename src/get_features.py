import collections
from enum import Enum
import itertools
import os

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats
import typer

from .logger import logger
from .progress_parallel import ProgressParallel, delayed

app = typer.Typer()

# region STRATEGIES


def _rescale(x: pd.Series, *, lower_bound: float = 0.2) -> pd.Series:
    """_rescale the provided array.

    Args:
      lower_bound: Instead of normalizing between 0 and 1, normalize between 
        lower_bound and 1.
    """
    lowest, highest = np.quantile(x, [0, 1])
    return lower_bound + (1-lower_bound)*(x-lowest)/(highest-lowest)


def _exp_time(x: pd.Series) -> pd.Series:
    """Apply y=3*exp(x) and normalize it between (0,1)."""
    return np.exp(3*x) / np.exp(3)


def lin(x: pd.Series, lower_bound=.2):
    return _rescale(_rescale(x.astype(int)), lower_bound=lower_bound)


def exp(x: pd.Series, lower_bound=.2):
    return _rescale(_exp_time(_rescale(x.astype(int))), lower_bound=lower_bound)


def sqrt(x: pd.Series, lower_bound=.2):
    return _rescale(np.sqrt(_rescale(x.astype(int))), lower_bound=lower_bound)


TIME_STRATEGIES = {'lin': lin, 'exp': exp, 'sqrt': sqrt}

AGGREGATION_STRATEGIES = {
    'q0': np.min,
    'q25': lambda array: np.quantile(array, .25),
    'q50': np.median,
    'q75': lambda array: np.quantile(array, .75),
    'q100': np.max,
    'm0': np.sum,
    'm1': np.mean,
    'm2': np.var,
    'm3': scipy.stats.skew,
    'm4': scipy.stats.kurtosis
}

NODEPAIR_STRATEGIES = {'sum': sum, 'diff': lambda x: x[1]-x[0],
                       'max': max, 'min': min}
# endregion

# region FEATURES


def aa(edgelist_mature, instances):
    graph_mature = nx.from_pandas_edgelist(edgelist_mature)
    return [p for _, _, p in nx.adamic_adar_index(graph_mature, instances)]


def aa_time_aware(edgelist_mature, instances,
                  time_strategy, aggregation_strategy):
    df = edgelist_mature[['source', 'target', 'datetime']].assign(
        datetime=lambda x: time_strategy(x['datetime'])
    )
    G = nx.from_pandas_edgelist(df, edge_attr=True, create_using=nx.MultiGraph)
    scores = list()
    for u, v in instances:
        score = [
            aggregation_strategy([e['datetime'] for e in G[u][z].values()]) *
            aggregation_strategy([e['datetime'] for e in G[v][z].values()]) /
            np.log(len(G[z]))
            for z in nx.common_neighbors(G, u, v)
        ]
        scores.append(sum(score))
    return scores


def na(edgelist_mature, instances, time_strategy, aggregation_strategy,
       nodepair_strategy):
    df = edgelist_mature[['source', 'target', 'datetime']].assign(
        datetime=lambda x: time_strategy(x['datetime'])
    )
    G = nx.from_pandas_edgelist(df, edge_attr=True, create_using=nx.MultiGraph)
    scores = list()
    for u, v in instances:
        activity_u = aggregation_strategy(
            [e['datetime'] for nb in G[u] for e in G[u][nb].values()]
        )
        activity_v = aggregation_strategy(
            [e['datetime'] for nb in G[v] for e in G[v][nb].values()]
        )
        scores.append(nodepair_strategy([activity_u, activity_v]))
    return scores


def single_source_number_paths_length_2(graph: nx.Graph, source):
    result = collections.Counter()
    for nb in graph[source]:
        for nnb in graph[nb]:
            result[nnb] += 1
    return result


def sp(edgelist_mature, instances):
    G = nx.from_pandas_edgelist(edgelist_mature[['source', 'target']])
    paths_of_length_2 = {
        n: single_source_number_paths_length_2(G, n) for n in instances[:, 0]
    }
    scores = [paths_of_length_2[u][v] for u, v in instances]
    return scores


def jc(edgelist_mature, instances):
    G = nx.from_pandas_edgelist(edgelist_mature[['source', 'target']])
    scores = [p for _, _, p in nx.jaccard_coefficient(G, instances)]
    return scores


def cn(edgelist_mature, instances):
    G = nx.from_pandas_edgelist(edgelist_mature[['source', 'target']])
    scores = [len(list(nx.common_neighbors(G, u, v))) for u, v in instances]
    return scores


def pa(edgelist_mature, instances):
    G = nx.from_pandas_edgelist(edgelist_mature[['source', 'target']])
    scores = [p for _, _, p in nx.preferential_attachment(G, instances)]
    return scores


def cn_time_aware(edgelist_mature, instances, time_strategy,
                  aggregation_strategy):
    df = edgelist_mature[['source', 'target', 'datetime']].assign(
        datetime=lambda x: time_strategy(x['datetime'])
    )
    G = nx.from_pandas_edgelist(df, create_using=nx.MultiGraph, edge_attr=True)

    scores = list()
    for u, v in instances:
        score = [
            aggregation_strategy([e['datetime'] for e in G[u][z].values()]) +
            aggregation_strategy([e['datetime'] for e in G[v][z].values()])
            for z in nx.common_neighbors(G, u, v)
        ]
        scores.append(sum(score))
    return scores


def jc_time_aware(edgelist_mature, instances, time_strategy,
                  aggregation_strategy):
    # logger.debug(f'Start converting edgelist.')
    df = edgelist_mature[['source', 'target', 'datetime']].assign(
        datetime=lambda x: time_strategy(x['datetime'])
    )
    G = nx.from_pandas_edgelist(df, create_using=nx.MultiGraph, edge_attr=True)

    scores = list()
    for u, v in instances:
        # logger.debug('Get CN')
        cn = [
            aggregation_strategy([e['datetime'] for e in G[u][z].values()]) +
            aggregation_strategy([e['datetime'] for e in G[v][z].values()])
            for z in nx.common_neighbors(G, u, v)
        ]
        # logger.debug('Get all activity of nodes')
        all_u = [aggregation_strategy([e['datetime'] for e in G[u][a].values()])
                 for a in G[u]]
        all_v = [aggregation_strategy([e['datetime'] for e in G[v][b].values()])
                 for b in G[v]]
        all_activity = sum(all_u) + sum(all_v)
        # logger.debug('Get score')
        score = sum(cn) / all_activity if all_activity != 0 else 0
        scores.append(score)
    return scores


def pa_time_aware(edgelist_mature, instances, time_strategy,
                  aggregation_strategy):
    df = edgelist_mature[['source', 'target', 'datetime']].assign(
        datetime=lambda x: time_strategy(x['datetime'])
    )
    G = nx.from_pandas_edgelist(df, create_using=nx.MultiGraph, edge_attr=True)

    scores = list()
    for u, v in instances:
        all_u = [aggregation_strategy([e['datetime'] for e in G[u][a].values()])
                 for a in G[u]]
        all_v = [aggregation_strategy([e['datetime'] for e in G[v][b].values()])
                 for b in G[v]]
        scores.append(sum(all_u) * sum(all_v))
    return scores
# endregion

@app.command()
def all(network: int = None, 
        network_from: int = 1,
        network_to: int = 31,
        method: str = None, 
        n_jobs: int = -1, 
        include_na: bool = True):
    assert network is None or (network_from == 1 and network_to == 31) 

    # WRAPPER
    def calculate_feature(feature_func, path, out_file, **kwargs):
        features_dir = os.path.join(path, 'features')
        os.makedirs(features_dir, exist_ok=True)
        out_filepath = os.path.join(features_dir, out_file)
        if os.path.isfile(out_filepath):
            return

        # logger.debug('Get edgelist')
        edgelist_mature = (
            pd.read_pickle(os.path.join(path, 'edgelist.pkl'))
            .loc[lambda x: (x['phase'] == 'mature') & (x['source'] != x['target'])]
        )
        # logger.debug('Get instances')
        instances = np.array(
            [i for i in pd.read_pickle(
                os.path.join(path, 'samples.pkl')).index]
        )
        # logger.debug('Get scores')
        scores = np.array(feature_func(edgelist_mature, instances, **kwargs))
        assert len(instances) == len(scores)
        assert scores.ndim == 1
        logger.debug('Save')
        np.save(out_filepath, scores)

    if network is None:
        networks = [network for network in np.arange(network_from, network_to) 
                    if not network in [15, 17, 26, 27]]
    else:
        networks = [network]
    if method is None:
        paths = [
            f'data/{network:02}/{nswap_perc:+04.0f}{method if method != 0 else ""}/'
            for nswap_perc in np.arange(-100, 101, 20)
            for method in ['a', 'b']
            for network in networks
            if not nswap_perc == 0 and method == 'b'
        ]
    else:
        assert method in ['a', 'b']
        paths = [
            f'data/{network:02}/{nswap_perc:+04.0f}{method}/'
            for network in networks
            for nswap_perc in np.arange(-100, 101, 20)
            if not nswap_perc == 0 and method == 'b'
        ]

    # aa
    ProgressParallel(n_jobs=len(paths), total=len(paths), desc='aa')(
        delayed(calculate_feature)(aa, path, 'aa.npy') for path in paths
    )

    # na
    if include_na:
        total = (len(paths)*len(TIME_STRATEGIES)*len(AGGREGATION_STRATEGIES) *
                len(NODEPAIR_STRATEGIES))
        ProgressParallel(n_jobs=n_jobs, total=total, desc='na')(
            delayed(calculate_feature)(
                feature_func=na,
                path=path,
                out_file=f'na_{time_str}_{agg_str}_{nodepair_str}.npy',
                time_strategy=time_func,
                aggregation_strategy=agg_func,
                nodepair_strategy=nodepair_func
            )
            for time_str, time_func in TIME_STRATEGIES.items()
            for agg_str, agg_func in AGGREGATION_STRATEGIES.items()
            for nodepair_str, nodepair_func in NODEPAIR_STRATEGIES.items()
            for path in paths
        )

    # sp
    ProgressParallel(n_jobs=len(paths), total=len(paths), desc='sp')(
        delayed(calculate_feature)(
            feature_func=sp, path=path, out_file='sp.npy')
        for path in paths
    )

    # jc
    ProgressParallel(n_jobs=len(paths), total=len(paths), desc='jc')(
        delayed(calculate_feature)(
            feature_func=jc, path=path, out_file='jc.npy')
        for path in paths
    )

    # cn
    ProgressParallel(n_jobs=len(paths), total=len(paths), desc='cn')(
        delayed(calculate_feature)(
            feature_func=cn, path=path, out_file='cn.npy')
        for path in paths
    )

    # pa
    ProgressParallel(n_jobs=len(paths), total=len(paths), desc='pa')(
        delayed(calculate_feature)(
            feature_func=pa, path=path, out_file='pa.npy')
        for path in paths
    )

    # aa_time_aware
    total = len(paths)*len(TIME_STRATEGIES)*len(AGGREGATION_STRATEGIES)
    ProgressParallel(n_jobs=n_jobs, total=total, desc='aa time-aware')(
        delayed(calculate_feature)(feature_func=aa_time_aware,
                                   path=path,
                                   out_file=f'aa_{time_str}_{agg_str}.npy',
                                   time_strategy=time_func,
                                   aggregation_strategy=agg_func)
        for path in paths
        for time_str, time_func in TIME_STRATEGIES.items()
        for agg_str, agg_func in AGGREGATION_STRATEGIES.items()
    )

    # cn_time_aware
    total = len(paths)*len(TIME_STRATEGIES)*len(AGGREGATION_STRATEGIES)
    ProgressParallel(n_jobs=n_jobs, total=total, desc='cn time-aware')(
        delayed(calculate_feature)(
            feature_func=cn_time_aware,
            path=path,
            out_file=f'cn_{time_str}_{agg_str}.npy',
            time_strategy=time_func,
            aggregation_strategy=agg_func
        )
        for path in paths
        for time_str, time_func in TIME_STRATEGIES.items()
        for agg_str, agg_func in AGGREGATION_STRATEGIES.items()
    )

    # jc_time_aware
    total = len(paths)*len(TIME_STRATEGIES)*len(AGGREGATION_STRATEGIES)
    ProgressParallel(n_jobs=n_jobs, total=total, desc='jc time-aware')(
        delayed(calculate_feature)(
            feature_func=jc_time_aware,
            path=path,
            out_file=f'jc_{time_str}_{agg_str}.npy',
            time_strategy=time_func,
            aggregation_strategy=agg_func
        )
        for path in paths
        for time_str, time_func in TIME_STRATEGIES.items()
        for agg_str, agg_func in AGGREGATION_STRATEGIES.items()
    )

    # pa_time_aware
    total = len(paths)*len(TIME_STRATEGIES)*len(AGGREGATION_STRATEGIES)
    ProgressParallel(n_jobs=n_jobs, total=total, desc='pa time-aware', 
                     pre_dispatch='1*n_jobs')(
        delayed(calculate_feature)(
            feature_func=pa_time_aware,
            path=path,
            out_file=f'pa_{time_str}_{agg_str}.npy',
            time_strategy=time_func,
            aggregation_strategy=agg_func
        )
        for path in paths
        for time_str, time_func in TIME_STRATEGIES.items()
        for agg_str, agg_func in AGGREGATION_STRATEGIES.items()
    )


@app.command()
def single(path: str, n_jobs: int = -1, verbose=True):

    def calculate_feature(function, out_filepath, edgelist_mature, instances,
                          **kwargs):
        if not os.path.isfile(out_filepath):
            logger.debug(f'Making {out_filepath}')
            scores = function(edgelist_mature, instances, **kwargs)
            np.save(out_filepath, scores)
        else:
            logger.debug(f'Skipping {out_filepath}')

    edgelist_file = os.path.join(path, 'edgelist.pkl')
    samples_file = os.path.join(path, 'samples.pkl')
    if (not os.path.isdir(path) or
        not os.path.isfile(edgelist_file) or
            not os.path.isfile(samples_file)):
        return

    features_dir = os.path.join(path, 'features')
    os.makedirs(features_dir, exist_ok=True)
    edgelist_mature = (
        pd.read_pickle(edgelist_file)
        .loc[lambda x: (x['phase'] == 'mature') & (x['source'] != x['target'])]
    )
    instances = np.array([i for i in pd.read_pickle(samples_file).index])

    # Simple features
    simple_funcs = [('aa', aa), ('sp', sp), ('jc', jc), ('cn', cn), ('pa', pa)]
    total = len(simple_funcs)
    ProgressParallel(use_tqdm=verbose,
                     total=total,
                     desc='simple features',
                     leave=True,
                     n_jobs=n_jobs if n_jobs < total else total)(
        delayed(calculate_feature)(func, os.path.join(features_dir, func_str + '.pkl'),
                                   edgelist_mature, instances)
        for func_str, func in simple_funcs
    )

    # NA
    total = (len(TIME_STRATEGIES) *
             len(AGGREGATION_STRATEGIES) *
             len(NODEPAIR_STRATEGIES))
    ProgressParallel(use_tqdm=verbose,
                     total=total,
                     desc='na features',
                     leave=True,
                     n_jobs=total if total < n_jobs else n_jobs)(
        delayed(calculate_feature)(
            function=na,
            out_filepath=os.path.join(features_dir,
                                      f'na_{time_str}_{agg_str}_{nodepair_str}.pkl'),
            edgelist_mature=edgelist_mature, instances=instances,
            time_strategy=time_func,
            aggregation_strategy=agg_func,
            nodepair_strategy=nodepair_func
        )
        for time_str, time_func in TIME_STRATEGIES.items()
        for agg_str, agg_func in AGGREGATION_STRATEGIES.items()
        for nodepair_str, nodepair_func in NODEPAIR_STRATEGIES.items()
    )

    # Time aware functions
    time_aware_funcs = [('aa', aa_time_aware),
                        ('jc', jc_time_aware),
                        ('cn', cn_time_aware),
                        ('pa', pa_time_aware)]
    total = len(TIME_STRATEGIES)*len(AGGREGATION_STRATEGIES) * \
        len(time_aware_funcs)
    ProgressParallel(n_jobs=n_jobs if n_jobs < total else total,
                     total=total,
                     desc='time-aware',
                     leave=True,
                     use_tqdm=verbose)(
        delayed(calculate_feature)(
            func, os.path.join(
                features_dir, f'{func_str}_{time_str}_{agg_str}.npy'),
            edgelist_mature=edgelist_mature, instances=instances,
            time_strategy=time_func, aggregation_strategy=agg_func
        )
        for time_str, time_func in TIME_STRATEGIES.items()
        for agg_str, agg_func in AGGREGATION_STRATEGIES.items()
        for func_str, func in time_aware_funcs
    )

class methods(str, Enum):
    a = "a"
    b = "b"


@app.command()
def check(method: methods):
    method = method.value
    iterator = list(
        itertools.product(
            [network for network in np.arange(1, 31)
             if network not in [15, 17, 26, 27]],
            np.arange(-100, 101, 20)
        )
    )
    result = dict()
    for n, nswap_perc in iterator:
        dir = f'data/{n:02}/{nswap_perc:+04.0f}{method if method != 0 else ""}/features'
        if os.path.isdir(dir):
            result[(n, nswap_perc)] = len(list(os.scandir(dir)))
    print(result)


if __name__ == '__main__':
    app()
