import datetime
import json
import os
import subprocess
import tempfile
import typing

import joblib
import matplotlib.axes
import matplotlib.axis
import matplotlib.ticker
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import sklearn.metrics
import seaborn as sns
from tqdm.auto import tqdm

from .pipeline.dataPreparation import SPLIT_FRACTION
from .features import TIME_STRATEGIES, Strategy, Experiment


def plot_datetime(
  datetimes: pd.Series, *, 
  t_min: typing.Optional[pd.Timestamp] = None,
  t_split: typing.Optional[pd.Timestamp] = None,
  t_max: typing.Optional[pd.Timestamp] = None
  ) -> None:
  """Plot the maturing and probing interval.

  Args:
    datetimes: pd.Series containing pd.Date
    t_min, t_split, t_max: Optional; Dates corresponding to the start of the
      maturing interval, end of the maturing interval and end of the probing
      interval, respectively.

  Usage:
   tlp.analysis.plot_datetime(edgelist['datetimes'])
  """
  t_min_, t_split_, t_max_ = datetimes.quantile([0,SPLIT_FRACTION,1])
  if t_min is None: t_min = t_min_
  if t_split is None: t_split = t_split_
  if t_max is None: t_max = t_max_
  with plt.rc_context({'xtick.top': True, 'ytick.right': True, 
                       'figure.figsize': (20, 4)}):
    datetimes.value_counts(normalize=True).sort_index().cumsum().plot(
      xlabel='Year', ylabel='CDF', xlim=(datetimes.min(), datetimes.max()),
      ylim=(0,1), grid=True, legend=False, label='cumulative distribution')
    plt.fill_between([t_min, t_split], 0, 1, color='C1', alpha=.5, 
                     label='maturing interval')
    plt.fill_between([t_split, t_max], 0, 1, color='C2', alpha=.5, 
                     label='probing interval')
    plt.legend() 
    
def class_imbalance(targets: np.ndarray) -> pd.DataFrame:
  """Report the class imbalance in the targets."""
  return pd.DataFrame({
    'absolute': pd.Series(targets).value_counts(),
    'relative': pd.Series(targets).value_counts(normalize=True)})
  
def plot_path_distributions(df: pd.DataFrame, smooth: bool = True) -> None:
  """Plot the path distributions in one plot.
  
  Args:
    df: pd.DataFrame containing the path distributions for various graphs.
    smooth: Optional; If true, make the graph more smooth.
    
  Typical usage is with the function tlp.analysis.network_stats:
    edgelist_dict = {
      'complete_graph': joblib.load(f'{dataset_id}/edgelist.pkl'),
      'mature_graph': joblib.load(f'{dataset_id}/edgelist_mature.pkl')}
    df = tlp.analysis.network_stats(edgelist_dict, path=stats')
    tlp.analysis.plot_path_distributions(df)
  """
  df = df.apply(lambda x: x/x.sum(), axis='index')
  if smooth:
    df = (
      df
      .reindex(labels=np.arange(0, df.index.max(), .1))
      .interpolate('akima', limit_area='inside')
      .dropna(how='all'))

  with plt.rc_context({'xtick.top': True, 'ytick.right': True}):
    ax = plt.gca()
    df.plot(kind='area', ax=ax, xlim=(1), ylim=(0), stacked=False, alpha=.5, 
            title='Path distribution', xlabel='Path length', 
            ylabel='Probability')
    assert isinstance(ax, matplotlib.axis.Axis)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator())

def plot_datetime_strategies(
  datetimes: pd.Series, *,
  time_strategies: dict[str, Strategy] = TIME_STRATEGIES):
  """Plot the mapping from datetime with the various time strategies.
  
  Args:
    datetimes: pd.Series containing all the considered datetimes (one per edge)
    time_strategies: Optional; A list containing functions that can map the
      datetime column of the edgelist to a float. See 
      tlp.feature.TIME_STRATEGIES.
  
  Usage:
    tlp.analysis.plot_datetime_strategies(edgelist['datetime'])
  """
  index = pd.to_datetime(
    np.linspace(datetimes.min().value, datetimes.max().value))
  with plt.rc_context({'xtick.top': True, 'ytick.right': True}):
    _, ax = plt.subplots(figsize=(4,4))
    df = pd.DataFrame(
      {str: func(index) for str, func in time_strategies.items()}, 
      index=index)
    df.plot(ax=ax, xlabel='Year', ylabel='Proportion',  
            xlim=datetimes.agg(['min', 'max']), ylim=(0,1))
    
    ax.legend(title='Time strategy') # type: ignore
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(.1)) # type: ignore
    
def plot_datetime_distribution(
  datetimes: pd.Series, *,
  time_strategies: dict[str, Strategy] = TIME_STRATEGIES) -> None:
  """Plot the distribution of the transformed datetime values by the various
  time strategies.
  
  Args:
    datetimes: pd.Series containing all the considered datetimes (one per edge).
    time_strategies: Optional; A list containing functions that can map the
      datetime column of the edgelist to a float. See 
      tlp.feature.TIME_STRATEGIES.
  
  Usage:
    tlp.analysis.plot_datetime_distribution(edgelist['datetime'])
  """
  data = pd.concat(
    {time_str: time_func(datetimes) 
     for time_str, time_func in time_strategies.items()}, 
    names=['Time strategy', 'index']
    )
  data = data.reset_index('Time strategy').reset_index(drop=True)
  rc = {'xtick.top': True, 'ytick.right': True, 'figure.figsize': (4,4)}
  with plt.rc_context(rc), sns.axes_style('ticks'):  
    ax = sns.ecdfplot(data=data, x='datetime', hue='Time strategy')
    ax.set_xlim((0,1)) # type: ignore
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(.1)) # type: ignore
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(.1)) # type: ignore
    
def plot_score(feature_dict: dict[Experiment, pd.Series], targets: np.ndarray
               ) -> None:
  """Plot the scores as obtained in feature construction.
  
  Args:
    feature_dict: A dict such as obtained from all methods in tlp.features.
    targets: np.array containing the targets belonging to the instances used in
      feature construction.
  
  Usage:
    tlp.analysis.plot_score(feature_dict, targets)
  """
  data = pd.concat(
    [pd.DataFrame(
      {'Ground truth': targets, 'Score': scores, **experiment._asdict()}) 
      for experiment, scores in feature_dict.items()],
    ignore_index=True)
  data['Feature'] = (
    data['feature'] + ' ' +
    data['time_aware'].replace({True: '(time-aware)', False: '(time-agnostic)'})
    + ' ' + data['time_strategy'] + ' ' + data['aggregation_strategy'])
  
  g = sns.displot(
    data=data, x='Score', hue='Ground truth', col='Feature', kind='kde', 
    palette=['red', 'green'], height=4.5, col_wrap=6,
    rug=True, rug_kws=dict(height=-.01, clip_on=False), 
    facet_kws=dict(sharex=False, sharey=False, despine=False))
  g.set_titles('{col_name}')
  
def plot_roc_auc(feature_dict: dict[Experiment, pd.Series], targets: np.ndarray
                 ) -> None:
  """Plot the ROC curves.

  Args:
      feature_dict (dict[Experiment, pd.Series]): A dict such as obtained from 
        all methods in tlp.features.
      targets (np.ndarray): The targets belonging to the instances used in 
        feature construction.
  """
  data = dict()
  no_positives = np.sum(targets)
  for experiment, scores in feature_dict.items():
    fpr, tpr, _ = (
      sklearn.metrics.roc_curve(y_true=targets, y_score=scores)) # type: ignore 

    rocauc = pd.Series(tpr*no_positives, 
                      index=pd.Index(fpr*no_positives, name='False positives'))
    rocauc = rocauc[~rocauc.index.duplicated()]
    data[experiment] = rocauc
    
  data = pd.DataFrame(data).interpolate().reset_index().melt(
    id_vars='False positives',
    var_name=list(Experiment._fields), # type: ignore
    value_name='True positives')
  
  g = sns.relplot(
    data=data, x='False positives', y='True positives', hue='time_aware', 
    col='feature', palette={True: 'green', False: 'red'}, aspect=1, kind='line', 
    height=4, ci=100)
  g.map(plt.axline, xy1=(0,0), xy2=(1,1), c='black') # type: ignore
  g.set(xlim=(0,no_positives), ylim=(0,no_positives), xlabel='False Positives', 
        ylabel='True positives')
  
def get_auc(feature_dict: dict[Experiment, pd.Series], targets: np.ndarray
            ) -> pd.Series:
  """Get the auc performance of the given features.

  Args:
    feature_dict (dict[Experiment, pd.Series]): A dict such as obtained from 
      all methods in tlp.features.
    targets (np.ndarray): The targets belonging to the instances used in 
      feature construction.

  Returns:
      pd.DataFrame: AUC performance. The columns are the same as the
        attributes of the Experiment class, except for one added column 'auc'.
  """  
  data = pd.Series(
    [sklearn.metrics.roc_auc_score(targets, scores) # type: ignore
      for scores in feature_dict.values()], 
    index=pd.Index(feature_dict, name=Experiment._fields), # type: ignore
    name='auc') 
  return data