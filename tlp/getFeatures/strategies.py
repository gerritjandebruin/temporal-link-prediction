import collections.abc
from numbers import Real
import typing

import scipy.stats
import numpy as np
import pandas as pd

# Typing
Strategy = collections.abc.Callable
Strategies = dict[str, Strategy]

# All transformed datetime values are mapped between LOWER_BOUND and 1.
LOWER_BOUND = 0.2

def _exp_time(x: pd.Series) -> pd.Series:
  """Apply y=3*exp(x) and normalize it between (0,1)."""
  return np.exp(3*x) / np.exp(3)

def _rescale(x: pd.Series, *, lower_bound: float = 0.2) -> pd.Series:
  """_rescale the provided array.

  Args:
    lower_bound: Instead of normalizing between 0 and 1, normalize between 
      lower_bound and 1.
  """
  lowest, highest = np.quantile(x, [0, 1])
  return lower_bound + (1-lower_bound)*(x-lowest)/(highest-lowest)

def lin(x: pd.Series, lower_bound=LOWER_BOUND):
  return _rescale(_rescale(x.astype(int)), lower_bound=lower_bound)

def exp(x: pd.Series, lower_bound=LOWER_BOUND):
  return _rescale( _exp_time(_rescale(x.astype(int))), lower_bound=lower_bound)

def sqrt(x: pd.Series, lower_bound=LOWER_BOUND):
  return _rescale(np.sqrt(_rescale(x.astype(int))), lower_bound=lower_bound)

# All strategies used to transform the datetime values.
TIME_STRATEGIES = {'lin': lin, 'exp': exp, 'sqrt': sqrt}

# All strategies used to aggregate multiple edges between two nodes in case of
# graphs with discrete event data.
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

def diff(x: tuple[Real, Real]) -> Real: return x[1] - x[0]

NODEPAIR_STRATEGIES = {'sum': sum, 'diff': diff, 'max': max, 'min': min}