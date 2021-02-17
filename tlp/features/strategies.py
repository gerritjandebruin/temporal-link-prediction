import collections.abc

import numpy as np

Strategy = collections.abc.Callable

# All transformed datetime values are mapped between LOWER_BOUND and 1.
LOWER_BOUND = 0.2

# All strategies used to transform the datetime values.
def _exp_time(x: np.ndarray) -> np.ndarray:
  """Apply y=3*exp(x) and normalize it between (0,1)."""
  return np.exp(3*x) / np.exp(3)

def _rescale(x: np.ndarray, *, lower_bound: float = 0) -> np.ndarray:
  """_rescale the provided array.

  Args:
    lower_bound: Instead of normalizing between 0 and 1, normalize between 
      lower_bound and 1.
  """
  lowest, highest = np.quantile(x, [0, 1])
  return lower_bound + (1-lower_bound)*(x-lowest)/(highest-lowest)

TIME_STRATEGIES = {
  'lin': lambda x: _rescale(
    _rescale(x.astype(int)), lower_bound=LOWER_BOUND), 
  'exp': lambda x: _rescale(
    _exp_time(_rescale(x.astype(int))), lower_bound=LOWER_BOUND), 
  'sqrt': lambda x: _rescale(
    np.sqrt(_rescale(x.astype(int))), lower_bound=LOWER_BOUND)}

# All strategies used to aggregate multiple edges between two nodes in case of
# graphs with discrete event data.
AGGREGATION_STRATEGIES = {
  'mean': np.mean, 'sum': np.sum, 'max': np.max, 'median': np.median}