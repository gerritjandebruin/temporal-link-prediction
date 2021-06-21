import joblib
from tqdm.auto import tqdm

delayed = joblib.delayed

class ProgressParallel(joblib.Parallel):
  def __init__(self, use_tqdm=True, total=None, desc=None, unit='it', 
               leave=False, *args, **kwargs):
    self._use_tqdm = use_tqdm
    self._total = total
    self._desc = desc
    self._unit = unit
    self._leave = leave
    super().__init__(*args, **kwargs)

  def __call__(self, *args, **kwargs):
    with tqdm(disable=not self._use_tqdm, total=self._total, 
              desc=self._desc, unit=self._unit, leave=self._leave, 
              miniters=1, mininterval=0) as self._pbar:
      return joblib.Parallel.__call__(self, *args, **kwargs)

  def print_progress(self):
    if self._total is None: 
      self._pbar.total = self.n_dispatched_tasks
    self._pbar.n = self.n_completed_tasks
    self._pbar.refresh()