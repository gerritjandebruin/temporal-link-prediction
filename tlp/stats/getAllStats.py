from .cheapStats import read_cheap_statistics
from .diameter import get_diameter
from .pathDistribution import get_average_shortest_simple_path_length
from ..predict import read_auc_from_logistic_regression
from ..helpers import recursive_file_loading, get_categories

def get_all() -> pd.DataFrame:
  """Get a pd.DataFrame containing all the statistics that are already 
  calculated."""
  stats = read_cheap_statistics()
  diameters = get_diameter()
  average_path = et_average_shortest_simple_path_length()
  scores = read_auc_from_logistic_regression()
  labels = recursive_file_loading('label.txt')
  categories = get_categories()
  scores_time_agnostic = read_auc_from_logistic_regression('auc_time_agnostic.float')