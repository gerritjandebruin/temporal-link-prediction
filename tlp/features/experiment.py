import typing

class Experiment(typing.NamedTuple):
  """Class containing all information to unique identify a given experiment.
  An experiment is the combination of a feature name, a time_aware bool and 
    aggregation and time strategy, if available. Iteration over this object is 
    allowed and yields the values of the attributes. This class can conveniently
    be turned in a dict using the functin _asdict().  
  """
  feature: str
  time_aware: bool
  aggregation_strategy: typing.Optional[str] = None
  time_strategy: typing.Optional[str] = None
  nodepair_strategy: typing.Optional[str] = None