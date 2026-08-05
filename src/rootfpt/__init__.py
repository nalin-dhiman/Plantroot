"""ROOT-FPT reduced research pipeline.

All outputs are synthetic unless an experiment manifest explicitly identifies
an empirical data source and calibration.
"""

from rootfpt.config import config_hash, load_yaml

__all__ = ["config_hash", "load_yaml"]
__version__ = "1.1.1"
