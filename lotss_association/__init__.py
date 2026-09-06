"""Rule-based radio Gaussian-component association package."""

from importlib.metadata import PackageNotFoundError, version as _distribution_version

from .association import AssociationResult, run_component_association
from .utils import load_yaml, validate_config

try:
    __version__ = _distribution_version("lotss-association")
except PackageNotFoundError:
    # Source checkouts remain importable before an editable install.
    __version__ = "0.1.1"

__all__ = ["__version__", "AssociationResult", "run_component_association", "load_yaml", "validate_config"]
