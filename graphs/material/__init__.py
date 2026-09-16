"""Material graph package: split by concern for maintainability."""

from .graph import build_material_graph
from .state import MaterialState

__all__ = ["MaterialState", "build_material_graph"]
