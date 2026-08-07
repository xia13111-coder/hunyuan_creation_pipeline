"""Compatibility facade for layered visual-quality contracts.

New code imports the owning module under :mod:`quality_contracts`.  This
facade preserves the previous symbols while callers migrate.
"""

from __future__ import annotations

from .quality_contracts.constants import *
from .quality_contracts.constants import __all__ as _constants_all
from .quality_contracts.diagnostics import *
from .quality_contracts.diagnostics import __all__ as _diagnostics_all
from .quality_contracts.metrics import *
from .quality_contracts.metrics import __all__ as _metrics_all
from .quality_contracts.repair import *
from .quality_contracts.repair import __all__ as _repair_all
from .quality_contracts.resolution import *
from .quality_contracts.resolution import __all__ as _resolution_all


__all__ = [
    *_constants_all,
    *_metrics_all,
    *_diagnostics_all,
    *_repair_all,
    *_resolution_all,
]
