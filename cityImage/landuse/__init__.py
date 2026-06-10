"""Land-use derivation, classification, sparse representation, and assignment.

The ``cityImage.landuse`` package contains the land-use semantics used by
cityImage building workflows and pragmatic landmark scoring. It groups together:

* tag taxonomies and OSM/domain group definitions;
* derivation of raw land-use candidates from OSM-style attributes;
* classification of raw values into cityImage land-use groups;
* assignment of point/polygon land-use layers to buildings;
* sparse and multi-label non-OSM land-use workflows.

The top-level ``import cityImage as ci`` API exposes the main public functions
through lazy loading. Direct specialist imports can use this package namespace,
for example::

    from cityImage.landuse import classify_land_use
    from cityImage.landuse.derive import derive_land_uses_raw_fromOSM
"""

from __future__ import annotations

from .assign import *  # noqa: F403
from .classify import *  # noqa: F403
from .derive import *  # noqa: F403
from .sparse import *  # noqa: F403
from .tags import *  # noqa: F403
from .utils import *  # noqa: F403
