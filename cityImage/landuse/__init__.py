"""Land-use derivation, classification, and assignment.

This package is the hard replacement for the previous flat modules:

* ``land_use_assign.py``  -> ``cityImage.landuse.assign``
* ``land_use_classify.py`` -> ``cityImage.landuse.classify``
* ``land_use_derive.py`` -> ``cityImage.landuse.derive``
* ``land_use_sparse.py`` -> ``cityImage.landuse.sparse``
* ``land_use_tags.py`` -> ``cityImage.landuse.tags``
* ``land_use_utils.py`` -> ``cityImage.landuse.utils``

The top-level ``import cityImage as ci`` API is preserved through lazy loading
in ``cityImage.__init__``. Direct imports should use the new package namespace,
for example:

```python
from cityImage.landuse import classify_land_use
from cityImage.landuse.derive import derive_land_uses_raw_fromOSM
```
"""

from __future__ import annotations

from .assign import *  # noqa: F403
from .classify import *  # noqa: F403
from .derive import *  # noqa: F403
from .sparse import *  # noqa: F403
from .tags import *  # noqa: F403
from .utils import *  # noqa: F403

