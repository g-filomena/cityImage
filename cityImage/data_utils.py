"""Small tabular helpers used by cityImage internals.

This module intentionally contains only lightweight helpers that preserve
cityImage scoring/export semantics. Generic list expansion, arbitrary range
rescaling, and file IO helpers were removed during the hard API cleanup; use
Pandas/GeoPandas directly for those operations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def scaling_columnDF(series: pd.Series, inverse: bool = False) -> pd.Series:
    """Scale numeric values to the unit interval.

    Parameters
    ----------
    series : pandas.Series
        Numeric values to scale.
    inverse : bool, default False
        If True, return the inverse scale where the original maximum maps to
        0 and the original minimum maps to 1.

    Returns
    -------
    pandas.Series
        Scaled values indexed like the input. Constant input maps to 0, or 1
        when ``inverse=True``.
    """
    if series.max() == series.min():
        return pd.Series(1.0 if inverse else 0.0, index=series.index)

    scaled = (series - series.min()) / (series.max() - series.min())
    return 1 - scaled if inverse else scaled


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stable scalar dtypes for graph export workflows.

    Integer and float NumPy dtypes are converted to their Python scalar
    equivalents where possible. Object columns are stringified to avoid mixed
    object/list values in downstream graph and file-export steps.
    """
    out = df.copy()
    for column in out.columns:
        if out[column].dtype == np.int64:
            out[column] = out[column].astype(int)
        elif out[column].dtype == np.float64:
            out[column] = out[column].astype(float)
        elif out[column].dtype == "object":
            out[column] = out[column].astype(str)

    return out
