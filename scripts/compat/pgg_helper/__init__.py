"""Minimal unpickling shim for the raw master-data pickles.

The pickles in ``data/raw/`` were written from a ``pgg_helper.preprocess.master_data``
instance in the original study repository. Unpickling only needs the *class* to
exist at the same import path; ``__init__`` is never called, and every attribute we
use (``df_rounds``, ``df_games``, ``df_players``, ``df_treatment_config``) is a plain
pandas DataFrame stored on the instance.

Vendoring this stub instead of the original package keeps the case study free of that
package's heavy dependencies (keras, scikit-optimize, xgboost).
"""

from . import preprocess  # noqa: F401
