"""Editable model menu for ``main.py``.

Paste Optuna feature lists into ``MODELS``. Empty list = skip that model.
Family hyperparameters stay in ``config/config_{lin,tr,nn}.py``.
Shared defaults: ``config/config.py``.
Run knobs (``VERBOSE_PRINT`` / ``SAVE_PREDICTIONS``): ``main.py``.
Run snapshot (not this file): ``output/config_main.json``.
"""

from __future__ import annotations

# family → model → one feature list, or a list of feature lists (multi-combo).
# Empty list = skip that model. A flat list is treated as a single combination.
MODELS: dict[str, dict[str, list]] = {
    "lin": {
        "ols": [
            [],
        ],
        "ridge": [
            [],
        ],
        "lasso": [
            [],
        ],
    },
    "tr": {
        "decision_tree": [
            [],
        ],
        "random_forest": [
            [],
        ],
        "hist_gradient_boosting": [
            [],
        ],
    },
    "nn": {
        "mlp_shallow": [
            [],
        ],
        "mlp": [
            [],
        ],
        "mlp_deep": [
            [],
        ],
    },
}
