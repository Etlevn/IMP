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
        # "ols": [
        #     ["par_mkt", "log_mcap", "vol_21d", "vix", "dxy"],
        # ],
        # "ridge": [
        #     ["par_mkt", "log_mcap", "vol_21d", "vix", "dxy"],
        # ],
        # "lasso": [
        #     ["par_mkt", "log_mcap", "vol_21d", "vix", "dxy"],
        # ],
        "elasticnet": [
            ["par_mkt", "log_mcap", "vol_21d", "vix", "dxy"],
        ],
        "huber": [
            ["par_mkt", "log_mcap", "vol_21d", "vix", "dxy"],
        ],
        "quantile": [
            ["par_mkt", "log_mcap", "vol_21d", "vix", "dxy"],
        ],
    },
    "tr": {
        # "decision_tree": [
        #     ["par_mkt", "log_mcap", "vol_21d", "vix", "dxy"],
        # ],
        # "random_forest": [
        #     ["par_mkt", "log_mcap", "vol_21d", "vix", "dxy"],
        # ],
        # "hist_gradient_boosting": [
        #     ["par_mkt", "log_mcap", "vol_21d", "vix", "dxy"],
        # ],
        "lightgbm": [
            ["par_mkt", "log_mcap", "vol_21d", "vix", "dxy"],
        ],
        "xgboost": [
            ["par_mkt", "log_mcap", "vol_21d", "vix", "dxy"],
        ],
        "catboost": [
            ["par_mkt", "log_mcap", "vol_21d", "vix", "dxy"],
        ],
    },
    "nn": {
        # "mlp_shallow": [
        #     ["par_mkt", "log_mcap", "vol_21d", "vix", "dxy"],
        # ],
        # "mlp": [
        #     ["par_mkt", "log_mcap", "vol_21d", "vix", "dxy"],
        # ],
        # "mlp_deep": [
        #     ["par_mkt", "log_mcap", "vol_21d", "vix", "dxy"],
        # ],
        "mlp_dropout": [
            ["par_mkt", "log_mcap", "vol_21d", "vix", "dxy"],
        ],
        "wide_deep": [
            ["par_mkt", "log_mcap", "vol_21d", "vix", "dxy"],
        ],
        "ft_transformer": [
            ["par_mkt", "log_mcap", "vol_21d", "vix", "dxy"],
        ],
    },
}
