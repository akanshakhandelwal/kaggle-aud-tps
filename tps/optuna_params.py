

from typing import Any, Dict
from pydantic import BaseModel
from typing import Optional




def get_lgbm_params(trial: Any) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 2000, step=10),
        "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "min_child_samples": trial.suggest_int("min_child_samples", 100, 3000),
        "reg_lambda": trial.suggest_int("reg_lambda", 0, 100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0, 1),
        "max_bins": trial.suggest_int("max_bins", 128, 1024),
        "random_state": 42,
        "n_jobs": -1,
        "boosting": 'gbdt',
        "bagging_freq": 10,
        "num_boost_round": 1000,
        'feature_fraction': 0.90,
        'bagging_fraction': 0.90,
      
    }


