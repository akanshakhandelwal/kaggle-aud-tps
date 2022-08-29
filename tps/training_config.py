

from pathlib import Path
from typing import Union, Any
from pydantic import BaseModel


class TrainingConfig(BaseModel):
    data_path: Union[str, Path]
    train_path: Union[str, Path]
    test_path: Union[str, Path]
    log_path: Union[str, Path] = "log.md"
    submission_path: Union[str, Path] = "submission"
    n_splits: int = 5
    n_trials: int = 1
    model: Any
    hyperparams: Any

    class Config:
        arbitrary_types_allowed = True
