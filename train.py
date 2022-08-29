from tps.trainer import TPSTrainer
from tps.training_config import TrainingConfig
from tps.optuna_params import get_lgbm_params
from lightgbm import LGBMClassifier


c = TrainingConfig(
    data_path="data/",  # denoised data from raddar
    train_path="data/train.csv",
    test_path="data/test.csv",  # aggregated features
    n_splits=3,
    log_path="lgbm_log.md",
    submission_path="submission-stack",
    n_trials=20,
    model=LGBMClassifier,
    hyperparams=get_lgbm_params,
)
t = TPSTrainer(config=c).setup().model_stack_fn().create_submission()