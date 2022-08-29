import numpy as np  # type:ignore
import pandas as pd  # type:ignore
from pathlib import Path
from typing import Union
#import kaggle
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import os


from tps.utils import (
    download_kaggle_dataset,
    submit_to_competition,
    download_kaggle_competition_data,
    download_kaggle_competition_data_file
)





download_kaggle_competition_data_file(
   competition_name="tabular-playground-series-aug-2022", savepath="data/",filename="train.csv"
)
download_kaggle_competition_data_file(
   competition_name="tabular-playground-series-aug-2022", savepath="data/",filename="test.csv"
)