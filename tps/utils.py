
import numpy as np  # type:ignore
import pandas as pd  # type:ignore
from pathlib import Path
from typing import Union
#import kaggle
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import os

def download_kaggle_competition_data(
    *,
    competition_name: str,
    savepath: Union[str, Path] = ".",
    overwrite: bool = False,
):
    # Fetch the user's competition list and verigy that supplied value of competition_name is valid
    api = KaggleApi()
    api.authenticate()
    # os.environ['KAGGLE_CONFIG_DIR'] = "/config/.kaggle/"
    api.competition_download_files(competition_name, savepath)

    with zipfile.ZipFile(Path(savepath) / competition_name + ".zip", "r") as zipref:
        zipref.extractall(savepath)

def download_kaggle_competition_data_file(
    *,
    competition_name: str,
    filename :str,
    savepath: Union[str, Path] = ".",
    overwrite: bool = False,
):
    # Fetch the user's competition list and verigy that supplied value of competition_name is valid
    api = KaggleApi()
    api.authenticate()
    # os.environ['KAGGLE_CONFIG_DIR'] = "/config/.kaggle/"
    api.competition_download_file(competition = competition_name,file_name = filename,path=savepath)
    filename = filename + ".zip"
    with zipfile.ZipFile(Path(savepath) / filename, "r") as zipref:
        zipref.extractall(savepath)

def download_kaggle_dataset(
    *,
    dataset_name: str,
    savepath: Union[str, Path] = ".",
    overwrite: bool = False,
):

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_name, path=savepath, unzip=True)


def submit_to_competition(
    *, competition_name: str, submission_path: Union[str, Path], message: str
):
    api = KaggleApi()
    api.authenticate()
    api.competition_submit(
        file_name=submission_path,
        competition=competition_name,
        message=str,
    )
