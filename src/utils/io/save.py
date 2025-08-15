
from pathlib import Path

import pandas as pd


class RawDataSaver():


    def __init__(self,
                 raw_data_directory: Path
                 ):
        self.raw_data_directory = raw_data_directory
    def _save(self,df: pd.DataFrame, fname: str):
        df.to_csv(fname, index=False)
    def save_example_data(self, df: pd.DataFrame):
        file_name = self.raw_data_directory / "example_data.csv"
        self._save(df, file_name)

