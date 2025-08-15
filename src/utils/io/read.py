from pathlib import Path
import glob
import pickle
import pandas as pd



class RawDataReader():
    def __init__(self, raw_data_directory: Path):
        self.raw_data_directory = raw_data_directory

    def _read(self, fname:str)->pd.DataFrame:
        return pd.read_csv(fname)

    def read_30_equities(self) -> pd.ExcelFile:
        file_name = str(self.raw_data_directory  ) + "/dataset.csv"
        return  self._read(file_name)
