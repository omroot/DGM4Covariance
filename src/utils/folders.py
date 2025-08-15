import os
import glob
from pathlib import Path

import Plutus.config as cfg

def create_folder(folder_path: str)->None:
    """ Creates a folder if it does not exist """
    if os.path.exists(folder_path):
        print(f"Folder already exist. {folder_path}")
    else:
        os.mkdir(folder_path)
        print(f"Folder created: {folder_path}")


def get_most_recent_created_file(directory_name: str)->str:
    return max(glob.glob(directory_name) , key = os.path.getctime())

def create_persistence_folder_hierarchy():
    pass


