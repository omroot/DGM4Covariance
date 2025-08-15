import os
from pathlib import Path
import datetime

from dotenv import load_dotenv

try:
    load_dotenv()
except Exception as e:
    print(f"Error loading .env file : {e}")

# Env setup configs

DEBUG = str(os.getenv("DEBUG")).lower() in ['true']

DEBUG_ROOT_DIR = Path(os.getenv("DEBUG_ROOT_DIR"))
PROD_ROOT_DIR = Path(os.getenv("PROD_ROOT_DIR"))

if DEBUG:
    ROOT_DIR = DEBUG_ROOT_DIR
else:
    ROOT_DIR = PROD_ROOT_DIR



