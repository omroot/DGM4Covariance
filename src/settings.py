from pathlib import Path
import src.config as cfg


ROOT_DIR = cfg.ROOT_DIR

class Settings:
    LOGS_DIR = ROOT_DIR / "logs"

    