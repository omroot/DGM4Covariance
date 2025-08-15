from pathlib import Path
import src.config as cfg


ROOT_DIR = cfg.ROOT_DIR

class Settings:
    LOGS_DIR = ROOT_DIR / "logs"
    class backfill:
        class paths:
            RAW_DATA_PATH = ROOT_DIR / 'persisted' / 'backfill' / 'raw_data'
    class daily:
        class paths:
            RAW_DATA_PATH = ROOT_DIR / 'persisted' / 'daily' / 'raw_data'
    class loggers:
        DAILY = "daily"
        BACKFILL = "backfill"