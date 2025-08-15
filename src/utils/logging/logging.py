import logging
import os
import datetime as dt
from Hermes.utils.base import Singleton
from Hermes.settings import Settings



class Logger(metaclass=Singleton):

    def __init__(self):
        self._loggers = {
            Settings.loggers.DAILY: None,
            Settings.loggers.BACKFILL: None,

        }
    def init_logger(self, logger_name):
        log = logging.getLogger(logger_name)
        log_formatter = logging.Formatter(

            "%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"

        )
        log_directory = Settings.LOGS_DIR / (
            f"{str.lower(dt.date.today().strftime('%b'))}_"
            f"{dt.date.today().strftime('%Y')}"
        )
        log_file_name = log_directory / f"{logger_name}.log"
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        file_handler = logging.FileHandler(log_file_name, mode="a+",)
        file_handler.setFormatter(log_formatter)
        console_handler = logging.StreamHandler()
        log.addHandler(file_handler)
        log.addHandler(console_handler)
        log.setLevel(logging.DEBUG)
        self._loggers[logger_name] = log
    def get_logger(self, logger_name):
        if self._loggers.get(logger_name) is None:
            self.init_logger(logger_name)
        return self._loggers.get(logger_name)