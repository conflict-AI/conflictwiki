import collections
import os
import typing as th
from pathlib import Path
import pandas as pd
import toml


import logging
LOGGER = logging.getLogger("__main__")

CONFIG_FILE = 'wb0configs/configs.toml'
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class ConfigBase():


    def __init__(self):
        super().__init__()

        self.paths = {}
        self.base_columns = {}
        self.auxiliary_columns = {}
        self.metadata_keys = {}
        self.attach_config(self.load_configs(CONFIG_FILE))

    def attach_config(self, configs: th.Dict):

        for key, value in configs.items():
            self.__setattr__(key, value)

    def load_configs(self, config_files: str):

        CONFIG_PATH = Path(ROOT_DIR) / config_files
        CONFIG_DICT = toml.load(CONFIG_PATH, _dict=dict)

        return CONFIG_DICT


    def load_dataframe(self, filename: str):

        try:
            with open(filename, 'rb') as data_file:
                return pd.read_csv(data_file)
        except IOError as exc:
            LOGGER.error(f"Failed to load and parse the file. {str(exc)}", exc_info=1)
            raise exc


    def get_path(self, pathkey: str):

        resource_path = Path(DATA_ROOT_DIR) / self.paths[pathkey]
        return resource_path

