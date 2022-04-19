import os
import tempfile
from typing import Text

import h5py

from floodlight.io.utils.extract import extract_zip
from floodlight.io.utils.loader import down_loader
from settings import EIGD_HOST_URL, DATA_DIR


class Eigd_Iterator:
    def __init__(self):
        self._index = 0

    def __next__(self):
        pass


class Eigd:
    def __init__(self, dataset_path='eigd_dataset'):
        self._data_dir: Text = os.path.join(DATA_DIR, dataset_path)

        if not os.path.isdir(self._data_dir) or bool(os.listdir(self._data_dir)):
            self._download_and_extract()

    def __iter__(self):
        return Eigd_Iterator(self)

    def get_dataset(self, file_pos_h5):
        file_path = os.path.join(self._data_dir, file_pos_h5)

        with h5py.File(file_path) as h5f:
            pos_dict = {pos_set: positions[()] for pos_set, positions in h5f.items()}
        return pos_dict

    @staticmethod
    def _download_and_extract():
        tmp = tempfile.NamedTemporaryFile()
        tmp.write(down_loader(EIGD_HOST_URL))
        extract_zip(tmp.name, DATA_DIR)
        tmp.close()
