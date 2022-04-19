import os
import tempfile
from typing import List

import h5py

from floodlight.utils.extract import extract_zip
from floodlight.utils.loader import down_loader
from settings import EIGD_HOST_URL, ROOT_DIR, DATA_DIR


def load_eigd(file_pos_h5=f'{ROOT_DIR}/tmp/48dcd3_00-06-00.h5') -> List:
    with h5py.File(file_pos_h5) as h5f:
        pos_dict = {pos_set: positions[()] for pos_set, positions in h5f.items()}
    return pos_dict


class Eigd_Iterator:
    def __init__(self):
        self._index = 0

    def __next__(self):
        pass


class Dataset_Eigd:
    def __init__(self, dataset_path='eigd_dataset'):
        data_dir = os.path.isdir(os.path.join(DATA_DIR, dataset_path))
        if not data_dir or bool(os.listdir(data_dir)):
            self._download_and_extract()

    def __iter__(self):
        return Eigd_Iterator(self)

    def get_dataset(self):
        pass

    @staticmethod
    def _download_and_extract():
        tmp = tempfile.NamedTemporaryFile()
        tmp.write(down_loader(EIGD_HOST_URL))
        extract_zip(tmp.name, DATA_DIR)
        tmp.close()
