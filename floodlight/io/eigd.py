import shutil
import tempfile
import urllib.request
from typing import List

import h5py

from settings import EIGD_HOST_URL, ROOT_DIR


def load_eigd(file_pos_h5=f'{ROOT_DIR}/tmp/48dcd3_00-06-00.h5') -> List:
    with h5py.File(file_pos_h5) as h5f:
        pos_dict = {pos_set: positions[()] for pos_set, positions in h5f.items()}
    return pos_dict


def _download():
    response = urllib.request.urlopen(EIGD_HOST_URL)
    return response.status, response.read()


def _unpack():
    status, file = _download()

    tmp = tempfile.NamedTemporaryFile()
    tmp.write(file)

    shutil.unpack_archive(tmp.name, f'{ROOT_DIR}/tmp', format='zip')
    tmp.close()
