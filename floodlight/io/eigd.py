import urllib.request
from settings import EIGD_HOST_URL
import h5py
from typing import List


def load_eigd():
    return ["No", "Yes"]


def _download():
    # try:
    response = urllib.request.urlopen(EIGD_HOST_URL)
    # except Exception as ex:
    #     print(ex)
    # else:
    #     content = response.read()
    #     print(content)
    # finally:
    #     print(content)

    return response.status, response.read()


def _unpack():
    pass
