import shutil
import urllib.request
from typing import ByteString


def extract_zip(filepath: str, target: str, format: str = 'zip') -> None:
    """

    :param filepath:
    :param target:
    :param format:
    :return:
    """
    shutil.unpack_archive(filepath, target, format=format)


def down_loader(path: str) -> ByteString:
    """

    :param path:
    :return:
    """
    res = urllib.request.urlopen(path)
    return res.read()
