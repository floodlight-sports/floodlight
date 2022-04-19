import shutil
import urllib.request
from typing import ByteString


def extract_zip(filepath: str, target: str, format: str = "zip") -> None:
    """

    Parameters
    ----------
    filepath
    target
    format

    Returns
    -------

    """
    shutil.unpack_archive(filepath, target, format=format)


def down_loader(path: str) -> ByteString:
    """
    
    Parameters
    ----------
    path

    Returns
    -------

    """
    res = urllib.request.urlopen(path)
    return res.read()
