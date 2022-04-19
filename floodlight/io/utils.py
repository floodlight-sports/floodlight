import shutil
import urllib.request
from typing import AnyStr


def extract_zip(filepath: str, target: str, archive_type: str = "zip") -> None:
    """Extracts the content of an archive to disk.

    Parameters
    ----------
    filepath : str
        path to file
    target : str
        target to extract files to
    archive_type: "zip", optional
        type of archive, like zip, rar, gzip, etc.
    """
    shutil.unpack_archive(filepath, target, format=archive_type)


def download_from_url(path: str) -> AnyStr:
    """Downloads file from URL.

    Parameters
    ----------
    path : URL path to download data from

    Returns
    -------
    data: AnyStr
    """
    res = urllib.request.urlopen(path)
    return res.read()
