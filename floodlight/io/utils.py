import shutil
import urllib.request
from typing import ByteString


def extract_zip(filepath: str, target: str, archive_type: str = "zip") -> None:
    """Extracts the content of an archive to disk.

    Parameters
    ----------
    filepath : path to file
    target : target to extract files to
    archive_type: "zip", optional
    """
    shutil.unpack_archive(filepath, target, format=archive_type)


def download_from_url(path: str) -> ByteString:
    """Downloads file from URL.

    Parameters
    ----------
    path : URL path to download data from

    Returns
    -------
    data: ByteString
    """
    res = urllib.request.urlopen(path)
    return res.read()
