import shutil
import urllib.request
from typing import AnyStr, Any


def extract_zip(filepath: str, target: str, archive_type: str = "zip") -> None:
    """Extracts the content of an archive to disk.

    Parameters
    ----------
    filepath : str
        Path to file.
    target : str
        Target to extract files to.
    archive_type: "zip", optional
        Type of archive, like zip, rar, gzip, etc.
    """
    shutil.unpack_archive(filepath, target, format=archive_type)


def download_from_url(path: str) -> AnyStr:
    """Downloads file from URL.

    Parameters
    ----------
    path : str
        URL path to download data from

    Returns
    -------
    data: AnyStr
    """
    res = urllib.request.urlopen(path)
    return res.read()


def get_and_convert(dic: dict, key: Any, value_type: type, default: Any = None) -> Any:
    """Performs dictionary get and type conversion simultaneously.

    Parameters
    ----------
    dic: dict
        Dictionary to be queried.
    key: Any
        Key to be looked up.
    value_type: type
        Desired output type the value should be cast into.
    default: Any, optional
        Return value if key is not in dic, defaults to None.

    Returns
    -------
    value: value_type
        Returns the value for key if key is in dic, else default. Tries type conversion
        to `type(value) = value_type`. If type conversion fails, e.g. by trying to force
        something like `float(None)` due to a missing dic entry, value is returned in
        its original data type.
    """
    value = dic.get(key, default)
    try:
        value = value_type(value)
    except TypeError:
        pass
    except ValueError:
        pass

    return value
