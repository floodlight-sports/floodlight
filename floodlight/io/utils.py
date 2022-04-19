import shutil
import urllib


def extract_zip(filepath, target, format='zip'):
    shutil.unpack_archive(filepath, target, format=format)


def down_loader(path):
    res = urllib.request.urlopen(path)
    return res.read()
