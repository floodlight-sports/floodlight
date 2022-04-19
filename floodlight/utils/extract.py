import shutil


def extract_zip(filepath, target, format='zip'):
    shutil.unpack_archive(filepath, target, format=format)
