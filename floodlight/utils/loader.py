import urllib


def down_loader(path):
    res = urllib.request.urlopen(path)
    return res.read()

