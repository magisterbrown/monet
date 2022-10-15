from setuptools import setup, find_packages
from itertools import chain

import glob
def explore_dir(path: str):
    """Discoveres packages in all directories of the folder
        returns:
            packages: (list[int]) list of found packages
            package_dirs: (dict{str:str}) dirs of the parent packages 
    """
    if not path.endswith('/'):
        path += '/'
    folders = glob.glob(f'{path}*')

    packages = list()
    package_dirs = dict()
    for f in folders:
        new_packages = find_packages(where=f)
        for p in new_packages:
            if '.' not in p:
                package_dirs[p] = f'{f}/{p}'
        packages += new_packages

    return packages, package_dirs
print(explore_dir('submodules'))
