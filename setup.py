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

exceptions = [
        '@ file',
        'matplotlib',
        ]

packages, package_dirs = explore_dir('submodules')
packages += find_packages()

with open('requirements.txt', 'r') as f:
    deps = f.readlines()
    for exc in exceptions:
        deps = list(filter(lambda x:not (exc in x), deps))

    global_pack = deps

setup(name='monart',
      version='1.0',
      description='',
      author='magisterbrownie',
      author_email='magisterbrownie@gmail.com',
      url='',
      packages=packages,
      package_dir = package_dirs,
      install_requires=global_pack
     )
