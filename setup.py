from setuptools import setup, find_packages

exceptions = [
        '@ file',
        ]

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
      packages=find_packages(),
      install_requires=global_pack
     )
