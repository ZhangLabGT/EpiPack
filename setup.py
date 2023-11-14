from setuptools import setup

from pathlib import Path

ROOT_DIR = Path(__file__).parent
README = (ROOT_DIR / "README.md").read_text()

VERSION = {}
with open(ROOT_DIR / "epipackpy/_version.py") as fp:
    exec(fp.read(), VERSION)

setup(
    name="epipackpy",
    description='EpiPack: scATAC-seq integration, reference mapping and cell type annotation',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/ZhangLabGT/EpiPack', 
    author='Yuqi Cheng',
    author_email='ycheng430@gatech.edu',
    license='MIT',
    version=VERSION['__version__'],

    packages=[
        "epipackpy",
        "epipackpy.data",
        "epipackpy.model"
    ],

    zip_safe=False,
    python_requires=">=3.8, <3.12",
    install_requires=[
        "numpy>=1.17.0",
        "pandas>=1.0, <2.1.2",
        "scipy>=1.4, <2.0.0",
        "scikit-learn>=0.23, <2.0.0",
        "tqdm>=4.62",
        "typing_extensions",
        "torch>=2.0",
        "scanpy>=1.9",
        "polars>=0.19",
        "pyarrow>=14.0",
        "pybiomart>=0.2.0",
        "pyranges",
        "seaborn>=0.12",
    ],
    #extras_require={
    #    'recommend': ['scanpy>=1.9'],
    #    'all': ['epipackpy[recommend]']
    #}
)
