"""installer"""

from setuptools import setup, find_packages
setup_dic = dict(
    name="carmodel_calibration",
    description="a tool for calibration car-following models using sumo",
    author="Steve Oswald",
    author_email="stevepeter.oswald@gmail.com",
    packages=find_packages(exclude=["scripting", "tests", 'tests.*', ".*"]),
    version=0.1,
    entry_points={
        'console_scripts': [
            'carmodel-calibration = carmodel_calibration.cmd:main'
        ],
    },
    license="GPL-3.0",
    project_urls={
        'Issues': 'https://github.com/stepeos/pycarmodel_calibration/issues'
    },install_requires=[
        'pandas',
        'scipy',
        'numpy==1.20.0',
        'matplotlib',
        'lxml',
        'pygad==2.19.0',
        'regex',
        'argparse',
        'entrypoints>=0.2.2',
        'SALib==1.4.5',
        'tqdm',
        'cloudpickle==2.2.1'
    ],
    include_package_data=True,
    python_requires="==3.8.*"
)
if __name__ == "__main__":
    setup(**setup_dic)
