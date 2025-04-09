# Copyright 2022-2024 scuola authors
# SPDX-License-Identifier: Apache-2.0

"""scuola package setup."""

import os

import setuptools
from setuptools import setup

# Read the scuola version
# Cannot import from `scuola.__version__` since that will not be available when building or installing the package
with open(os.path.join(os.path.dirname(__file__), 'scuola', '_version.py')) as f:
    version_globals = {}
    version_locals = {}
    exec(f.read(), version_globals, version_locals)
    scuola_version = version_locals['__version__']

# Use repo README for PyPi description
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

# Hide the content between <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN --> and
# <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END --> tags in the README
while True:
    start_tag = '<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN -->'
    end_tag = '<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END -->'
    start = long_description.find(start_tag)
    end = long_description.find(end_tag)
    if start == -1:
        assert end == -1, 'there should be a balanced number of start and ends'
        break
    else:
        assert end != -1, 'there should be a balanced number of start and ends'
        long_description = long_description[:start] + long_description[end + len(end_tag):]

classifiers = [
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3.13',
]

install_requires = [
    "torch>=2.6,<3",
    "vllm>=0.8.2",
    "transformers>=4.51",
    "accelerate==1.4.0",
    "datasets==3.3.2",
    "deepspeed==0.16.4",
    "wandb==0.19.9",
    "ipykernel==6.29.5",
    "ipywidgets==8.1.5",
    "jupyter==1.1.1",
    "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
]

extra_deps = {}

extra_deps['dev'] = [
    'pre-commit>=2.18.1,<4',
    'pytest==8.3.4',
    'pytest_codeblocks==0.17.0',
    'pytest-cov>=4,<7',
    'toml==0.10.2',
    'yamllint==1.35.1',
    'moto>=4.0,<6',
    'fastapi==0.115.6',
    'pydantic==2.10.5',
    'uvicorn==0.34.0',
    'pytest-split==0.10.0',
]

extra_deps['docs'] = [
    'GitPython==3.1.42',
    'docutils==0.17.1',
    'furo==2022.9.29',
    'myst-parser==0.16.1',
    'nbsphinx==0.9.1',
    'pandoc==2.3',
    'pypandoc==1.13',
    'sphinx-argparse==0.4.0',
    'sphinx-copybutton==0.5.2',
    'sphinx==4.4.0',
    'sphinx-tabs==3.4.5',
    'sphinxcontrib.katex==0.9.6',
    'sphinxcontrib-applehelp==1.0.0',
    'sphinxcontrib-devhelp==1.0.0',
    'sphinxcontrib-htmlhelp==2.0.0',
    'sphinxcontrib-qthelp==1.0.0',
    'sphinxcontrib-serializinghtml==1.1.5',
]

extra_deps['all'] = sorted({dep for deps in extra_deps.values() for dep in deps})

package_name = 'scuola'

if package_name != 'scuola':
    print(f'Building scuola as {package_name}')

setup(
    name=package_name,
    version=scuola_version,
    author='Xiaohan Zhang',
    author_email='xiaohanzhang.cmu@gmail.com',
    description=
    'scuola lets students run online RL with LLM as policy in SPMD fashion',
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    package_data={
        'scuola': ['py.typed'],
    },
    packages=setuptools.find_packages(exclude=['tests*']),
    entry_points={
        'console_scripts': ['simulator = simulation.launcher:launch_simulation_ui',],
    },
    classifiers=classifiers,
    install_requires=install_requires,
    extras_require=extra_deps,
    python_requires='>=3.12',
)
