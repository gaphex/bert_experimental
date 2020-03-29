from setuptools import setup, find_packages

__version__ = '1.0.4'

setup(
    name='bert_experimental',
    version=__version__,
    description='Utilities for finetuning BERT-like models',
    url='https://github.com/gaphex/bert_experimental',
    long_description=open('README.md', 'r', encoding="utf8").read(),
    long_description_content_type='text/markdown',
    author='Denis Antyukhov',
    author_email='gaphex@gmail.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'tensorflow>=1.15, <2.0',
        'tensorflow-hub==0.7.0',
        'numpy'
    ],
    classifiers=(
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ),
    keywords='bert nlp tensorflow machine learning sentence encoding embedding finetuning',
)
