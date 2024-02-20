from setuptools import setup, find_packages

setup(
    name="knnn",
    version="0.0.7",
    author="Ori Nizan",
    author_email="restin3@gmail.com",
    description="An implementation of KNNN algorithm",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/onr/knnn",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7, <=3.10",
    install_requires=[
        'numpy',
        'scipy',
        'tqdm',
        'faiss-cpu',
    ],
    extras_require={
        'tests': [
            'pytest',
            'datasets',
            'pandas',
            'faiss-gpu',
        ],
        'faiss-gpu': ['faiss-gpu'],
    },
    keywords=['knnn', 'knn', 'embedding'],
    project_urls={
        "Homepage": "https://github.com/onr/knnn",
        "Bug Tracker": "https://github.com/onr/knnn/issues",
    },
)
