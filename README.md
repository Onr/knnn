# knnn
K-Nearest Neighbors of Neighbors
```bash 
pip install knnn
```

# Description
This package provides a simple implementation of the K-Nearest Neighbors of Neighbors algorithm. The algorithm is a simple extension of the K-Nearest Neighbors algorithm, which is used for anomaly detection. The algorithm is based on the idea that the neighbors of the neighbors of a point gives more information than its neighbors. The algorithm can be used to improve the accuracy of the KNN algorithm.


# Usage
``` python
from knnn import KNNN
import numpy as np

# Random data
x_normal = np.random.rand(100, 2)
x_test = np.random.rand(20, 2) + 1

# Create a KNNN object
knnn = KNNN(num_neighbors=3, num_neighbors_of_neighbors=25)
# Fit the model
knnn.fit(x_normal)
# Predict the labels of the test data
y_pred = knnn.predict(x_test)

```


# Installation
The simplest way to install the package is to run:
```bash 
pip install knnn
```
If you want to install the latest version from the master branch: 

(-e option will allow you to change the code without reinstalling the package)
```bash
git clone https:\\github.com\knnn
cd knnn
python3 -m pip install -e . 
```
If you want to build the package from source, run:
```bash
python3 -m build
``` 
and to install the built package, run:
```bash
python3 -m pip install --force-reinstall dist/*.whl
```
To run the tests, run:
```bash
pytest
```