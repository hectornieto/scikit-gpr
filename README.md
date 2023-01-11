# Scikit-GPR 

## Synopsis

This project builds upon GPTorch and PyTorch to construct a standard Gaussian Process Regression with Radial Basis Kernel likelihood.

## Installation

Download the project to your local system, enter the download directory and then type

`python setup.py install` 

if you want to install Scikit-GPR  and its low-level modules in your Python distribution. 

The following Python libraries will be required:

- scikit-learn
- PyTorch
- GPTorch
- numpy

With `conda`, you can create a complete environment with
```
conda env create -f environment.yml
```

## Code Example

```python
import scikit_gpr.gaussian_process as gptorch
from sklearn.datasets import make_friedman2
X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
gpr = gptorch-GaussianProcessRegressorWithTorch(n_iter=200, normalize_y=True).fit(X, y)
gpr.score(X, y)
print(f"Observed: {y[:2]}\nPredicted: {gpr.predict(X[:2, :], return_std=True)}")
```

You can type
`help(gptorch.GaussianProcessRegressorWithTorch)`
to understand better the inputs needed and the outputs returned


## License
scikit-gpr: A scikit-learn regressor-like wrapper for Gaussian Process Regression using GPTorch

Copyright 2023 Hector Nieto.
    
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
