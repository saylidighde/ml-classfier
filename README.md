
## Introduction

This repository contains implementation of the scikit-learn library's linear SVM function.

## How to Run

Setup python3 on your system

Install the requirements and package using:

```
pip install -r requirements.txt
python setup.py install
```

To run the Jupyter notebooks, you need to first install a local copy of Jupyter and ipykernel in the virtualenv:

```
python -m ipykernel install --user
```

Then the Jupyter notebook can be started as usual:

```
jupyter notebook
```

### Implementation Note

In all of the algorithms in this repository, unless explicitly stated otherwise, the convention for binary classification labels is -1/+1, as opposed to 0/1.
