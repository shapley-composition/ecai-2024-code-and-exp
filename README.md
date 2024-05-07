Code and simple experiments for the Shapley Composition project: a multidimensional and multiclass extension of the Shapley value using the Aitchison geometry of the simplex for explaining probabilistic predictions of machine learning classifiers.


## Development

It is recommended to create a virtual environment to install all the
dependencies and run the experiments with the same versions that it was
intended during development. One way to create a virtual environment and load
it is with the following commands


```
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

This creates a virtual environment with Python version 3.9, loads the
environment and upgrades the `pip` package to its latest version.

## Requirements:

All the requirements are listed in the `requirements.txt` file. Those can be
automatically installed in the virtual environment with

```
# Ensure that the virtual environment has been loaded
source venv/bin/activate 
pip install -r requirements
```

Do not forget to install the package graphviz
[https://graphviz.org/download/](https://graphviz.org/download/) to properly
use python package graphviz.

## Jupyter Notebook

Use Jupyter Notebook to visualise and edit the experiment files.

You may want to create first a kernel with the current virtual environment

```
python -m ipykernel install --user --name=shapleycomposition
```

then open Jupyter notebook and select the created kernel.

```
jupyter notebook
```

## Run experiments and generate the figures

The figures are automatically generated and saved in the folder `figures` when
running each Jupyter Notebook. It is also possible to run the notebook and
generate the figures from the terminal without the need to open the Juypter
Notebook web interface.

```
jupyter execute 3classes_example.ipynb
```

## About zeros

A composition cannot contain zeros. Dealing with zeros has been problematic in compositional data analysis [[1]](#1).
In this work, all zeros are considered as rounded zeros rather than essential zeros (following a similar argument as the Cromwell's rule [[2]](#2), we assume that no probabilistic prediction should give a zero probability).
The multiplicative replacement strategy is applied here to the probabilistic predictions. Moreover, in order to avoid exploding values due to too small probabilities, the predictions are casted to float16 such that the minimum probability is 6.104e-05.

## References
<a id="1">[1]</a> 
Josep A Martín-Fernández, Carles Barceló-Vidal, and Vera Pawlowsky-Glahn (2003).
Dealing with zeros and missing values in compositional data sets using nonparametric imputation.
In: Mathematical Geology 35.3, pp. 253–278.

<a id="2">[2]</a> 
Dennis V. Lindley (2006).
Understanding Uncertainty.
Wiley-Interscience.