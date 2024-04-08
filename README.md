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

```
pip install jupyter
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
