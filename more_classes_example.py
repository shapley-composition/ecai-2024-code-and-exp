{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d582af5e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn import datasets, svm\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.preprocessing import scale\n",
    "import numpy as np\n",
    "from math import sqrt\n",
    "from shapleycomposition import ShapleyExplainer, fig_3D_ilr_space\n",
    "from composition_stats import ilr, sbp_basis\n",
    "from bifurc_tree import create_tree_from_sbp, init_graph, build_graph\n",
    "import plotly.graph_objects as go"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "66e6bf58",
   "metadata": {
    "lines_to_next_cell": 2
   },
   "outputs": [],
   "source": [
    "K = 10         #Index of the instance you want to test in the test set\n",
    "N_class = 10    #Number of class, the dimension of the simplex is therefore N_class-1\n",
    "N_feat  = 6    #In this example, since the number of feature of the digit dataset is quite large (64), we propose to reduce it with a PCA\n",
    "\n",
    "#load the dataset, take a subset of N_class classes, scale it and split into a training and testing set\n",
    "X, Y = datasets.load_digits(return_X_y=True)\n",
    "X = scale(X)\n",
    "X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)\n",
    "\n",
    "#Reduce the number of feature to N_feat with a PCA\n",
    "pca = PCA(n_components=N_feat)\n",
    "X_train = pca.fit_transform(X_train)\n",
    "X_test  = pca.transform(X_test)\n",
    "\n",
    "\n",
    "#train an SVM classifier\n",
    "svc_linear = svm.SVC(kernel='rbf', probability=True)\n",
    "svc_linear.fit(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8259236c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Choose a sequential binary partition matrix and plot the corresponding bifurcation tree\n",
    "sbpmatrix=np.array([[1,1,-1],[1,-1,0]])\n",
    "root = create_tree_from_sbp(sbpmatrix, 3)\n",
    "\n",
    "#Visualise the tree using graphviz\n",
    "graph = init_graph()\n",
    "build_graph(root, graph)\n",
    "graph       #plot the bifurcation tree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "81d6ff89",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# explain all the predictions in the test set\n",
    "explainer = ShapleyExplainer(svc_linear.predict_proba, X_train, N_class)\n",
    "(shapley, base) = explainer.explain_instance(np.array(X_test[K]))\n",
    "\n",
    "print(\"True label of the tested instance: \", end=\"\")\n",
    "print(Y_test[K])\n",
    "\n",
    "#get the model prediction for the Kth instance of the test partition\n",
    "pred = svc_linear.predict_proba(X_test[K].reshape(1,-1))\n",
    "print(\"Prediction on the simplex: \",end=\"\")\n",
    "print(pred)\n",
    "ilr_pred = ilr(pred)\n",
    "print(\"Prediction in the ILR space: \",end=\"\")\n",
    "print(ilr_pred)\n",
    "\n",
    "#The sum of the base distribution and the shapley composition in the ILR space is equal to the predicted probability distribution\n",
    "sum_shap_base = np.array(shapley).sum(axis=0)+base\n",
    "print(\"Sum of the shapley composition and the base distribution in the ILR space: \", end=\"\")\n",
    "print(sum_shap_base)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "55faaf3c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "jupytext": {
   "cell_metadata_filter": "-all",
   "main_language": "python",
   "notebook_metadata_filter": "-all"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}