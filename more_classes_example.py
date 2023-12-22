# %%

from sklearn.model_selection import train_test_split
from sklearn import datasets, svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from math import sqrt
from shapleycomposition import ShapleyExplainer
from composition_stats import ilr, sbp_basis
from bifurc_tree import create_tree_from_sbp, init_graph, build_graph, sbp_from_aggloclustchildren
import plotly.graph_objects as go
from mahalanobis_matrix import mahalanobis_matrix

# %%
K = 10         #Index of the instance you want to test in the test set
N_class = 10   #Number of class, the dimension of the simplex is therefore N_class-1
N_feat  = 6    #In this example, since the number of feature of the digit dataset is quite large (64), we propose to reduce it with a PCA

#load the dataset, take a subset of N_class classes, scale it and split into a training and testing set
X, Y = datasets.load_digits(return_X_y=True)
subset_i = np.where(Y < N_class)
X = X[subset_i]
Y = Y[subset_i]
X = scale(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Reduce the number of feature to N_feat with a PCA
pca = PCA(n_components=N_feat)
X_train = pca.fit_transform(X_train)
X_test  = pca.transform(X_test)


#train an SVM classifier
svc_linear = svm.SVC(kernel='rbf', probability=True)
svc_linear.fit(X_train, Y_train)

# %%

#Choose a sequential binary partition matrix and plot the corresponding bifurcation tree
# sbpmatrix = np.array([[1,1,-1,-1,1,1,1],[1,-1,0,0,-1,-1,-1],[0,1,0,0,-1,-1,-1],[0,0,0,0,1,-1,-1],[0,0,0,0,0,1,-1],[0,0,1,-1,0,0,0]])
#sbpmatrix = np.array([[1,-1,1,-1,1,-1,1],[1,0,1,0,-1,0,-1],[0,0,0,0,1,0,-1],[1,0,-1,0,1,0,0],[0,1,0,1,0,-1,0],[0,1,0,-1,0,0,0]])

#Get a sequential binary parition from the agglomeration of classes from a distance matrix between classes. Distance matrix is here made of the mahalanobis distance between a pair of classes in the ILR space output by the classifier (assuming normal distribution with same covariance matrix between a pair of classes).
M = mahalanobis_matrix(svc_linear.predict_proba, X_test, Y_test)
sbpmatrix = sbp_from_aggloclustchildren(AgglomerativeClustering(affinity='precomputed', linkage='average').fit(M).children_)

basis = np.flip(sbp_basis(sbpmatrix), axis=0)
root = create_tree_from_sbp(sbpmatrix, N_class)

##Visualise the tree using graphviz
graph = init_graph()
build_graph(root, graph)
#graph.render('tree.pdf', view=True)
graph       #plot the bifurcation tree

# %%

# Explain all the predictions in the test set
explainer = ShapleyExplainer(svc_linear.predict_proba, X_train, N_class, sbpmatrix=sbpmatrix,
                            names_classes=[str(i) for i in range(N_class)])
(shapley, base) = explainer.explain_instance(np.array(X_test[K]))

print("True label of the tested instance: ", end="")
print(Y_test[K])

#get the model prediction for the Kth instance of the test partition
pred = svc_linear.predict_proba(X_test[K].reshape(1,-1))
print("Prediction on the simplex: ",end="")
print(pred)
ilr_pred = ilr(pred, basis=None)
print("Prediction in the ILR space: ",end="")
print(ilr_pred)

#The sum of the base distribution and the shapley composition in the ILR space is equal to the predicted probability distribution
sum_shap_base = np.array(shapley).sum(axis=0)+base
print("Sum of the shapley composition and the base distribution in the ILR space: ", end="")
print(sum_shap_base)

# %%
#SUMMARIZE WITH NORM, COSINE AND INNER PRODUCTS

(norm_shapley, proj_shap_class, cos_shap_shap) = explainer.summarize()

# %%
#PLOT the a ILR subspace (corresponding to the chosen ILR components listed in balances).
#plot range [-lim, lim]
#If shapley_sum is True, the sum of the shapley vectors are summed from the base distribution to the prediction

fig = explainer.plot_ilr_space(balances=[3, 5], lim=6, figsize=750)
fig = explainer.plot_ilr_space(balances=[3, 5], shapley_sum=True, lim=6, figsize=750)

# %%
#PLOT the a ILR subspace (corresponding to the chosen ILR components listed in balances).
#plot range [-lim, lim]
#If shapley_sum is True, the sum of the shapley vectors are summed from the base distribution to the prediction

fig = explainer.plot_ilr_space(balances=[1, 2, 4], lim=3)
fig = explainer.plot_ilr_space(balances=[1, 2, 4], shapley_sum=True, lim=3)

# %%
#Plot the Shapley compositions as histograms
explainer.shapley_histogram(figheight=200)
