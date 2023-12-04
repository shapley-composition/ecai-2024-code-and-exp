import random
import numpy as np
from math import sqrt
from composition_stats import closure, ilr, ilr_inv, inner, perturb, perturb_inv, power, multiplicative_replacement, sbp_basis
import plotly.graph_objects as go
from scipy.spatial import ConvexHull


def welford_update(aggregates, count, new_element):
    (mean, M2) = aggregates
    diff = new_element - mean
    mean += diff/(count+1)
    diff2 = new_element - mean
    M2 += np.dot(diff,np.transpose(diff2))
    return (mean, M2)


class ShapleyExplainer():

    """ Compute the Shapley for each feature with the sampling approximation (Algo 2) in "Explaining prediction models and individual predictions
with feature contributions" by Erik Štrumbelj and Igor Kononenko.
    """    
    def __init__(self, model, train_data, n_class, m_min=100, m_max=10000, sbpmatrix=None):
        self.model      = model
        self.train_data = train_data
        self.len_tr_data = train_data.shape[0]
        self.n_feat     = train_data.shape[-1]
        self.n_class    = n_class
        self.m_min      = m_min
        self.m_max      = m_max
        if sbpmatrix is None:
            self.basis = None
        else:
            self.basis = np.flip(sbp_basis(sbpmatrix), axis=0)
        self.base       = ilr(model(train_data), basis=self.basis).mean(axis=0)
        self.shapley
        
    def explain_instance(self, x, adjust_sum=True):
        pi       = list(range(self.n_feat))
        phi      = np.zeros((self.n_feat,self.n_class-1))
        m        = np.zeros(self.n_feat)
        var_phi  = np.zeros(self.n_feat)
        mean_phi = np.zeros((self.n_feat, self.n_class-1))
        M2_phi   = np.zeros((self.n_feat, self.n_class-1, self.n_class-1))
        while m.sum() <= self.m_max:
            if (m < self.m_min).sum() == 0:
                j = np.argmax((var_phi/m) - (var_phi/(m+1)))
            else:
                j = np.argwhere(m<self.m_min)[0][0]
            x_sampled = self.train_data[random.randint(0,self.len_tr_data-1),:] #Gen a random sample from the training set
            random.shuffle(pi)                                          #Gen a random permutation
            pos_j      = np.argwhere(np.array(pi)==j).item()
            x_perturb1 = np.copy(x)
            x_perturb2 = np.copy(x)
            for k in range(self.n_feat):
                pos_k = np.argwhere(np.array(pi)==k).item()
                if pos_k == pos_j:
                    x_perturb2[k] = x_sampled[k]
                if pos_k > pos_j:
                    x_perturb1[k] = x_sampled[k]
                    x_perturb2[k] = x_sampled[k]
            current_phi = ilr(self.model(x_perturb1.reshape(1,-1)), basis=self.basis) - ilr(self.model(x_perturb2.reshape(1,-1)), basis=self.basis)
            phi[j] += + current_phi
            (new_mean, new_M2) = welford_update((mean_phi[j], M2_phi[j]), m[j], (current_phi))
            mean_phi[j] = new_mean
            M2_phi[j]   = new_M2
            m[j] += 1
            var_phi[j]  = np.trace(new_M2/m[j])
        for i in range(self.n_feat):
            phi[i] = phi[i]/m[i]
        if adjust_sum:
            #adjust the sum of shapley compositions as in https://github.com/shap/shap/blob/master/shap/explainers/_sampling.py (last visit November 2023)
            v = 1e6 *var_phi/var_phi.max()
            sum_error = ilr(self.model(x.reshape(1,-1)), basis=self.basis) - phi.sum(axis=0) - self.base
            adj = np.ones((self.n_feat,self.n_class))/self.n_class
            for i in range(self.n_feat):
                phi[i] += sum_error*(v[i]-(v[i]*v.sum())/(1+v.sum()))
            self.shapley = phi
        return (phi, self.base)

    def summarize():
        

def fig_2D_ilr_space(lim=5, figsize=500, names_classes=None):
    #CREATE A PLOTLY GRAPH_OBJECTS FIGURE OF THE 2D ILR SPACE (with gram-schmidt basis)
    #plot range [-lim, lim]
    #names_classes should be a list of 3 strings.
    fig = go.Figure(layout=go.Layout(autosize=False, width=figsize, height=figsize))
    fig.update_xaxes(range=[-lim, lim])
    fig.update_yaxes(range=[-lim, lim])
    fig.update_layout(xaxis_title="ILR1", yaxis_title="ILR2", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), font=dict(size=10))
    
    #Draw maximum probability region boundaries
    fig.add_trace(go.Scatter(x=[0,0], y=[0,lim], mode='lines', line={ 'color': 'black', 'dash': 'dot'}, 
                             opacity=0.4, name='Max. proba. region boundaries'))
    fig.add_trace(go.Scatter(x=[0,lim], y=[0,-lim/sqrt(3)], mode='lines', line={ 'color': 'black', 'dash': 'dot'}, 
                         opacity=0.4, showlegend=False, name='Max. proba. region boundaries'))
    fig.add_trace(go.Scatter(x=[0,-lim], y=[0,-lim/sqrt(3)], mode='lines', line={ 'color': 'black', 'dash': 'dot'}, 
                         opacity=0.4, showlegend=False, name='Max. proba. region boundaries'))

    if names_classes is None:
        names_classes = ['class 1','class 2','class 3']
    
    #Draw the class vectors, meaning the vectors going straight in favor of one class with a norm 1.
    fig.add_trace(go.Scatter(x=[0,sqrt(3)/2], y=[0,1/2], mode='lines', line={ 'color': 'blue', 'dash': 'dot'}, name=names_classes[0], legendgroup='class', legendgrouptitle_text='Classes'))
    fig.add_trace(go.Scatter(x=[0,-sqrt(3)/2], y=[0,1/2], mode='lines', line={ 'color': 'red', 'dash': 'dot'}, name=names_classes[1], legendgroup='class', legendgrouptitle_text='Classes'))
    fig.add_trace(go.Scatter(x=[0,0], y=[0,-1], mode='lines', line={ 'color': 'green', 'dash': 'dot'}, name=names_classes[2], legendgroup='class', legendgrouptitle_text='Classes'))

    return fig


def fig_3D_ilr_space(lim=5, figsize=500, names_classes=None):
    #CREATE A PLOTLY GRAPH_OBJECTS FIGURE OF THE 3D ILR SPACE (with gram-schmidt basis)
    #plot range [-lim, lim]
    #names_classes should be a list of 3 strings.

    v = np.vstack(( [sqrt(2/3), sqrt(2)/3, 1/3], [-sqrt(2/3), sqrt(2)/3, 1/3], [0, -4/(3*sqrt(2)), 1/3], [0, 0, -1]))      #class vectors, meaning the vectors going straight in favor of one class with a norm 1.

    fig = go.Figure(layout=go.Layout(autosize=False, width=figsize, height=figsize))
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                      font=dict(size=10),
                      scene = dict(xaxis = dict(title="ilr1", range=[-lim,lim]),
                                   yaxis = dict(title="ilr2", range=[-lim,lim]),
                                   zaxis = dict(title="ilr3", range=[-lim,lim])))

    if names_classes is None:
        names_classes = ['class 1','class 2','class 3', 'class 4']
    
    for i in range(v.shape[0]):    
        x = -10*lim*v
        x[i,:] = 0
        xc = x[ConvexHull(x).vertices]
        fig.add_trace(go.Mesh3d(x=xc[:, 0], 
                                y=xc[:, 1], 
                                z=xc[:, 2], 
                                color="black", 
                                opacity=.15,
                                alphahull=0))
        
    fig.add_trace(go.Scatter3d(x=[0,v[0,0]], y=[0,v[0,1]],z=[0,v[0,2]], mode='lines', line={ 'color': 'blue', 'dash': 'dash', 'width' : 5}, name=names_classes[0], legendgroup='class', legendgrouptitle_text='Classes')) 
    fig.add_trace(go.Scatter3d(x=[0,v[1,0]], y=[0,v[1,1]],z=[0,v[1,2]], mode='lines', line={ 'color': 'red', 'dash': 'dash', 'width' : 5}, name=names_classes[1], legendgroup='class', legendgrouptitle_text='Classes'))
    fig.add_trace(go.Scatter3d(x=[0,v[2,0]], y=[0,v[2,1]],z=[0,v[2,2]], mode='lines', line={ 'color': 'green', 'dash': 'dash', 'width' : 5}, name=names_classes[2], legendgroup='class', legendgrouptitle_text='Classes'))
    fig.add_trace(go.Scatter3d(x=[0,v[3,0]], y=[0,v[3,1]],z=[0,v[3,2]], mode='lines', line={ 'color': 'orange', 'dash': 'dash', 'width' : 5}, name=names_classes[3], legendgroup='class', legendgrouptitle_text='Classes')) 
    
    return fig


