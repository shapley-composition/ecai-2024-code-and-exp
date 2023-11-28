import random
import numpy as np
import math
from composition_stats import closure, ilr, ilr_inv, inner, perturb, perturb_inv, power, multiplicative_replacement, sbp_basis

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
        print(self.base.shape)
        print("test")

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
        return (phi,self.base)
