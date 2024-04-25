import random
import numpy as np
from math import sqrt, exp
from composition_stats import closure, ilr, ilr_inv, inner, perturb, perturb_inv, power, multiplicative_replacement, sbp_basis
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import ConvexHull


def welford_update(aggregates, count, new_element):
    (mean, M2) = aggregates
    diff = new_element - mean
    mean += diff/(count+1)
    diff2 = new_element - mean
    M2 += np.dot(diff,np.transpose(diff2))
    return (mean, M2)

def class_compositions(N):
    #return N class compositions
    #v[k,:] is the kth class composition meaning a distribution going straight in favor of the class k, with an Aitchison norm of 1
    eps = exp(-sqrt(N/(N-1)))/(1+(N-1)*exp(-sqrt(N/(N-1))))
    v = np.ones((N, N))*eps
    np.fill_diagonal(v,1-(N-1)*eps)
    return v

class ShapleyExplainer():

    """ Compute the Shapley for each feature with the sampling approximation (Algo 2) in "Explaining prediction models and individual predictions
with feature contributions" by Erik Štrumbelj and Igor Kononenko.
    """    
    def __init__(self, model, train_data, n_class, m_min=100, m_max=10000, sbpmatrix=None, names_classes=None, names_features=None):
        self.model       = model
        self.train_data  = train_data
        self.len_tr_data = train_data.shape[0]
        self.n_feat      = train_data.shape[-1]
        self.n_class     = n_class
        self.m_min       = m_min
        self.m_max       = m_max
        self.class_compo = class_compositions(n_class)   #class_compo[k,:] is the kth class composition meaning a distribution going straigth in favor of class k with an unit norm.
        if sbpmatrix is None:
            self.basis = None
            self.sbpmatrix = np.zeros((n_class-1, n_class))
            for i in range(n_class-1):
                self.sbpmatrix[i,[j for j in range(i+1)]] = 1
                self.sbpmatrix[i,i+1] = -1
            print(self.sbpmatrix)
        else:
            #self.basis = np.flip(sbp_basis(sbpmatrix), axis=0)
            self.basis = sbp_basis(sbpmatrix)
            self.sbpmatrix = sbpmatrix
        self.base      = ilr(model(train_data), basis=self.basis).mean(axis=0)
        self.pred      = None
        self.shapley   = None

        #Names of the classes and features if not precised in the calling of the plotting function
        if names_classes is None:
            self.names_classes = ['class '+str(i+1) for i in range(self.n_class)]
        elif (type(names_classes) is list) or (type(names_classes) is np.ndarray):
            if len(names_classes) != self.n_class:
                raise NameError('The number of strings in the list (or array) of class names must be the number of classes: '+str(self.n_class))
            else:
                self.names_classes = names_classes
        else:
            raise NameError('names_classes must be a list (or array) of '+str(self.n_class)+' strings.')
                
        if names_features is None:
            self.names_features = ['feature n.'+str(i+1) for i in range(self.n_feat)]
        elif (type(names_features) is list) or (type(names_features) is np.ndarray):
            if len(names_features) != self.n_feat:
                raise NameError('The number of strings in the list (or array) of feature names must be the number of features: '+str(self.n_feat))
            else:
                self.names_features = names_features
        else:
            raise NameError('names_features must be a list (or array) of '+str(self.n_feat)+' strings.')
           
    def explain_instance(self, x, adjust_sum=True):
        pi       = list(range(self.n_feat))
        phi      = np.zeros((self.n_feat,self.n_class-1))
        m        = np.zeros(self.n_feat)
        tr_var_phi  = np.zeros(self.n_feat)
        mean_phi = np.zeros((self.n_feat, self.n_class-1))
        M2_phi   = np.zeros((self.n_feat, self.n_class-1, self.n_class-1))
        while m.sum() <= self.m_max:
            if (m < self.m_min).sum() == 0:
                j = np.argmax((tr_var_phi/m) - (tr_var_phi/(m+1)))
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
            phi[j] += current_phi
            (new_mean, new_M2) = welford_update((mean_phi[j], M2_phi[j]), m[j], (current_phi))
            mean_phi[j] = new_mean
            M2_phi[j]   = new_M2
            m[j] += 1
            tr_var_phi[j]  = np.trace(new_M2/m[j])

        for i in range(self.n_feat):
            phi[i] = phi[i]/m[i]

        self.pred = ilr(self.model(x.reshape(1,-1)), basis=self.basis)
        if adjust_sum:
            #adjust the sum of shapley compositions as in https://github.com/shap/shap/blob/master/shap/explainers/_sampling.py (last visit November 2023)
            v = 1e6 *tr_var_phi/tr_var_phi.max()
            sum_error = self.pred - phi.sum(axis=0) - self.base
            adj = np.ones((self.n_feat,self.n_class))/self.n_class
            for i in range(self.n_feat):
                phi[i] += sum_error*(v[i]-(v[i]*v.sum())/(1+v.sum()))
        self.shapley = phi
        return (phi, self.base)

    def summarize(self):
        #Compute and return the (unsorted) norm of each Shapley compositions, the cosine between the Shapley compositions and the class vectors and the cosine between each Shapley compositions

        if self.shapley is None:
            raise NameError('You need to run explain_instance() first.')
        
        norm_shapley = np.linalg.norm(self.shapley, axis=1)
        print("List of the features sorted by their Shapley strength (norm of their Shapley composition):")
        ord = np.argsort(-norm_shapley)
        for i in ord:
            print("\t "+self.names_features[i]+": ", end="")
            print(round(norm_shapley[i], 7))
        print()

        class_vect  = ilr(self.class_compo).reshape((self.n_class, self.n_class-1))
        proj_shap_class = np.zeros((self.n_class,self.n_feat))
        cos_shap_shap = np.zeros((self.n_feat,self.n_feat)) 
        print("Projection of the Shapley compositions on the class vectors:")
        print('\t\t',end='')
        for i in range(self.n_feat):
            print('{:10s}'.format(self.names_features[i]),end='\t')
        print()
        for i in range(self.n_class):
            print('{:10s}'.format(self.names_classes[i]+":"), end='\t')
            for j in range(self.n_feat):
                proj_shap_class[i,j] = np.dot(class_vect[i,:],self.shapley[j,:])
                print('{:3.7f}'.format(round(proj_shap_class[i,j], 7)),end='\t')
            print()
        print()

        #Print the cosine between each Shapley compositions
        print("Cosine between each Shapley compositions:")
        print('\t\t',end='')
        for i in range(self.n_feat):
            print('{:10s}'.format(self.names_features[i]),end='\t')
        print()
        for i in range(self.n_feat):
            print('{:10s}'.format(self.names_features[i]+':'), end='\t')
            for j in range(self.n_feat):
                cos_shap_shap[i,j] = np.dot(self.shapley[i,:],self.shapley[j,:])/(norm_shapley[i]*norm_shapley[j])
                print('{:3.7f}'.format(round(cos_shap_shap[i,j], 7)),end='\t')
            print()
        
        return (norm_shapley, proj_shap_class, cos_shap_shap)
    
        
    def plot_ilr_space(self, balances=[1, 2], shapley_sum=False, lim=5, figsize=500, sort=False):
        #return a plotly figure of a 2D or 3D ILR (sub)space corresponding to the chosen ILR components listed in balances.
        #plot range [-lim, lim]
        #If shapley_sum is True, the sum of the shapley vectors are summed from the base distribution to the prediction

        if sort:
            norm_shapley = np.linalg.norm(self.shapley, axis=1)
            ord = np.argsort(-norm_shapley)
        else:
            ord = [i for i in range(self.n_feat)]
        
        if len(balances) > self.n_class-1:
            raise NameError('The number of ILR component to plot must be 2 or 3. If you are in a 3 class problem the number of ILR to plot cannot be above 2.')
        
        if self.shapley is None:
            raise NameError('You need to run explain_instance() first.')
        
        if len(balances) != 2 and len(balances) != 3:
            raise NameError('You need to choose 2 or 3 ILR components to visualize. They should be listed in the list balanced')
        
        fig = go.Figure(layout=go.Layout(autosize=False, width=figsize, height=figsize))
        if len(balances) == 2:
            fig.update_xaxes(range=[-lim, lim])
            fig.update_yaxes(range=[-lim, lim])
            fig.update_layout(xaxis_title="ILR"+str(balances[0]), yaxis_title="ILR"+str(balances[1]), colorway=px.colors.qualitative.D3, legend=dict(bgcolor='rgba(255,255,255,0.4)',yanchor="top", y=0.99, xanchor="right", x=0.99), margin=dict(l=0, r=0, t=0, b=0))
        else:
            fig.update_layout(legend=dict(bgcolor='rgba(255,255,255,0.4)', yanchor="top", y=0.8, xanchor="right", x=0.9),
                              margin=dict(l=0, r=0, t=0, b=0),  colorway=px.colors.qualitative.D3,
                              scene = dict(xaxis = dict(title="ILR"+str(balances[0]), range=[-lim,lim]),
                                           yaxis = dict(title="ILR"+str(balances[1]), range=[-lim,lim]),
                                           zaxis = dict(title="ILR"+str(balances[2]), range=[-lim,lim])))
        
         #Draw the class vectors, meaning the vectors going straight in favor of one class with a norm 1.
        #(Only those with non zero projection)
        class_vect  = ilr(self.class_compo, basis=self.basis)
        
        if len(balances) == 2:
            for i in np.unique(np.concatenate((np.argwhere(self.sbpmatrix[balances[0]-1,:]!=0), np.argwhere(self.sbpmatrix[balances[1]-1,:]!=0))).squeeze()):
                fig.add_trace(go.Scatter(x=[0,class_vect[i,balances[0]-1]], y=[0,class_vect[i,balances[1]-1]], mode='lines', line={'dash': 'dot'}, name=self.names_classes[i], legendgroup='class', legendgrouptitle_text='Class composition:'))
                #fig.add_trace(go.Scatter(x=[0,class_vect[i,balances[0]-1]], y=[0,class_vect[i,balances[1]-1]], mode='lines', line={'dash': 'dot'}, name=self.names_classes[i], legendgroup='class'+str(i)))

        else:
            #for i in range(self.n_class):
            for i in np.unique(np.concatenate((np.argwhere(self.sbpmatrix[balances[0]-1,:]!=0), np.argwhere(self.sbpmatrix[balances[1]-1,:]!=0), np.argwhere(self.sbpmatrix[balances[2]-1,:]!=0))).squeeze()):
                fig.add_trace(go.Scatter3d(x=[0,class_vect[i,balances[0]-1]], y=[0,class_vect[i,balances[1]-1]],z=[0,class_vect[i,balances[2]-1]], mode='lines', line={'dash': 'dash', 'width' : 5}, name=self.names_classes[i], legendgroup='class', legendgrouptitle_text='Class composition:'))

                
        #PLOT THE SHAPLEY COMPOSITION IN THE ILR (SUB)SPACE
        #If shapley_sum is True, the sum of the shapley vectors are summed fro the base distribution to the prediction
        if shapley_sum:
            if len(balances) == 2:
                s = self.base.copy()

                for i in ord: #enumerate(self.shapley):
                    p = self.shapley[i,:]
                    fig.add_trace(go.Scatter(x=[s[balances[0]-1],(s+p)[balances[0]-1]], y=[s[balances[1]-1],(s+p)[balances[1]-1]], mode='lines', name=self.names_features[i],
                             legendgroup='shapley', legendgrouptitle_text='Shapley composition:'))
                    s += p

                #Draw base distribution and prediction
                fig.add_trace(go.Scatter(x=[self.base[balances[0]-1]], y=[self.base[balances[1]-1]], mode='markers', name='base'))
                fig.add_trace(go.Scatter(x=[self.pred[balances[0]-1]], y=[self.pred[balances[1]-1]], mode='markers', name='prediction'))
            else:
                s = self.base.copy()
                for i in ord: #,p in enumerate(self.shapley):
                    p = self.shapley[i,:]
                    fig.add_trace(go.Scatter3d(x=[s[balances[0]-1],(s+p)[balances[0]-1]], y=[s[balances[1]-1],(s+p)[balances[1]-1]], z=[s[balances[2]-1],(s+p)[balances[2]-1]], mode='lines', line={'width' : 5}, name=self.names_features[i], legendgroup='shapley', legendgrouptitle_text='Shapley composition:'))
                    s += p
                #Draw base distribution and prediction
                fig.add_trace(go.Scatter3d(x=[self.base[balances[0]-1]], y=[self.base[balances[1]-1]], z=[self.base[balances[2]-1]], mode='markers', marker={'size' : 4}, name='base'))
                fig.add_trace(go.Scatter3d(x=[self.pred[balances[0]-1]], y=[self.pred[balances[1]-1]], z=[self.pred[balances[2]-1]], mode='markers', marker={'size' : 4}, name='prediction'))
            
        else:
            if len(balances) == 2:
                for i in ord:#,s in enumerate(self.shapley):
                    s = self.shapley[i,:]
                    fig.add_trace(go.Scatter(x=[0,s[balances[0]-1]],y=[0,s[balances[1]-1]], mode='lines', name=self.names_features[i],
                             legendgroup='shapley', legendgrouptitle_text='Shapley composition:'))
            else:
                for i in ord:#,s in enumerate(self.shapley):
                    s = self.shapley[i,:]
                    fig.add_trace(go.Scatter3d(x=[0,s[balances[0]-1]],y=[0,s[balances[1]-1]], z=[0,s[balances[2]-1]], mode='lines', line={'width' : 5}, name=self.names_features[i],
                             legendgroup='shapley', legendgrouptitle_text='Shapley composition:'))


        #Draw maximum probability region boundaries if this is a 3 or a 4 class problem and the number of ILR components to visualize are respectively 2 and 3
                
        if len(balances) == 2 and self.n_class == 3:
            fig.add_trace(go.Scatter(x=[0,-10*lim*class_vect[0,balances[0]-1]], y=[0,-10*lim*class_vect[0,balances[1]-1]], mode='lines', line={ 'color': 'black', 'dash': 'dot'}, opacity=0.4, name='Max. proba.<br>region boundaries'))
            fig.add_trace(go.Scatter(x=[0,-10*lim*class_vect[1,balances[0]-1]], y=[0,-10*lim*class_vect[1,balances[1]-1]], mode='lines', line={ 'color': 'black', 'dash': 'dot'}, opacity=0.4, showlegend=False, name='Max. proba.<br>region boundaries'))
            fig.add_trace(go.Scatter(x=[0,-10*lim*class_vect[2,balances[0]-1]], y=[0,-10*lim*class_vect[2,balances[1]-1]], mode='lines', line={ 'color': 'black', 'dash': 'dot'}, opacity=0.4, showlegend=False, name='Max. proba.<br>region boundaries'))
            
        if len(balances) == 3 and self.n_class == 4:
            for i in range(self.n_class):    
                x = -10*lim*class_vect[:,[k-1 for k in balances]]
                x[i,:] = 0
                xc = x[ConvexHull(x).vertices]
                fig.add_trace(go.Mesh3d(x=xc[:, 0], 
                                y=xc[:, 1], 
                                z=xc[:, 2], 
                                color="black", 
                                opacity=.15,
                                alphahull=0))

        fig.show()

        return fig

    def shapley_histogram(self, sort=False, figwidth=500, figheight=400, fontsize=14,
                          stacked=False, horizontal=False):
        barmode = 'stack' if stacked else 'group'
        fig = go.Figure(layout=go.Layout(autosize=False, width=figwidth, height=figheight))
        fig.update_layout(barmode=barmode, font=dict(size=fontsize), colorway=px.colors.qualitative.D3, legend=dict(orientation='h', font=dict(size=12), bgcolor='rgba(255,255,255,0.4)',yanchor="top", y=0.99, xanchor="right", x=1), margin=dict(l=0, r=0, t=0, b=0))
        if sort:
            norm_shapley = np.linalg.norm(self.shapley, axis=1)
            ord = np.argsort(-norm_shapley)
        else:
            ord = [i for i in range(self.n_feat)]
        shapley_ord = self.shapley[ord,:]
        for i in range(self.n_class):
            x = ['base'] + [self.names_features[k] for k in ord]
            y = ilr_inv(self.shapley, basis=self.basis)[:,i]
            o = 'v'
            if horizontal:
                x, y = y, x
                o = 'h'
            fig.add_trace(go.Bar(name=self.names_classes[i], x=x, y=y,
                                 orientation=o))
        x_label = ''
        y_label = ''
        if horizontal:
            x_label, y_label = y_label, x_label
        fig.update_layout(bargroupgap=0, xaxis_title=x_label,
                          yaxis_title=y_label)
        fig.show()
        
        return fig


    def shapley_histogram_from_base_to_pred(self, sort=False, figwidth=500, figheight=400, fontsize=14,
                          stacked=False, horizontal=False):
        barmode = 'stack' if stacked else 'group'
        fig = go.Figure(layout=go.Layout(autosize=False, width=figwidth, height=figheight))
        fig.update_layout(barmode=barmode, font=dict(size=fontsize), colorway=px.colors.qualitative.D3, legend=dict(orientation='h', font=dict(size=12), bgcolor='rgba(255,255,255,0.4)',yanchor="top", y=0.99, xanchor="right", x=1), margin=dict(l=0, r=0, t=0, b=0))
        if sort:
            norm_shapley = np.linalg.norm(self.shapley, axis=1)
            ord = np.argsort(-norm_shapley)
        else:
            ord = [i for i in range(self.n_feat)]
        shapley_ord = self.shapley[ord,:]
        for i in range(self.n_class):
            x = ['base'] + ['⊕'+self.names_features[k] for k in ord]
            y = ilr_inv(np.array([self.base + shapley_ord[0:f].sum(axis=0) for f in range(self.n_feat+1)]), basis=self.basis)[:,i]
            o = 'v'
            if horizontal:
                x, y = y, x
                o = 'h'
            fig.add_trace(go.Bar(name=self.names_classes[i], x=x, y=y,
                                 orientation=o))
        x_label = ''
        y_label = ''
        if horizontal:
            x_label, y_label = y_label, x_label
        fig.update_layout(bargroupgap=0, xaxis_title=x_label,
                          yaxis_title=y_label)
        fig.show()
        
        return fig


    
    def shapley_stacked_hbars(self, *args, **kwargs):
        return self.shapley_histogram(*args, **kwargs, stacked=True,
                                      horizontal=True)
