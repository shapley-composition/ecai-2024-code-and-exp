import plotly.graph_objects as go
import itertools, math, numpy as np
from math import sqrt
from scipy.spatial import ConvexHull

lim = 2

o  = np.array([0, 0, 0])
v1 = np.array([sqrt(2/3), sqrt(2)/3, 1/3])
v2 = np.array([-sqrt(2/3), sqrt(2)/3, 1/3])
v3 = np.array([0, -4/(3*sqrt(2)), 1/3])
v4 = np.array([0, 0, -1])

v = np.vstack((v1,v2,v3,v4))

fig = go.Figure()
fig.update_layout(
    scene = dict(
        xaxis = dict(range=[-lim,lim]),
                     yaxis = dict(range=[-lim,lim]),
                     zaxis = dict(range=[-lim,lim]),)
    )


for i in range(v.shape[0]):
    
    x = -100*v
    x[i,:] = 0
    xc = x[ConvexHull(x).vertices]


    fig.add_trace(go.Mesh3d(x=xc[:, 0], 
                        y=xc[:, 1], 
                        z=xc[:, 2], 
                        color="blue", 
                        opacity=.2,
                        alphahull=0))

    fig.add_trace(go.Scatter3d(x=[0,v[i,0]], y=[0,v[i,1]],z=[0,v[i,2]], mode='lines', name='class'+str(i+1), line=dict(width=5)))

fig.show()
