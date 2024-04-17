import numpy as np
import plotly.graph_objects as go

s = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1]
])

s = np.array(s)
        
fig = go.Figure(layout=go.Layout(autosize=False, width=500, height=500))
fig.update_layout(legend=dict(bgcolor='rgba(255,255,255,0.4)', yanchor="top", y=0.99, xanchor="right", x=1),
                              margin=dict(l=0, r=0, t=0, b=0),
                              scene = dict(xaxis = dict(title="p1", range=[0,1]),
                                           yaxis = dict(title="p2", range=[0,1]),
                                           zaxis = dict(title="p3", range=[0,1])))


i = np.array([0])
j = np.array([1])
k = np.array([2])

fig.add_trace(go.Mesh3d(x=s[:,0], y=s[:,1], z=s[:,2], opacity=0.4, color='blue', i=i, j=j, k=k ))

fig.show()
