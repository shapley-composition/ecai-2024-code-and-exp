import plotly.graph_objects as go
import itertools, math, numpy as np
from math import sqrt
from scipy.spatial import ConvexHull

# This simply creates a set of points:

x = np.array([[0,0,-10], [0, 0, 10],[0, 10, 10] , [0, 10, -10], [10, -10/sqrt(3), -10], [10, -10/sqrt(3), 10], [10, 10, 10], [10, 10, -10]])
# Then I compute the convex hull using scipy:
xc = x[ConvexHull(x).vertices]

fig = go.Figure()

fig.add_trace(go.Mesh3d(x=xc[:, 0], 
                        y=xc[:, 1], 
                        z=xc[:, 2], 
                        color="blue", 
                        opacity=.5,
                        alphahull=0))


x = np.array([[0,0,0], [0,0,10], [0, 10, 10], [0, 10, 10/sqrt(2)], [-10, -10/sqrt(3), -10/sqrt(6)], [-10, -10/sqrt(3), 10], [-10, 10, 10], [-10, 10, 10/sqrt(2)]])
# x = np.array([[0,0,0], [0,0,10], [0, 10, 10], [0, 10, 10/sqrt(2)], [-10, -10/sqrt(3), -10/sqrt(6)], [-10, -10/sqrt(3), 10], [-10, 10, 10], [-10, 10, 10/sqrt(2)]])
# Then I compute the convex hull using scipy:
xc = x[ConvexHull(x).vertices]

fig.add_trace(go.Mesh3d(x=xc[:, 0], 
                        y=xc[:, 1], 
                        z=xc[:, 2], 
                        color="orange", 
                        opacity=.5,
                        alphahull=0))


x = np.array([[-10, -10/sqrt(3), -10/sqrt(6)], [-10, -10/sqrt(3), 10], [-10, -10, 10], [-10, -10, -10/sqrt(2)], [0,0,0], [0,0,10], [0,-10, -10/sqrt(2)], [10,-10/sqrt(3),10], [10,-10/sqrt(3),-10/sqrt(6)], [10, -10, -10/sqrt(2)], [10, -10, 10]])
# Then I compute the convex hull using scipy:
xc = x[ConvexHull(x).vertices]

fig.add_trace(go.Mesh3d(x=xc[:, 0], 
                        y=xc[:, 1], 
                        z=xc[:, 2], 
                        color="green", 
                        opacity=.5,
                        alphahull=0))


x = np.array([[10, -10, -10/sqrt(2)], [10, -10, -10] , [-10, -10, -10/sqrt(2)], [-10, -10, -10], [10, 10, -10], [10, 10, 10/sqrt(2)], [-10, 10, -10], [-10, 10, 10/sqrt(2)]])
# Then I compute the convex hull using scipy:
xc = x[ConvexHull(x).vertices]

fig.add_trace(go.Mesh3d(x=xc[:, 0], 
                        y=xc[:, 1], 
                        z=xc[:, 2], 
                        color="pink", 
                        opacity=.5,
                        alphahull=0))



fig.show()
