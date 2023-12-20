import numpy as np
import graphviz

class Node:
    def __init__(self, typ, data, parts, left=None, right=None):
        self.left  = left
        self.right = right
        self.data  = data
        self.typ   = typ
        self.list_part = parts
    def Print(self):
        print("(",end="")
        if self.left != None and self.right != None:
            print(self.typ+str(self.data)+", ",end="")
            self.left.Print()
            print(", ",end="")
            self.right.Print()
        elif self.left == None and self.right == None:
            print(self.typ+str(self.data),end="")
        else: raise NameError('This is not a proper bifurcation tree')
        print(")",end="")


def create_tree_from_sbp(S, Np, depth=1, p=None):
    #check dim of the matrix
    if S.shape[0] != S.shape[1]-1: raise NameError('This is not a proper sequential binary partition matrix')

    if p is None:
        p = np.linspace(1,Np,Np, dtype=int)
        if S.shape[1] != Np: raise NameError('The sequential binary parition matrix does not match the number of parts in the composition')
    
    colr = np.argwhere(S[0,:]==1).squeeze(axis=1)
    cols = np.argwhere(S[0,:]==-1).squeeze(axis=1)
    r = p[colr]
    s = p[cols]
    if len(colr) == 1 and len(cols) == 1:
        root = Node("ilr", Np-depth, list(r)+list(r), Node("p", r[0], list(r)), Node("p", s[0], list(s)))
    elif len(colr) == 1:
        root = Node("ilr", Np-depth, list(r)+list(r), Node("p", r[0], list(r)), create_tree_from_sbp(S[1:,cols], Np, depth=depth+1, p=s))
    elif len(cols) == 1:
        root = Node("ilr", Np-depth, list(r)+list(r), create_tree_from_sbp(S[1:len(colr),colr], Np, depth=depth+1,p=r), Node("p", s[0], list(s)))
    else:
        if S[1,colr][0] == 0:
            root = Node("ilr", Np-depth, list(r)+list(s), create_tree_from_sbp(S[len(cols):,colr], Np, depth=depth+len(cols),p=r), create_tree_from_sbp(S[1:len(cols),cols], Np, depth=depth+1,p=s))
        else:
            root = Node("ilr", Np-depth, list(r)+list(s), create_tree_from_sbp(S[1:len(colr),colr], Np, depth=depth+1,p=r), create_tree_from_sbp(S[len(colr):,cols], Np, depth=depth+len(colr),p=s))
    return root



def init_graph():
    return graphviz.Digraph(strict=True)

def build_graph(root, graph):
    if root.left is not None:
        if root.left.typ == "ilr":
            graph.attr('node', shape='ellipse')
            c="black"
        if root.left.typ == "p":
            graph.attr('node', shape='square')
            c="blue"
        graph.node(root.left.typ+str(root.left.data), color=c)
        graph.edge(root.typ+str(root.data), root.left.typ+str(root.left.data))
        build_graph(root.left, graph)
    if root.right is not None:
        if root.right.typ == "ilr":
            graph.attr('node', shape='ellipse')
            c="black"
        if root.right.typ == "p":
            graph.attr('node', shape='square')
            c="blue"
        graph.node(root.right.typ+str(root.right.data), color=c)
        graph.edge(root.typ+str(root.data), root.right.typ+str(root.right.data))
        build_graph(root.right, graph)

# S = np.array([[1,1,-1,-1,1,1],[1,-1,0,0,-1,-1],[0,1,0,0,-1,-1],[0,0,0,0,1,-1],[0,0,1,-1,0,0]])
# # #S = np.array([[1,1,1,1,1,-1],[1,1,1,1,-1,0],[1,1,1,-1,0,0],[1,1,-1,0,0,0],[1,-1,0,0,0,0]])

# root = create_tree_from_sbp(S, 6)
# # root.Print()
# # print("")

# tree = init_graph()
# build_graph(root, tree)

# tree.render('tree.pdf', view=True)  

