import torch
from torch_geometric.nn.conv import MessagePassing

class GraphConvolution(MessagePassing): 
    #Al heredar de MessagePassing hay que definir los metodos de forward, message y update como mínimo

    def __init__(self, dim_repr_nodo,metodo_agregacion):
        super(GraphConvolution, self).__init__(aggr=metodo_agregacion) # Por defecto la agregación se hace con add (se puede cambiar)

        # Definir los parámetros a aprender en el proceso de MessagePassing, torch.nn.Linear es una definición parecida a la siguiente https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
        # Solo vamos a usar como representación a aprender en el MessagePassing las de cada nodo y las de las aristas
        self.node_transformer = torch.nn.Linear(dim_repr_nodo, dim_repr_nodo)
        self.edge_transformer = torch.nn.Linear(7, dim_repr_nodo)
    
        
    
