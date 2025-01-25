import torch
from torch_geometric.nn.conv import MessagePassing

class GraphConvolution(MessagePassing):
    def __init__(self, emb_dim):
        super(GraphConvolution, self).__init__() # Por defecto la agregación se hace con add

        # Definir los parámetros a aprender en el proceso de MessagePassing, torch.nn.Linear es una definición parecida a la siguiente https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
        self.node_transformer = torch.nn.Linear(emb_dim, emb_dim)
        self.root_embedding = torch.nn.Embedding(1, emb_dim)
        self.edge_transformer = torch.nn.Linear(7, emb_dim)
