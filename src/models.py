import torch
from layers import GraphConvolution
from torch.nn import Linear
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

class GCN(torch.nn.Module):
    def __init__(self, num_clases, num_layers, dim_repr_nodo, metodo_agregacion):
        super(GCN, self).__init__()
        # Crear una lista de capas GraphConvolution
        self.capas = torch.nn.ModuleList
        for layer in range(num_layers):
            self.capas.append(GraphConvolution(dim_repr_nodo, metodo_agregacion))
        # A continuación definimos la capa de salida que será alimentada con el pooling del grafo completo y determinará la clasificación por cada instancia
        self.perceptron = torch.nn.Linear(dim_repr_nodo, num_clases)
