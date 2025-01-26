import torch
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.conv import MessagePassing

class GraphConvolution(MessagePassing): 
    #Al heredar de MessagePassing hay que definir los metodos de forward, message y update como mínimo

    def __init__(self, dim_repr_nodo,metodo_agregacion):
        super(GraphConvolution, self).__init__(aggr=metodo_agregacion) # Por defecto la agregación se hace con add (se puede cambiar)

        # Definir las representaciones a aprender en el proceso de MessagePassing
        # Vamos a usar como representación las de cada nodo y las de las aristas
        self.node_transformer = torch.nn.Linear(dim_repr_nodo, dim_repr_nodo)
        self.edge_transformer = torch.nn.Linear(7, dim_repr_nodo)
    
    #  Siguiendo la siguiente guia: https://pytorch-geometric.readthedocs.io/en/2.4.0/tutorial/create_gnn.html
    def forward(self, input, edge_index, edge_attr):  # Realizar el proceso de MessagePassing con la información que nos interesa(representación de nodos y de aristas)
        # edge_index y edge_attr según la documentación:
        # edge_index (list: 2 x #edges): pairs of nodes constituting edges
        # edge_attr (list: #edges x #edge-features): for the aforementioned edges, contains their features

        # Paso 1: Añadir "self-loops" a la matriz de adyacencia
        edge_index, _ = add_self_loops(edge_index, num_nodes=input.size(0))
        
        # Paso 2: Transformaciones lineales
        input = self.node_transformer(input)
        nuevas_aristas = self.edge_transformer(edge_attr)

        # Paso 3: Normalización
        row, col = edge_index
        deg = degree(col, input.size(0), dtype=input.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Pasos 4 y 5: Propagación de mensajes
        return self.propagate(edge_index, x=input, edge_attr=nuevas_aristas, norm=norm)
        # No veo necesario el Paso 6 ya que gestiona automaticamente el bias (no se ha indicado False al usar Linear)
    
    def message(self, x_j, edge_attr, norm):
        return norm.view * (x_j + edge_attr)

    # No hay que definir update porque no vamos a cambiarla (nos sirve la definición actual heredada)
        
    
