import torch
from layers import GraphConvolution
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

class GCN(torch.nn.Module):
    def __init__(self, num_clases, num_capas, dim_repr_nodo, metodo_agregacion, drop_ratio, graph_pooling): # Por simplificación, de momento las dimensiones de todas las capas ocultas se mantienen iguales
        super(GCN, self).__init__()
        # Definiendo variables para poder utilizarlas en los siguientes metodos 
        self.graph_pooling = graph_pooling
        self.drop_ratio = drop_ratio
        self.num_capas = num_capas
        # Crear una lista de capas GraphConvolution
        self.capas = torch.nn.ModuleList()
        for capa in range(num_capas):
            self.capas.append(GraphConvolution(dim_repr_nodo, metodo_agregacion))
        self.node_encoder = torch.nn.Embedding(1, dim_repr_nodo) # Codificador para transformar las dimensiones de los nodos
        # A continuación definimos la capa de salida que será alimentada con el pooling del grafo completo y determinará la clasificación por cada instancia
        self.perceptron = torch.nn.Linear(dim_repr_nodo, num_clases)

    def forward(self, x, edge_index, edge_attr, batch):
        # Propagación a través de todas las capas convolucionales
        num_capa=0
        # x es un tensor que tiene las características de todos los nodos de todos los grafos en el batch
        x = self.node_encoder(x) # Transformará cada representación de los nodos a (num_nodes, dim_repr_nodo). Esto permite poder usar los "embeddings" de cada nodo en el proceso de MessagePassing al conseguir una matriz compatible con la multiplicación definida en la capa convolucional para la representación de cada nodo.
        for capa in self.capas:
            if num_capa == self.num_capas- 1:  # si es la ultima capa no se añade función lineal
                x =  F.dropout(capa(x, edge_index, edge_attr), p=self.drop_ratio, training=self.training) # Obtener representaciones actualizadas
            else:
                x = capa(x, edge_index, edge_attr) # Obtener representaciones actualizadas
                x = F.dropout(F.relu(x), p=self.drop_ratio, training=self.training)  # Activación no lineal (en este caso ReLu)
            num_capa = num_capa + 1
        # Transformando readout
        if(self.graph_pooling == "sum"):
            self.pooling = global_add_pool
        elif(self.graph_pooling == "mean"):
            self.pooling = global_mean_pool
        elif(self.graph_pooling == "max"):
            self.pooling = global_max_pool
        x = self.pooling(x, batch) # la variable batch según la documentación oficial sirve para indicar el grafo al que pertenece cada nodo

        # Aplicando el clasificador final (una red neuronal simple de una capa con dropout)
        x = F.dropout(x, p=self.drop_ratio, training=self.training) # self.training define si aplicar dropout (True) o no (Flase). Se gestiona automaticamente al hacer train y eval
        x = self.perceptron(x)
        
        return x