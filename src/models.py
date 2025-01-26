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
        # A continuación definimos la capa de salida que será alimentada con el pooling del grafo completo y determinará la clasificación por cada instancia
        self.perceptron = torch.nn.Linear(dim_repr_nodo, num_clases)

    def forward(self, x, edge_index, edge_attr, batch):
        # Propagación a través de todas las capas convolucionales
        num_capa=0
        for capa in self.capas:
            if num_capa == self.num_capas- 1:  # si es la ultima capa no se añade función lineal
                x = capa(x, edge_index, edge_attr) # Obtener representaciones actualizadas
            else:
                x = capa(x, edge_index, edge_attr) # Obtener representaciones actualizadas
                x = F.relu(x)  # Activación no lineal (en este caso ReLu)
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