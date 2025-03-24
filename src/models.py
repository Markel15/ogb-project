import torch
from layers import GraphConvolution
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, TopKPooling, SAGPooling

class GCN(torch.nn.Module):
    def __init__(self, num_clases, num_capas, dim_repr_nodo, metodo_agregacion, drop_ratio, graph_pooling, ratio=0.4): # Por simplificación, de momento las dimensiones de todas las capas ocultas se mantienen iguales
        super(GCN, self).__init__()
        # Definiendo variables para poder utilizarlas en los siguientes metodos 
        self.graph_pooling = graph_pooling
        self.drop_ratio = drop_ratio
        self.num_capas = num_capas
        # Crear una lista de capas GraphConvolution
        self.capas = torch.nn.ModuleList()
        for capa in range(num_capas):
            self.capas.append(GraphConvolution(dim_repr_nodo, metodo_agregacion))
        self.node_encoder = torch.nn.Embedding(1, dim_repr_nodo) # Codificador para transformar las dimensiones de los nodos. Es como una capa de unión para conectar las representaciones iniciales con las de las capas 
        # A continuación definimos la capa de salida que será alimentada con el pooling del grafo completo y determinará la clasificación por cada instancia
        
         # Ajustar la dimensión de la capa final dependiendo el tipo de pooling
        if self.graph_pooling == "combinacion":
            final_dim = dim_repr_nodo * 3
        elif self.graph_pooling == "topk":
            # Se define la capa de TopKPooling. El parámetro ratio se puede ajustar.
            self.topk_pool = TopKPooling(dim_repr_nodo, ratio=ratio)
            final_dim = dim_repr_nodo
        elif self.graph_pooling == "sag":
            self.sag_pool = SAGPooling(dim_repr_nodo, ratio=ratio)
            final_dim = dim_repr_nodo
        else:
            final_dim = dim_repr_nodo
        self.perceptron = torch.nn.Linear(final_dim, num_clases)

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
            x = global_add_pool(x, batch) # la variable batch según la documentación oficial sirve para indicar el grafo al que pertenece cada nodo
        elif(self.graph_pooling == "mean"):
            x = global_mean_pool(x, batch)
        elif(self.graph_pooling == "max"):
            x = global_max_pool(x, batch)
        elif self.graph_pooling == "combinacion":
            x_sum = global_add_pool(x, batch)
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_sum, x_mean, x_max], dim=1)
        elif self.graph_pooling == "topk":
            # Aplicar TopKPooling para reducir los nodos del grafo
            x, edge_index, edge_attr, batch, _, _ = self.topk_pool(x, edge_index, edge_attr, batch=batch)
            # Se aplica un pooling global simple para obtener una representación fija por grafo
            x = global_mean_pool(x, batch)
        elif self.graph_pooling == "sag":
            x, edge_index, _, batch, _, _ = self.sag_pool(x, edge_index, None, batch)
            x = global_mean_pool(x, batch)
        else:
            raise ValueError("El método de pooling no está definido")   

        # Aplicando el clasificador final (una red neuronal simple de una capa con dropout)
        x = F.dropout(x, p=self.drop_ratio, training=self.training) # self.training define si aplicar dropout (True) o no (Flase). Se gestiona automaticamente al hacer train y eval
        x = self.perceptron(x)
        
        return x