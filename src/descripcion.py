from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader

# Cargar el dataset OGBG-PPA
dataset = PygGraphPropPredDataset(name='ogbg-ppa')
numero = 0

# Obtener los datos de entrenamiento y prueba
split_idx = dataset.get_idx_split()
train_loader = DataLoader(dataset[split_idx['train']], batch_size = 32, shuffle=True)
valid_loader = DataLoader(dataset[split_idx['valid']], batch_size = 32, shuffle=False)
test_loader = DataLoader(dataset[split_idx['test']], batch_size = 32, shuffle=False)

data = dataset[numero]

# Describir el dataset
print(f"Nombre del dataset: {dataset.name}")
print(f"Describiendo el grafo Nº : {numero}")
print(f"Número de nodos: {data.num_nodes}")
print(f"Número de aristas: {data.num_edges}")
print(f"Número de características de nodos: {data.num_node_features}")
print(f"Número de características de aristas: {data.num_edge_features}")
print(f"Es el grafo dirigido: {data.is_directed()}")
print(f"Tiene bucles: {data.has_self_loops()}")
