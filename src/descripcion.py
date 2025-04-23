from ogb.graphproppred import PygGraphPropPredDataset
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Cargar el dataset OGBG-PPA
dataset = PygGraphPropPredDataset(name='ogbg-ppa')

print(f"Nombre del dataset: {dataset.name}")
print(f"Cantidad de grafos: {len(dataset)}")

# Obtener las etiquetas de los grafos
# Si el dataset tiene múltiples clases y las etiquetas son enteros, podemos contar su distribución
labels = []
for data in dataset:
    labels.append(data.y.item())  # Almacenamos las etiquetas de cada grafo en la lista

# Convertir la lista de etiquetas a un array de numpy
labels = np.array(labels)

# Calcular la distribución de clases
unique, counts = np.unique(labels, return_counts=True)

# Imprimir la distribución
for u, c in zip(unique, counts):
    print(f"Clase {u}: {c} grafos")

# Visualizar la distribución en un gráfico de barras
plt.bar(unique, counts)
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.title('Distribución de clases en el dataset OGBG-PPA')
plt.show()

# Visualizar el primer grafo
data = dataset[0]

# Acceder a la información del grafo
edge_index = data.edge_index  # Conexiones de los nodos
y = data.y                   # Etiqueta de la propiedad del grafo

# Convertir el grafo a un formato de NetworkX
G = nx.Graph()

# Añadir las aristas del grafo
for i in range(edge_index.shape[1]):  # edge_index tiene la forma (2, num_aristas)
    node1, node2 = edge_index[:, i].numpy()
    G.add_edge(node1, node2)

# Visualizar el grafo
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, seed=42)  # Distribución de los nodos
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=12, font_weight='bold', edge_color='gray')
plt.title(f"Visualización del grafo con propiedad: {y.item()}")
plt.show()

split_idx = dataset.get_idx_split()
# Función para describir un split con estadísticas agregadas
def describe_split(split_name, split_indices):
    print(f"\nDescribiendo el split: {split_name}")
    
    # Inicializar variables para estadísticas agregadas
    num_graphs = len(split_indices)
    node_counts = []
    edge_counts = []
    
    # Calcular estadísticas para cada grafo en el split
    for i in range(num_graphs):
        data = dataset[split_indices[i]]
        
        # Contar nodos y aristas
        node_counts.append(data.num_nodes)
        edge_counts.append(data.num_edges)
        
    # Calcular promedios y desviaciones
    avg_nodes = np.mean(node_counts)
    avg_edges = np.mean(edge_counts)
    std_nodes = np.std(node_counts)
    std_edges = np.std(edge_counts)
    
    # Mostrar estadísticas agregadas
    print(f"Número total de grafos en el split {split_name}: {num_graphs}")
    print(f"Promedio de nodos por grafo: {avg_nodes:.2f} ± {std_nodes:.2f}")
    print(f"Promedio de aristas por grafo: {avg_edges:.2f} ± {std_edges:.2f}")
    
    # Mostrar distribución de clases en el split
    labels = [dataset[idx].y.item() for idx in split_indices]
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nDistribución de clases en el split {split_name}:")
    for u, c in zip(unique, counts):
        print(f"Clase {u}: {c} grafos")

    # Visualizar la distribución de clases en un gráfico de barras
    plt.figure(figsize=(6, 4))
    plt.bar(unique, counts)
    plt.xlabel('Clases')
    plt.ylabel('Frecuencia')
    plt.title(f'Distribución de clases en el split {split_name}')
    plt.show()

# Describir el split de entrenamiento
describe_split('Entrenamiento', split_idx['train'])

# Describir el split de validación
describe_split('Validación', split_idx['valid'])

# Describir el split de prueba
describe_split('Prueba', split_idx['test'])
'''
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
'''