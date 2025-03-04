import re
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import degree
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from torch_scatter import scatter_mean  # Para agregar valores en base a índices

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

# Parámetros globales
LR = 0.005
DR = 0.20

# Parámetros para la representación de los nodos:
# Se utiliza un one-hot del grado (limitado a max_degree) + raw degree (1) + average neighbor degree (1)
max_degree = 18  
dim_repr_nodo = max_degree + 3

# Parámetros para el MLP (perceptrón)
NUM_LAYERS = 2      # Número total de capas del perceptrón (incluye la capa de salida)
HIDDEN_DIM = 200    # Dimensión de las capas ocultas

def inicializar_x(data):
    """
    Enriquecer la representación inicial de los nodos incluyendo:
      - Codificación one-hot del grado (limitado a max_degree)
      - El grado crudo (raw degree) como valor float
      - El grado promedio de los vecinos
    """
    # Calcular el grado de cada nodo (usando el primer índice de edge_index)
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
    deg_float = deg.to(torch.float)
    
    # Codificación one-hot del grado (limitamos el grado a max_degree)
    one_hot = F.one_hot(torch.clamp(deg, max=max_degree), num_classes=max_degree + 1).to(torch.float)
    
    # Calcular el grado promedio de los vecinos.
    # Para cada nodo 'i', se promedia el grado de los nodos 'j' tales que existe la arista (j -> i).
    avg_neighbor_deg = scatter_mean(deg_float[data.edge_index[0]], data.edge_index[1], dim=0, dim_size=data.num_nodes)
    # En caso de que algún nodo no tenga vecinos (resultando en NaN), lo reemplazamos por 0.
    avg_neighbor_deg[avg_neighbor_deg != avg_neighbor_deg] = 0.0

    # Concatenar las características: [one_hot, raw degree, average neighbor degree]
    data.x = torch.cat([one_hot, deg_float.unsqueeze(1), avg_neighbor_deg.unsqueeze(1)], dim=1)
    return data

class MLP(torch.nn.Module):
    def __init__(self, num_clases, dim_repr_nodo, drop_ratio, graph_pooling, num_layers, hidden_dim):
        """
        Args:
            num_clases (int): Número de clases de salida.
            dim_repr_nodo (int): Dimensión de las representaciones de los nodos.
            drop_ratio (float): Porcentaje de dropout.
            graph_pooling (str): Tipo de pooling global a utilizar ('sum', 'mean' o 'max').
            num_layers (int): Número total de capas en el MLP (incluye capa de salida).
            hidden_dim (int): Dimensión de las capas ocultas.
        """
        super(MLP, self).__init__()
        self.drop_ratio = drop_ratio
        self.num_layers = num_layers

        # Seleccionar la función de pooling global según el parámetro
        if graph_pooling == "sum":
            self.pooling = global_add_pool
        elif graph_pooling == "mean":
            self.pooling = global_mean_pool
        elif graph_pooling == "max":
            self.pooling = global_max_pool
        else:
            raise ValueError(f"Pooling type '{graph_pooling}' not supported.")

        # Construir las capas del MLP de forma automática
        self.layers = torch.nn.ModuleList()
        if num_layers == 1:
            # Caso base: una única capa que mapea directamente a la salida
            self.layers.append(torch.nn.Linear(dim_repr_nodo, num_clases))
        else:
            # Primera capa: de la representación de entrada a la dimensión oculta
            self.layers.append(torch.nn.Linear(dim_repr_nodo, hidden_dim))
            # Capas ocultas intermedias (si existen)
            for _ in range(num_layers - 2):
                self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            # Última capa: de la dimensión oculta al número de clases
            self.layers.append(torch.nn.Linear(hidden_dim, num_clases))

    def forward(self, x, batch):
        """
        Args:
            x (Tensor): Representaciones de los nodos de forma [num_nodos, dim_repr_nodo].
            batch (Tensor): Vector que asigna cada nodo a un grafo en el batch.
        Returns:
            Tensor: Salida del MLP (logits) para cada grafo.
        """
        # Aplicar pooling global para obtener una representación por grafo
        x = self.pooling(x, batch)
        # Aplicar dropout inicial
        x = F.dropout(x, p=self.drop_ratio, training=self.training)
        
        if self.num_layers == 1:
            x = self.layers[0](x)
        else:
            for i, layer in enumerate(self.layers):
                # Para todas las capas excepto la última, aplicar ReLU y dropout
                if i < self.num_layers - 1:
                    x = F.relu(layer(x))
                    x = F.dropout(x, p=self.drop_ratio, training=self.training)
                else:
                    x = layer(x)
        return x

def train(model, train_loader, optimizador, criterio, device):
    model.train()
    total_loss = 0
    for data in tqdm(train_loader, desc="Entrenando...", unit="batch"):
        data = data.to(device)
        optimizador.zero_grad()
        # El MLP usa únicamente x y batch
        pred = model(data.x, data.batch)
        loss = criterio(pred, data.y.view(-1))
        loss.backward()
        optimizador.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, loader, evaluator, device):
    model.eval()
    y_true, y_pred = [], []
    for data in tqdm(loader, desc="Evaluando...", unit="batch"):
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.x, data.batch)
        y_true.append(data.y.cpu())
        y_pred.append(pred.argmax(dim=-1, keepdim=True).cpu())
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})

def plot_learning_curve(accuracy_validation, accuracy_test, last_test_score, best_valid_score, loss_final, tiempo_total):
    epochs = range(1, len(accuracy_validation) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, accuracy_validation, label='Precisión de validación', color='blue')
    plt.plot(epochs, accuracy_test, label='Precisión de test', color='green')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.title('Curva de Precisión - Validación y Test')
    plt.legend()
    plt.grid(True)
    hora_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists("img"):
        os.makedirs("img")
    nombre_archivo = f'img/MLP_{hora_actual}_NumCapas_{NUM_LAYERS}_lr{LR}_drop{DR}_hidden_dim_{HIDDEN_DIM}_maxtest{last_test_score:.3f}_maxval{best_valid_score:.4f}_loss{loss_final}_{tiempo_total}.png'
    nombre_archivo = re.sub(r':', '-', nombre_archivo)
    plt.savefig(nombre_archivo)
    plt.show()

def reducir_tamaño(split_idx, porcentaje=0.3):
    num_train = int(len(split_idx['train']) * porcentaje)
    num_valid = int(len(split_idx['valid']) * porcentaje)
    num_test = int(len(split_idx['test']) * porcentaje)
    split_idx['train'] = split_idx['train'][:num_train]
    split_idx['valid'] = split_idx['valid'][:num_valid]
    split_idx['test'] = split_idx['test'][:num_test]
    return split_idx

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Se aplica la transformación para asignar la representación enriquecida a cada nodo.
    dataset = PygGraphPropPredDataset(name='ogbg-ppa', transform=inicializar_x)
    
    print(f'Usando el dispositivo: {device}')
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
    
    split_idx = dataset.get_idx_split()
    # Si se quiere trabajar con un subconjunto del dataset, se puede usar la función reducir_tamaño
    # split_idx = reducir_tamaño(split_idx, porcentaje=0.5)

    evaluator = Evaluator('ogbg-ppa')
    train_loader = DataLoader(dataset[split_idx['train']], batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset[split_idx['test']], batch_size=32, shuffle=False)

    # Se crea el modelo MLP utilizando la dimensión enriquecida de las representaciones de los nodos y los parámetros para configurar las capas automáticamente.
    model = MLP(num_clases=dataset.num_classes,
                dim_repr_nodo=dim_repr_nodo,
                drop_ratio=DR,
                graph_pooling='mean',
                num_layers=NUM_LAYERS,
                hidden_dim=HIDDEN_DIM)
    model = model.to(device)

    optimizador = optim.Adam(model.parameters(), lr=LR)
    criterio = torch.nn.CrossEntropyLoss()

    accuracy_validation = []
    accuracy_test = []
    best_valid_score = 0
    paciencia = 7  # Épocas sin mejora para early stopping
    epochs_sin_mejora = 0
    parar = False
    start_time = datetime.now()

    for epoch in range(1, 61):
        loss = train(model, train_loader, optimizador, criterio, device)
        print(f'Época {epoch}, Pérdida de entrenamiento: {loss:.4f}')

        valid_result = evaluate(model, valid_loader, evaluator, device)
        accuracy_validation.append(valid_result['acc'])
        print(f'Época {epoch}, Resultado de validación: {valid_result}')

        if valid_result['acc'] > best_valid_score:
            best_valid_score = valid_result['acc']
            print(f'Nuevos mejores resultados en validación (accuracy): {best_valid_score:.4f}')
            epochs_sin_mejora = 0
        else:
            epochs_sin_mejora += 1
            if epochs_sin_mejora >= paciencia:
                print(f"Early stopping activado en la época {epoch}. No hubo mejora durante {paciencia} épocas.")
                parar = True

        test_result = evaluate(model, test_loader, evaluator, device)
        accuracy_test.append(test_result['acc'])
        print(f'Resultados en test: {test_result}')
        last_test_score = test_result['acc']
        loss_final = loss
        if parar:
            break

    end_time = datetime.now()
    tiempo_total = str(timedelta(seconds=(end_time - start_time).seconds))
    plot_learning_curve(accuracy_validation, accuracy_test, last_test_score, best_valid_score, loss_final, tiempo_total)

if __name__ == "__main__":
    main()
