import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim

from models import GCN
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PygGraphPropPredDataset(name = 'ogbg-ppa')
    
    print(f'Usando el dispositivo: {device}')
    print(torch.cuda.get_device_name(0))

    split_idx = dataset.get_idx_split()

    # Utilizar el evaluador del paquete
    evaluator = Evaluator('ogbg-ppa')
    train_loader = DataLoader(dataset[split_idx['train']], batch_size = 32, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx['valid']], batch_size = 32, shuffle=False)
    test_loader = DataLoader(dataset[split_idx['test']], batch_size = 32, shuffle=False)

    model = GCN(num_clases=dataset.num_classes, num_capas=3, dim_repr_nodo=64, metodo_agregacion='add', drop_ratio=0.5, graph_pooling='mean')
    model = model.to(device)

    # Configuración de optimizador y criterio de pérdida
    optimizador = optim.Adam(model.parameters(), lr=0.001) # Se puede poner como variable el learning rate
    criterio = torch.nn.BCEWithLogitsLoss()


if __name__ == "__main__":
    main()