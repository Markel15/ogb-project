import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from tqdm import tqdm # Para barra de carga (conocer tiempo previsto de cada proceso)

from models import GCN
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

def train(model, train_loader, optimizador, criterio, device):
    model.train() # metodo heredado de torch.nn.Module, pone el modelo en modo entrenamiento(dropout y más)
    total_loss = 0
    for data in tqdm(train_loader, desc="Entrenando...", unit="batch"):
        data = data.to(device)
        optimizador.zero_grad() # Antes de realizar una actualización de los pesos del modelo, es necesario poner a cero los gradientes acumulados de la iteración anterior
        pred = model(data.x, data.edge_index, data.edge_attr, data.batch) # Llamada implicita al metodo forward del modelo con todo lo necesario (características de los nodos, aristas y batch)
        loss = criterio(pred, data.y.view(-1)) # Conseguir un unico vector con las etiquetas de todas las predicciones del batch
        loss.backward() # El método backward() calcula los gradientes de la pérdida respecto a los parámetros del modelo
        optimizador.step() # Actualiza los pesos del modelo de acuerdo con los gradientes calculados durante la fase anterior (con el loss.backward())
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, loader, evaluator, device):
    model.eval() # Iniciar modo de evaluación 
    y_true, y_pred = [], [] # Inicializar listas vacias con las etiquetas correctas y las predicas.
    for data in tqdm(loader, desc="Evaluando...", unit="batch"):
        data = data.to(device)
        with torch.no_grad(): # Utilizado para no calcular los gradientes ya que como aparece en la documentacion no es necesario en la inferencia puesto que no estamos entrenando nada y libera memoria
            pred = model(data.x, data.edge_index, data.edge_attr, data.batch) # Obtener predicción
        y_true.append(data.y.cpu()) # Actualizar las listas y mover los datos a la cpu por conveniencia para liberar memoria y posteriores posibles calculos
        y_pred.append(pred.cpu())
    y_true = torch.cat(y_true, dim=0) # Concatenar los elementos en un único tensor
    y_pred = torch.cat(y_pred, dim=0)
    return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})

def inicializar_x(data): 
    data.x = torch.zeros(data.num_nodes, dtype=torch.long) # El dataset no cuenta con representaciones iniciales de los nodos, así que se inicializan a 0 y se irán actualizando con el pase de mensajes
    return data

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PygGraphPropPredDataset(name = 'ogbg-ppa', transform = inicializar_x)
    
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
    criterio = torch.nn.CrossEntropyLoss() # CrossEntropyLoss es para clasificación multiclase

    # Entrenamiento y evaluación
    best_valid_score = 0
    for epoch in range(1, 101):  # Nº epochs fijado a 100 pero se puede cambiar
        loss = train(model, train_loader, optimizador, criterio, device)
        print(f'Epoca {epoch}, Pérdida de entrenamiento: {loss:.4f}')

        # Evaluación en conjunto de validación
        valid_result = evaluate(model, valid_loader, evaluator, device)
        print(f'Epoca {epoch}, Resultado de validación: {valid_result}')

        if valid_result['accuracy'] > best_valid_score:
            best_valid_score = valid_result['accuracy']
            print(f'Nuevos mejores resultados en validación (accuracy): {best_valid_score:.4f}')

        # Evaluación final en el conjunto de prueba
        test_result = evaluate(model, test_loader, evaluator, device)
        print(f'Resultados finales en test: {test_result}')


if __name__ == "__main__":
    main()