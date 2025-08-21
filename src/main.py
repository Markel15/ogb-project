import re
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from tqdm import tqdm # Para barra de carga (conocer tiempo previsto de cada proceso)
import matplotlib.pyplot as plt
import os
from datetime import datetime
from datetime import timedelta
import argparse

from models import GNN
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

def parse_args():
    """Configure argumetns"""
    parser = argparse.ArgumentParser(
        description='OGB Protein Classification with GNNs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic arguments
    parser.add_argument('--model', type=str, default='gin',
                       choices=['gin', 'gcn'],
                       help='GNN type: gcn or gin. gin by default')
    
    parser.add_argument('--layers', type=int, default=4,
                       help='Number of layers (4 by default)')
    
    parser.add_argument('--hidden_dim', type=int, default=200,
                       help='Node representation dimension (200 by default)')
    
    parser.add_argument('--epochs', type=int, default=60,
                       help='Number of epochs for training (60 by default)')
    
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (0.001 by default)')
    
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate (0.2 by default)')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size, 32 by default')
    
    # Model specific options
    parser.add_argument('--pooling', type=str, default='mean',
                       choices=['mean', 'add', 'max', 'combinacion', 'topk', 'sagpooling'],
                       help='Graph pooling type')
    
    parser.add_argument('--aggregation', type=str, default='add',
                       choices=['add', 'mean', 'max'],
                       help='Agreggation method')
    
    parser.add_argument('--batch_norm', action='store_true',
                       help='Use batch normalization')
    
    parser.add_argument('--no_batch_norm', dest='batch_norm', action='store_false',
                        help='No batch normalization')

    parser.set_defaults(batch_norm=True)
    
    parser.add_argument('--residual', action='store_true',
                       help='Use residual connections')
    
    parser.add_argument('--no_residual', dest='residual', action='store_false',
                        help='No residual connections')

    parser.set_defaults(residual=False)

    parser.add_argument('--eps', type=float, default=0.01, 
                        help="Value of epsilon for GINs")

    parser.add_argument('--ratio', type=float, default=0.40, 
                        help="Percentage of instances to left after aplying Top-K pooling or SAGPooling. 0.40 by default")

    # Training options
    parser.add_argument('--patience', type=int, default=12,
                       help='Patience for the early stopping (12 by default)')
    
    # Opciones de sistema
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use: auto, cpu or cuda')
    
    return parser.parse_args()

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
    y_true, y_pred = [], [] # Inicializar listas vacias con las etiquetas correctas y las predichas.
    for data in tqdm(loader, desc="Evaluando...", unit="batch"):
        data = data.to(device)
        with torch.no_grad(): # Utilizado para no calcular los gradientes ya que como aparece en la documentacion no es necesario en la inferencia puesto que no estamos entrenando nada y libera memoria
            pred = model(data.x, data.edge_index, data.edge_attr, data.batch) # Obtener predicción
        y_true.append(data.y.cpu()) # Actualizar las listas y mover los datos a la cpu por conveniencia para liberar memoria y posteriores posibles calculos
        y_pred.append(pred.argmax(dim=-1, keepdim=True).cpu())  # Aplicar argmax para obtener la clase predicha y poder comprobar con la correcta
    y_true = torch.cat(y_true, dim=0) # Concatenar los elementos en un único tensor
    y_pred = torch.cat(y_pred, dim=0)
    return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})

def inicializar_x(data): 
    data.x = torch.zeros(data.num_nodes, dtype=torch.long) # El dataset no cuenta con representaciones iniciales de los nodos, así que se inicializan a 0 y se irán actualizando con el pase de mensajes
    return data

def main():
    args = parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    dataset = PygGraphPropPredDataset(name = 'ogbg-ppa', transform = inicializar_x)
    
    print(f'Usando el dispositivo: {device}')
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    split_idx = dataset.get_idx_split()
    # split_idx = reducir_tamaño(split_idx, porcentaje=0.5)

    # Utilizar el evaluador del paquete
    evaluator = Evaluator('ogbg-ppa')
    train_loader = DataLoader(dataset[split_idx['train']], batch_size = args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx['valid']], batch_size = args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[split_idx['test']], batch_size = args.batch_size, shuffle=False)


    # TODO: HAY QUE AÑADIR TAMBIEN AL ARGS EL EPSILON E IGUAL ALGO MÁS QUE AHORA SE ME HAYA OLVIDADO (Ratio del topk o sagpooling creo (el mismo para los 2 pero no sé si está implementado en los 2))
    model = GNN(num_clases=dataset.num_classes, tipo_gnn=args.model, num_capas=args.layers, dim_repr_nodo=args.hidden_dim, metodo_agregacion=args.aggregation, drop_ratio=args.dropout, graph_pooling=args.pooling, usar_residual=args.residual,  usar_batch_norm=args.batch_norm, ratio=args.ratio, epsilon=args.eps)
    model = model.to(device)

    # Configuración de optimizador y criterio de pérdida
    optimizador = optim.Adam(model.parameters(), lr=args.lr) # Se puede poner como variable el learning rate
    criterio = torch.nn.CrossEntropyLoss() # CrossEntropyLoss es para clasificación multiclase

    # Entrenamiento y evaluación
    accuracy_validation = []
    accuracy_test = []
    best_valid_score = 0
    best_test_score = 0
    paciencia = args.patience  # Número máximo de épocas sin mejora
    epochs_sin_mejora = 0
    parar = False
    start_time = datetime.now()
    for epoch in range(1, args.epochs + 1):  # Nº epochs fijado a 100 pero se puede cambiar
        loss = train(model, train_loader, optimizador, criterio, device)
        print(f'Epoca {epoch}, Pérdida de entrenamiento: {loss:.4f}')

        # Evaluación en conjunto de validación
        valid_result = evaluate(model, valid_loader, evaluator, device)
        accuracy_validation.append(valid_result['acc'])
        print(f'Epoca {epoch}, Resultado de validación: {valid_result}')

        if valid_result['acc'] > best_valid_score:
            best_valid_score = valid_result['acc']
            print(f'Nuevos mejores resultados en validación (accuracy): {best_valid_score:.4f}')
            epochs_sin_mejora = 0
        else:
            epochs_sin_mejora += 1
            if epochs_sin_mejora >= paciencia:
                print(f"Early stopping activado en la época {epoch}. No hubo mejora en la validación durante {paciencia} épocas.")
                parar=True; #Para que los gráficos coincidan en dimensiones usamos parar y no break
        # Evaluación final en el conjunto de prueba
        test_result = evaluate(model, test_loader, evaluator, device)
        accuracy_test.append(test_result['acc'])
        print(f'Resultados finales en test: {test_result}')
        if test_result['acc'] > best_test_score:
            best_test_score = test_result['acc']
        loss_final = loss
        if (parar==True): break
    end_time = datetime.now()
    tiempo_total = str(timedelta(seconds=(end_time - start_time).seconds))
    plot_learning_curve(accuracy_validation, accuracy_test, best_test_score, best_valid_score, loss_final, tiempo_total, args)

def plot_learning_curve(accuracy_validation, accuracy_test, best_test_score, best_valid_score, loss_final, tiempo_total, args):
    epochs = range(1, len(accuracy_validation) + 1)
    args = args

    # Graficar la precisión de validación y prueba
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, accuracy_validation, label='Precisión de validación', color='blue')
    plt.plot(epochs, accuracy_test, label='Precisión de test', color='green')

    # Etiquetas y título
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.title('Curva de Precisión - Validación y Test')
    plt.legend()
    plt.grid(True)

    hora_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Obtener hora para poder almacenar más de una imagen y no se sobreescriba 

    # Guardar la gráfica
    if not os.path.exists("img"):
        os.makedirs("img") 
    nombre_archivo = f'img/curvas_precision_{hora_actual}_lr{args.lr}_drop{args.dropout}_layers{args.layers}_dim{args.hidden_dim}_maxtest{best_test_score:.3f}_maxval{best_valid_score:.4f}_loss{loss_final}_{tiempo_total}.png'
    nombre_archivo = re.sub(r':', '-', nombre_archivo)
    plt.savefig(nombre_archivo)
    plt.show()

def reducir_tamaño(split_idx, porcentaje=0.3):  # Funcion para reducir el tamaño del dataset a un porcentaje para poder hacer pruebas más rapidas
    """
    Reducir el tamaño del dataset a un porcentaje dado.
    """
    num_train = int(len(split_idx['train']) * porcentaje)
    num_valid = int(len(split_idx['valid']) * porcentaje)
    num_test = int(len(split_idx['test']) * porcentaje)

    # Seleccionamos solo una fracción de los índices de cada conjunto
    split_idx['train'] = split_idx['train'][:num_train]
    split_idx['valid'] = split_idx['valid'][:num_valid]
    split_idx['test'] = split_idx['test'][:num_test]

    return split_idx

if __name__ == "__main__":
    main()