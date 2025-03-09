import re
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from tqdm import tqdm # Para barra de carga (conocer tiempo previsto de cada proceso)
import matplotlib.pyplot as plt
import os
from datetime import datetime
from datetime import timedelta

from models import GCN
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

LR=0.001
DR=0.20
NC=4
ReprNodo=164

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
        y_pred.append(pred.argmax(dim=-1, keepdim=True).cpu())  # Aplicar argmax para obtener la clase predicha y poder comprobar con la correcta
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
    # split_idx = reducir_tamaño(split_idx, porcentaje=0.5)

    # Utilizar el evaluador del paquete
    evaluator = Evaluator('ogbg-ppa')
    train_loader = DataLoader(dataset[split_idx['train']], batch_size = 32, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx['valid']], batch_size = 32, shuffle=False)
    test_loader = DataLoader(dataset[split_idx['test']], batch_size = 32, shuffle=False)

    model = GCN(num_clases=dataset.num_classes, num_capas=NC, dim_repr_nodo=ReprNodo, metodo_agregacion='add', drop_ratio=DR, graph_pooling='combinacion')
    model = model.to(device)

    # Configuración de optimizador y criterio de pérdida
    optimizador = optim.Adam(model.parameters(), lr=LR) # Se puede poner como variable el learning rate
    criterio = torch.nn.CrossEntropyLoss() # CrossEntropyLoss es para clasificación multiclase

    # Entrenamiento y evaluación
    accuracy_validation = []
    accuracy_test = []
    best_valid_score = 0
    paciencia = 10  # Número máximo de épocas sin mejora
    epochs_sin_mejora = 0
    parar = False
    start_time = datetime.now()
    for epoch in range(1, 61):  # Nº epochs fijado a 100 pero se puede cambiar
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
        last_test_score = test_result['acc']
        loss_final = loss
        if (parar==True): break
    end_time = datetime.now()
    tiempo_total = str(timedelta(seconds=(end_time - start_time).seconds))
    plot_learning_curve(accuracy_validation, accuracy_test, last_test_score, best_valid_score, loss_final, tiempo_total)

def plot_learning_curve(accuracy_validation, accuracy_test, last_test_score, best_valid_score, loss_final, tiempo_total):
    epochs = range(1, len(accuracy_validation) + 1)

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
    nombre_archivo = f'img/curvas_precision_{hora_actual}_lr{LR}_drop{DR}_layers{NC}_dim{ReprNodo}_maxtest{last_test_score:.3f}_maxval{best_valid_score:.4f}_loss{loss_final}_{tiempo_total}.png'
    # Reemplazar los dos puntos ':' por un guion '-'
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