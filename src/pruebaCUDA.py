import torch

# Verificar si CUDA está disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Usando el dispositivo: {device}')
print(torch.cuda.get_device_name(0))

# Suponiendo que 'model' es tu red neuronal y 'data' son tus datos de grafo
# model = model.to(device)
# data = data.to(device)