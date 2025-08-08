# Graph Neural Networks for Graph Classification on OGBG-PPA

This repository contains a collection of graph neural network (GNN) models implemented and evaluated for graph classification tasks using the [OGBG-PPA dataset](https://ogb.stanford.edu/docs/graphprop/). The work explores and compares various architectures such as GCN and GIN, applying different pooling and aggregation strategies.

## 📦 Models Implemented

All models are built using PyTorch Geometric. Each model can be configured with a wide variety of training parameters.

- **Baseline** (Optional model without graph neural network architecture to compare )
- **Graph Convolutional Network (GCN)**
- **Graph Isomorphism Network (GIN)**
- **Custom configurations**:
  - Number of layers
  - Node representation dimensionality
  - Aggregation method (`sum`, `mean`, `max`, `add`)
  - Dropout rate
  - Use of batch normalization
  - Use of residual connections
  - Pooling method (`mean`, `add`, `max`, `top-k`, etc.)

## 🔧 Training Techniques

- **Batch Normalization**: optional, improves training stability
- **Residual Connections**: optional, helps deeper networks
- **Dropout**: configurable dropout ratio

## 📊 Best Results (Summary)

The models were evaluated using the standarized evaluator from the package.

| Model     | Layers | Dim | Pooling | Batch Norm | Residual | LR     | Dropout | Epsilon | Val Acc. | Test Acc. | Epochs | Time (GPU)    |
|-----------|--------|-----|---------|------------|----------|--------|---------|---------|----------|-----------|--------|----------|
| Baseline  | 3      | 128 | mean    | ❌         | ❌       | 0.001  | 0.2     | -       | ~11.9%   | ~13.3%    | 25     | 30 min   |
| GCN       | 5      | 300 | mean    | ✅         | ❌       | 0.001  | 0.2     | -       | ~61.5%   | ~65.3%    | 60     | 4h 44min |
| GIN       | 4      | 200 | mean    | ✅         | ❌       | 0.001  | 0.2     | 0.01    | ~68.2%   | ~73.1%    | 60     | 3h 24min |

The reported training times were obtained using:

- **GPU**: NVIDIA GeForce RTX 3080 (with CUDA)
- **CPU**: Intel® Core™ i7-10700K @ 3.80GHz
- **RAM**: 32 GB DDR4 3200 MHz

## ⚙️ Requirements

- Python 3.8+
- PyTorch >= 1.12
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- tqdm, matplotlib

```bash
pip install torch-geometric
pip install matplotlib tqdm
