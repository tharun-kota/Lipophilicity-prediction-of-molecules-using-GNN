# 💧 Predicting Molecular Lipophilicity using Graph Neural Networks (GCN with PyTorch Geometric)

This repository contains a complete implementation of a **Graph Neural Network (GNN)** model using **PyTorch Geometric (PyG)** to predict **lipophilicity (logD at pH 7.4)** of organic molecules from the **MoleculeNet** `lipo` dataset.

---

## 📦 Key Components Used

### 📚 Dataset
- **Name:** `lipo` from [MoleculeNet](https://moleculenet.org/)
- **Target:** LogD at pH 7.4 (a measure of lipophilicity)
- **Graphs:** Each molecule is represented as a graph with:
  - **Nodes** = atoms (with features like degree, charge, aromaticity)
  - **Edges** = bonds
- **Total Samples:** ~4,200 molecules

---

### 🧠 Model: Graph Convolutional Network (GCN)

- Built using `torch_geometric.nn.GCNConv`
- Layer Structure:
  - 3 GCN layers with `tanh` activation
  - Global max & mean pooling (concatenated)
  - Linear output layer for regression
- Trained using:
  - **Loss:** Mean Squared Error (MSE)
  - **Optimizer:** Adam
  - **Epochs:** 2000
  - **Batch Size:** 64

---

### 🔬 Libraries Used

| Purpose                 | Library        |
|------------------------|----------------|
| Graph Learning         | PyTorch Geometric |
| Deep Learning Engine   | PyTorch        |
| Molecular Chemistry    | RDKit, PubChemPy |
| Visualization          | Matplotlib, Seaborn, NetworkX |
| Evaluation             | scikit-learn (R², RMSE) |
| Data Manipulation      | pandas, numpy |

---

## 📈 Results & Metrics

### ✅ Training Metrics
- **R² Score:** 70–80% depending on data split
- **RMSE:** ~0.6–0.9 on average
- These values indicate that the GCN model learns meaningful chemical signals for predicting lipophilicity.

### 🧪 Test Metrics
- **Final Test R² Score:** `0.451` (≈ 45.1%)
- **Final Test RMSE:** `0.867` logD units

This means:
- ~45% of the variance in molecular lipophilicity is explained by the GCN model.
- Predictions are on average ~0.87 logD units away from true values.

---

## 📊 Parity Plot

The parity plot compares predicted vs. actual values:

- **Dots near red line** → Good predictions
- **Dots far from red line** → Larger prediction errors

![Parity Plot](assets/parity_plot.png) *(image placeholder)*

---

## 🧪 Workflow Overview

1. **Load and preprocess dataset** (`MoleculeNet(name="lipo")`)
2. **Explore SMILES strings**, visualize molecules, extract names using `PubChemPy`
3. **Visualize molecular graphs** using `NetworkX`
4. **Define GCN model**
5. **Train the model** using PyTorch GPU/CPU
6. **Evaluate** using R² and RMSE
7. **Plot training & test predictions**, accuracy and error metrics

---

## 🧠 Insights

- The model captures underlying molecular patterns that affect lipophilicity.
- GCNs are well-suited for structured molecular data.
- Performance can be improved by:
  - Using advanced models (e.g., GIN, SchNet)
  - Adding richer features (e.g., quantum descriptors, 3D geometry)
  - Hyperparameter tuning

---

## 🛠 Future Improvements

- Cross-validation for robustness
- Advanced pooling (attention-based)
- Integration with generative models for molecule design
- Deploy as a WebApp or API for real-time predictions

---

## 🤝 Acknowledgements

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [MoleculeNet Benchmark](https://moleculenet.org/)
- [RDKit](https://www.rdkit.org/)
- [PubChemPy](https://pubchempy.readthedocs.io/en/latest/)

---

## 📁 Repo Structure

```bash
.
├── lipophilicity_gcn.ipynb     # Main notebook (training + evaluation)
├── README.md                   # This file
├── assets/
│   └── parity_plot.png         # Sample output (optional)
