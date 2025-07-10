# Quantum State Prediction with Transformers

A quantum machine learning project that trains transformer models to predict quantum density matrices from measurement sequences. The model learns to reconstruct quantum states of two-qubit systems based on preparation and measurement data with varying probe qubit spatial separations.

## Project Overview

This project uses a modified Llama transformer architecture to learn quantum state tomography - predicting the full quantum density matrix of a two-qubit system from partial measurement information. The model processes sequences of quantum measurements and outputs 4×4 complex density matrices representing the quantum state across different spatial configurations.

### Key Features

- **Transformer-based quantum state prediction**: Uses HuggingFace's Llama architecture adapted for quantum data
- **Multiple spatial separations**: Supports different probe qubit spatial separations (d=3,4,5,6)
- **Comprehensive metrics**: Tracks quantum-specific performance metrics (Quantum-Classical Cross Entropy, Negativity, Entanglement)
- **Robust training pipeline**: Includes checkpointing, reproducibility, and baseline tracking
- **Professional data handling**: Reproducible non-overlapping train/test splits with proper batching

## File Structure

### Core Training Files

#### `train_models.ipynb`
**Main training notebook** - The heart of the training pipeline.

**Features:**
- Trains LlamaPredictor models on quantum circuit data
- Supports multiple hyperparameter configurations (d, theta_idx, train_size)
- Implements comprehensive checkpointing system (20 checkpoints per epoch)
- Baseline checkpoint saving for consistent training curves
- Proper reproducibility with deterministic seeds
- Real-time training metrics monitoring
- Automatic train/test evaluation

**Key Parameters:**
- `d`: Spatial separation of probe qubits (3, 4, 5, 6)
- `theta_idx`: Parameter index for quantum gates
- `train_size`: Training dataset size (e.g., 81M samples)
- `test_size`: Test dataset size (e.g., 1M samples)
- `batch_size`: 1000 samples per batch

#### `TFM.py`
**Model architecture** - Custom transformer for quantum state prediction.

**LlamaPredictor Class:**
- Based on HuggingFace's Llama architecture
- Outputs 4×4 complex quantum density matrices
- Uses attention masking for different probe qubit spatial separations
- 36 layers, 96 embedding dimensions, 48 attention heads
- Custom final layer: Linear(96) → 32 → complex 4×4 matrix

**Attention Masks:**
- Pre-defined causal masks for each spatial separation configuration
- Ensures proper quantum causality in attention patterns based on probe geometry
- Different masks for L_max=36 with d∈{3,4,5,6}

#### `utils.py`
**Utility functions** - Quantum-specific operations and data handling.

**Quantum Functions:**
- `bSqc()`: Batched quantum-classical cross entropy calculation
- `Neg()`: Quantum Negativity measurement  
- `Sa()`: Entanglement entropy calculation
- `eps()`: Depolarization channel
- `purity()`: Quantum state purity

**Data Processing:**
- `torch_data()`: Loads and processes quantum measurement data
- `shuffle()`: Randomizes data order for training (with reproducible seeds)
- `create_train_test_split()`: Creates deterministic non-overlapping train/test splits using fixed indices

**Training Infrastructure:**
- `save_checkpoint()`: Saves model checkpoints with epoch/step naming
- `load_checkpoint()`: Loads model state for resuming training
- `save_checkpoint_and_test()`: Combined checkpoint saving and testing

### Data Processing Files

#### `post_process_d=3,4,5.ipynb`
**Data preprocessing** - Prepares raw quantum measurement data for training.

**Functions:**
- Loads raw measurement data from multiple experimental loops
- Shuffles and combines data across different quantum parameters
- Saves preprocessed data in PyTorch tensor format
- Handles different probe qubit spatial separations (d=3,4,5)

**Output Files:**
- `all_prepseq_theta={theta_idx}.pt`: Preparation sequences
- `all_shadow_state_theta={theta_idx}.pt`: Post-measurement states  
- `all_rhoS_theta={theta_idx}.pt`: True quantum density matrices

### Analysis Files

#### `read_record.ipynb`
**Training analysis** - Visualizes training progress and results.

**Features:**
- Loads training records from checkpoint saves
- Plots training curves (loss, quantum metrics)
- Compares performance across different configurations
- Handles baseline checkpoint data for consistent plotting

## Data Format

### Input Data Structure
```
data/
├── theta{theta_idx}/
│   ├── loop{loop}/
│   │   └── theta={theta_idx}  # Raw measurement data
│   ├── all_prepseq_theta={theta_idx}.pt      # Preprocessed sequences
│   ├── all_shadow_state_theta={theta_idx}.pt # Post-measurement states
│   └── all_rhoS_theta={theta_idx}.pt         # True density matrices
```

### Output Structure
```
save/
├── models/
│   └── model_d{d}_theta_idx{theta_idx}_epoch{epoch:04d}_step{step:04d}.pt
└── record/
    ├── epoch={epoch}_d={d}_theta_idx={theta_idx}_size{size}_train.pt
    └── epoch={epoch}_d={d}_theta_idx={theta_idx}_size{size}_test.pt
```

## Quantum Metrics

The model is evaluated on three key quantum information metrics:

1. **Quantum-Classical Cross Entropy (Sqc)**: Measures the cross entropy between predicted and true quantum states
2. **Negativity (Neg)**: Quantifies quantum entanglement between qubits
3. **Von Neumann Entropy (Sa)**: Measures quantum entanglement entropy

## Training Pipeline

1. **Data Loading**: Load preprocessed quantum measurement sequences
2. **Model Setup**: Initialize LlamaPredictor with quantum-specific architecture
3. **Training Loop**: 
   - Process batches of measurement sequences
   - Predict quantum density matrices
   - Calculate quantum loss (negative log-likelihood)
   - Update model parameters
4. **Evaluation**: Regular testing on held-out quantum states
5. **Checkpointing**: Save model state and metrics every ~1600 batches

## Getting Started

### Setup
```bash
# 1. Environment setup
python -m venv quantum_ml_env && source quantum_ml_env/bin/activate
pip install -r requirements.txt

# 2. Create directories
mkdir -p data/theta{0..10}/loop{0..16} save/{models,record}
```

### Data Access
**Google Drive**: https://drive.google.com/drive/folders/1mW342CtuutjiGhPIPRSAz8-r8XKolCJk?usp=sharing  
**Request access**: Email wandahou96@gmail.com with your Gmail address

### Quick Start
```bash
# 1. Download data from Google Drive → extract to data/
# 2. Preprocess data
jupyter notebook post_process_d=3,4,5.ipynb

# 3. Train model  
jupyter notebook train_models.ipynb

# 4. Monitor progress
jupyter notebook read_record.ipynb
```

### Hardware Requirements
- **Minimum**: 8GB GPU, 32GB RAM, 500GB storage
- **Recommended**: 24GB GPU, 64GB+ RAM, 1TB+ SSD

### Configuration
```python
# Default: 2.3M parameters, 81M train + 1M test samples
batch_size = 1000  # Reduce if GPU memory issues
n_embd = 96       # Reduce for smaller GPUs
n_layer = 36      # Reduce for smaller GPUs
```

## Reproducibility & Troubleshooting

### Full Reproducibility
- **Deterministic data split**: Fixed indices ensure identical train/test splits
- **Reproducible training**: `torch.manual_seed(42 + start_epoch)` + CUDA determinism
- **Resume training**: Set `start_epoch = N` to resume from epoch N

### Common Issues
```python
# GPU memory error → reduce batch size or model size
batch = 500           # instead of 1000
n_embd, n_layer = 48, 18  # instead of 96, 36

# Resume training from specific epoch
start_epoch = 1  # automatically loads previous checkpoint
```

## Technical Highlights

- **Professional checkpointing**: Epoch/step naming with automatic resuming
- **Memory efficient**: Batched processing with proper device management
- **Quantum-aware**: Specialized loss functions and evaluation metrics
- **Baseline tracking**: Untrained model performance for comparison
- **Comprehensive logging**: Real-time metrics and progress tracking

## Model Performance

The model learns to predict quantum density matrices with increasing accuracy across training. Key performance indicators:

- **Training Sqc**: Quantum-classical cross entropy on training data (with attention mask)
- **Test Sqc**: Quantum-classical cross entropy on test data (without attention mask)  
- **Test Neg**: Quantum entanglement measure
- **Test Sa**: Entanglement entropy

Baseline performance starts around Sqc=2.33, with trained models achieving significantly better quantum state fidelity.

---

*This project represents cutting-edge research in quantum machine learning, combining transformer architectures with quantum information theory for practical quantum state tomography.* 