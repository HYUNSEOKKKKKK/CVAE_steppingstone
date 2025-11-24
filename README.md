# CVAE-based Stepping Stone Generator

This repository contains a Variational Autoencoder (VAE) model designed to learn and generate stepping-stone sequences for footstep planning research.  
The model takes in terrain/stepping-stone data and produces new, plausible sequences that can be used for planning or training.

---

## 1. Environment Setup

This project uses a conda environment.  
You can install all dependencies using the provided `environment.yml` file.

### **Create and activate the environment**

```bash
conda env create -f environment.yml
conda activate vae_steppingstone
```

This installs:
- Python 3.10  
- PyTorch (+ torchvision)  
- NumPy, Matplotlib  
- Additional scientific computing dependencies

---

## 2. Project Structure

```
CVAE_steppingstone/
├── environment.yml        
├── main.py                # Entry point for VAE generation
└── README.md
```

---

## 3. How to Run

```bash
python main.py
```

This will make new dataset `vae_terrain.json`  

---

