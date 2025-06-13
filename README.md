# Thesis Repository: Evaluating the Impact of 3D Geometry in GNNs for Molecular Property Prediction

This repository contains the experimental framework, data processing logic, and results for the master's thesis _"An Empirical Study: Evaluating the Impact of 3D Geometry in GNNs for Molecular Property Prediction."_ The goal of this project is to investigate the robustness and generalizability of 3D-augmented graph neural networks (GNNs) using large-scale benchmarking.

## Repository Structure

- `3DInfomax/` — Cloned and modified implementation of 3DInfomax including our integrated experiments.
- `GraphMVP/` — Cloned and modified implementation of GraphMVP. Includes our reimplementation of GraphCL and integrated experiments.
- `results/` — Aggregated results and evaluation outputs.

## Reproducibility and Reimplementations

To ensure fair and reproducible comparisons:
- We reimplemented **GraphCL** in both the 3DInfomax and GraphMVP repositories using consistent hyperparameters and datasets.
- We containerized each repository using **Docker** to ensure environment consistency, especially across GPU drivers and library versions. All docker images are available at https://hub.docker.com/repositories/niklasbille
- Deterministic settings were enforced wherever possible. See respective README files in each subfolder for environment setup and training commands.

## Experimental Modifications

### Noise Injection
We introduced feature-level flip perturbations to assess robustness. Noise levels were applied per atom and bond feature, using value dictionaries built from the dataset.

### Custom Splitting
To test generalizability, we implemented support for **custom scaffold and random splits** at given proportions.

## Experiments

We ran ~2,600 fine-tuning runs:
- **Models**: 3DInfomax, GraphMVP, GraphCL (x2)
- **Datasets**: 11 MoleculeNet datasets
- **Configurations**: Noise levels (η ∈ {0.05, 0.1, 0.2}), Split strategies (scaffold/random) at train porportions 0.6, 0.7, and 0.8
- **Seeds**: 6 per run (10 for regression)

Total GPU usage: ~2,600 hours on **NVIDIA H100s** via the MeluXina supercomputer.

##  Logging and Outputs

Each run logs:
- `TensorBoard` training curves
- Final metric results per split
- Best model checkpoint (lowest validation loss)

## 📚 Additional Information
- See individual READMEs in `3DInfomax/` and `GraphMVP/` for detailed setup, training, and evaluation instructions.

## 🔗 Links
- 3DInfomax (original): https://github.com/HannesStark/3DInfomax  
- GraphMVP (original): https://github.com/chao1224/GraphMVP  
- MeluXina: https://luxprovide.lu/meluxina
