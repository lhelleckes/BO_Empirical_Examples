# BO_Empirical_Examples

[![DOI](https://zenodo.org/badge/placeholder.svg)](https://zenodo.org/doi/10.5281/zenodo.xxxxxxxx)

Accompanying repository for the review manuscript  
**"A Guide to Bayesian Optimization in Bioprocess Engineering"**  
by Maximilian Siska, Emma Pajak, Katrin Rosenthal, Eric von Lieres, Antonio del Rio
Chanona and Laura Marie Helleckes.

This repository provides supplementary code, empirical examples, and figures for the review. It is designed to make the concepts of Bayesian optimization (BO) in bioprocess engineering more accessible and to serve as a practical reference for implementation and experimentation.

## Structure

The repository includes:

- **`notebooks/`** – Interactive notebooks containing empirical examples discussed in the review.
- **`Figures/`** – All figures used in the review paper.
- **`code/`** – Additional Python modules and utilities used for model construction and visualization.

## Empirical Examples

Two practical BO examples are included in this repository:

### 1. `simple_example.ipynb` – Crude Extract Case Study (BoTorch)

A minimal working example showcasing the BO workflow using BoTorch.

In this example, a crude extract is known to produce a target compound in measurable quantities. The enzymes and pathways responsible for its biosynthesis are unknown. The goal is to optimize the pH to maximize product yield.

This notebook highlights a typical BO iteration and intentionally demonstrates the consequences of a poorly fitted Gaussian process model to aid learning.

### 2. `hybrid_model_pymc.ipynb` – Enzyme Kinetics Case Study (PyMC)

A more advanced surrogate modeling case implemented in PyMC.

Here, we model the activity of a known enzyme using a hybrid surrogate model that combines a Gaussian process with a mechanistic first-order kinetic term. This example also shows how to explicitly model batch effects in the surrogate to better reflect bioprocess variability.

This notebook is intended for readers interested in advanced surrogate modeling and flexible model design.

## Figures

The `Figures/` folder contains all figures used in the review, including:

- Schematic workflows
- Gaussian process model illustrations
- Acquisition functions
- Visualizations of empirical optimization runs

## License

This repository is licensed under the [GNU Affero General Public License v3.0](https://github.com/lhelleckes/BO_Empirical_Examples/blob/main/LICENSE.md).  
You are free to use, modify, and distribute the material, provided that derivative works are also shared under the same license.

## Citation

To cite this repository or reuse its content in your own work, please refer to the latest release:  
➡ [Zenodo Link](https://doi.org/10.5281/zenodo.xxxxxxxx)
