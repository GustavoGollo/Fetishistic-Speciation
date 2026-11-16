# Fetishistic-Speciation
An Individual-Based Model (IBM) in Python simulating fetishistic sympatric speciation via mutual mate choice, including a GUI and statistical analysis pipeline.
# Fetishistic Speciation Model

This repository contains the complete computational pipeline developed to investigate sympatric speciation driven by the emergence of idiosyncratic mating preferences ("fetishes"). The model demonstrates how mutual mate choice and behavioral isolation can fragment a panmictic population even in the absence of geographic barriers.

The code is structured into three main Python scripts:

## Repository Structure

*   **`spec_core_islands.py`**: The core simulation engine. It handles the individual-based genetic model (Desire and Phenotype genomes), mating mechanics, mutation, and population dynamics. It also performs phenotypic clustering in real-time using PCA and DBSCAN, and logs the pedigree data.
*   **`IslandGUI.py`**: A Graphical User Interface (GUI) built with Tkinter. It allows users to configure simulation parameters (e.g., population size, mutation rate, mating thresholds) and visualize the simulation dynamics in real-time.
*   **`IslandAnalysis.py`**: The statistical analysis and visualization pipeline. It runs multiple independent replicates (seeds), processes the output data, and generates the final plots and statistical tests presented in the manuscript (e.g., Kinship matrices, PCA phenotypic space, reproductive isolation, Mann-Whitney tests).

## Requirements

Ensure you have Python 3.8+ installed. The required libraries can be installed via pip:

```bash
pip install numpy pandas matplotlib scipy scikit-learn seaborn
