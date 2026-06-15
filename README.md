# ADAM Optimization Suite in MATLAB

A comprehensive MATLAB framework evaluating the performance of the ADAM optimization algorithm. This repository scales from classic mathematical benchmarks to large-scale design variables and Artificial Neural Network (ANN) training. Developed as an extension for lecture materials.

## 🚀 Project Roadmap

*   **Phase 1: Benchmark Problems**
    *   Application to small-scale test functions (Beale, Rosenbrock, Himmelblau, etc.).
*   **Phase 2: Large-Scale Optimization**
    *   Scaling the ADAM optimizer to medium and large engineering problems (~100 design variables).
*   **Phase 3: ANN Training**
    *   Applying ADAM to train Artificial Neural Networks.

## 👥 Contributors

*   **Jasurbek Odilov** ([@Jacund96](https://github.com)) — Project Maintainer
*   **Usama Renders** ([@usama-469](https://github.com/usama-469)) — Core Collaborator (ANN Implementation Support)

## 📂 Repository Structure

*   `adam_optimizer.m` — Core implementation of the ADAM algorithm.
*   `adam_runner_*.m` — Execution scripts for specific benchmark functions.
*   `visualize.m` — Visualization and convergence plotting tools.

## 🛠️ Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR-REPO-NAME.git
   ```
2. Open MATLAB.
3. Run any runner script (e.g., `adam_runner_Beale_func.m`) to view the optimization path.
