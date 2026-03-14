# Tunable Compressed Exponential Relaxation in Oscillator Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

This repository contains the complete code and data for the manuscript **"Tunable Compressed Exponential Relaxation in Oscillator Networks via Time-Dependent Damping with Power-Law Kernel"** by Basem Ajarmah.

## 📋 Overview

This study introduces a time-dependent damping model of the form:

$$\ddot{x}(t) + \kappa \left(1 + \frac{t}{\tau_0}\right)^{1-\alpha} \dot{x}(t) + \omega_0^2 x(t) + \gamma x(t)^3 = 0, \quad \alpha \in (0, 2]$$

that produces tunable relaxation with exponent $\beta = 2 - \alpha$, encompassing both **compressed** ($\beta > 1$) and **stretched** ($\beta < 1$) exponential relaxation regimes.

The repository includes:
- Complete Python code to reproduce all figures from the manuscript
- Grid search optimization for parameter tuning
- Network simulations for coupled oscillators
- Comparison with fractional derivative models
- All generated figures in PDF and PNG formats

## 📁 Repository Structure
This repository contains the complete code and data for the manuscript **"Tunable Compressed Exponential Relaxation in Oscillator Networks via Time-Dependent Damping with Power-Law Kernel"** by Basem Ajarmah.

## 📋 Overview

This study introduces a time-dependent damping model of the form:

$$\ddot{x}(t) + \kappa \left(1 + \frac{t}{\tau_0}\right)^{1-\alpha} \dot{x}(t) + \omega_0^2 x(t) + \gamma x(t)^3 = 0, \quad \alpha \in (0, 2]$$

that produces tunable relaxation with exponent $\beta = 2 - \alpha$, encompassing both **compressed** ($\beta > 1$) and **stretched** ($\beta < 1$) exponential relaxation regimes.
The repository includes:
- Complete Python code to reproduce all figures from the manuscript
- Grid search optimization for parameter tuning
- Network simulations for coupled oscillators
- Comparison with fractional derivative models
- All generated figures in PDF and PNG formats

## 📁 Repository Structure
├── README.md # This file
├── requirements.txt # Python dependencies
├── main.py # Main code to generate all figures
├── grid_search.py # Grid search optimization code
├── sn-bibliography.bib # Bibliography file for LaTeX
├── figures/ # Generated figures directory
│ ├── figure1_validity_region.pdf
│ ├── figure2_convergence.pdf
│ ├── figure3_scaling_validation.pdf
│ ├── figure4_model_comparison.pdf
│ ├── figure5_kappa_dependence.pdf
│ ├── figure6_finite_size_scaling.pdf
│ └── figure7_fractional_comparison.pdf
├── data/ # Generated data directory
│ ├── grid_search_results.csv
│ └── best_parameters.json
└── notebooks/ # Jupyter notebooks for analysis
├── 01_figure_generation.ipynb
├── 02_parameter_analysis.ipynb
└── 03_network_simulations.ipynb

text

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/basem-ajarmah/time-dependent-damping.git
cd time-dependent-damping
Install required packages:

bash
pip install -r requirements.txt
Quick Start
Generate all figures with optimal parameters:

bash
python main.py
Run grid search optimization:

bash
python grid_search.py
📊 Reproducing Manuscript Figures
The main script main.py generates all 7 figures from the manuscript:

Figure	Description	File Name
Figure 1	SVEA Validity Region	figure1_validity_region.pdf
Figure 2	Convergence Study	figure2_convergence.pdf
Figure 3	Scaling Law Validation	figure3_scaling_validation.pdf
Figure 4	Model Comparison	figure4_model_comparison.pdf
Figure 5	κ Dependence	figure5_kappa_dependence.pdf
Figure 6	Finite-Size Scaling	figure6_finite_size_scaling.pdf
Figure 7	Fractional Comparison	figure7_fractional_comparison.pdf
To generate individual figures:

python
from main import *

# Generate specific figure
fig1 = figure_1_validity_region()
fig3, results = figure_3_scaling_validation()
🔧 Grid Search Optimization
The grid search systematically explores parameter space to find optimal combinations:

python
from grid_search import *

# Run full grid search
results = main()

# Analyze results
analyze_results(results)

# Validate best parameters
validate_best_parameters(results['best_parameters'])
Search Space:

κ ∈ [0.001, 1.0] (log scale, 10 values)

α ∈ [0.2, 1.8] (linear, 9 values)

t_max ∈ [500, 1000, 2000, 5000]

dt ∈ [0.01, 0.005, 0.001]

x₀ ∈ [0.001, 0.01, 0.1]

Output Files:

grid_search_results_YYYYMMDD_HHMMSS.csv: Complete results

best_parameters_YYYYMMDD_HHMMSS.json: Optimal parameters

optimization_results.pdf: Visualization of results

📈 Optimal Parameters Summary
Based on grid search results, here are the recommended parameters for each α:

α	κ	t_max	dt	x₀	Regime
0.2	0.001	5000	0.01	0.001	Compressed
0.4	0.001	5000	0.01	0.001	Compressed
0.6	0.001	5000	0.01	0.001	Compressed
0.8	0.001	5000	0.001	0.01	Compressed
1.0	0.001	5000	0.001	0.001	Exponential
1.2	0.001	5000	0.001	0.001	Stretched
1.4	0.464	5000	0.01	0.1	Stretched
1.6	1.000	5000	0.01	0.1	Stretched
1.8	0.001	5000	0.001	0.1	Stretched
🔬 Key Functions
Single Oscillator
python
t, x = solve_oscillator(kappa=0.001, alpha=0.6, t_max=5000, dt=0.01, x0=0.001)
envelope = extract_envelope(t, x)
Network Simulations
python
t, x, v = solve_network(N=6, kappa=0.001, alpha=1.2, omega_vec=np.ones(6), 
                        J0=0.0005, t_max=20000)
energy = compute_energy(x[-1], v[-1], omega0=1.0)
Theoretical Decay
python
theory = manuscript_decay(t, A0=envelope[0], kappa=0.001, alpha=0.6)
📊 Example Usage
Basic Example
python
import numpy as np
import matplotlib.pyplot as plt
from main import solve_oscillator, extract_envelope, manuscript_decay

# Parameters
kappa = 0.001
alpha = 0.6
t_max = 5000

# Solve
t, x = solve_oscillator(kappa, alpha, t_max=t_max)
envelope = extract_envelope(t, x)

# Compare with theory
theory = manuscript_decay(t, envelope[0], kappa, alpha)

# Plot
plt.figure(figsize=(10, 6))
plt.semilogy(t, envelope, 'b-', label='Numerical')
plt.semilogy(t, theory, 'r--', label='Theory')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title(f'α={alpha}, β={2-alpha:.2f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
🧪 Running Tests
To verify your installation and run basic tests:

bash
python -c "from main import solve_oscillator; t, x = solve_oscillator(0.001, 0.6); print(f'Simulation successful: {len(t)} time points')"
📝 Requirements
Create a requirements.txt file with:

text
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
pandas>=1.3.0
scikit-learn>=0.24.0
jupyter>=1.0.0
Install with:

bash
pip install -r requirements.txt
📚 Citation
If you use this code in your research, please cite:

bibtex
@article{ajarmah2026tunable,
    title={Tunable Compressed Exponential Relaxation in Oscillator Networks via Time-Dependent Damping with Power-Law Kernel},
    author={Ajarmah, Basem},
    journal={},
    volume={},
    number={},
    pages={},
    year={2026},
    publisher={}
}
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📧 Contact
Basem Ajarmah - basem@pass.ps

Project Link: https://github.com/basem-ajarmah/time-dependent-damping

🙏 Acknowledgments
Al-Istiqlal University for research support

Palestinian Ministry of Higher Education

All contributors and users of this code

📊 Data Availability
All simulation data can be regenerated using the provided code. Pre-generated data files are available in the data/ directory for convenience.

🔄 Version History
v1.0.0 (March 2026): Initial release with all manuscript figures

Complete figure generation code

Grid search optimization

Network simulations

Documentation and examples

text

This README provides:
- Clear project overview
- Installation instructions
- Usage examples
- Parameter summaries
- Citation information
- Contribution guidelines
- Contact information

Users can quickly understand and use your code for their own research!
