<p align="center">
  <img width="900" alt="readme home" src="Images/readme home.png" />
</p>

# Overview
Self-driving laboratories (SDLs) address a common bottleneck in experimentation: time and effort required to explore large parameter spaces. By combining automated experimentation with machine learning (ML) and Bayesian optimization (BO), an SDL can iteratively design experiments and learn from resulting data in a closed-loop, fully autonomous workflow.

This repository contains the instructions/code/files for the assembly and operations of a [Pen Plotter Driven Liquid Handler](/Pen%20Plotter%20Liquid%20Handler) and 
[Pipette Driven Liquid Handler](https://github.com/GormleyLab/Pipette-Liquid-Handler) as well as a collection of hands-on tutorial notebooks for active learning. Both devices are designed as low-cost automation tools to produce SDLs. The pen plotter-driven liquid handler is created using rigid pre-built spatial components that are modified to fit a user-developed 
fluidic system. The pipette-driven liquid handler is designed using modular components to provide the most flexibility, paired with a wireless autopipette.

In addition to hardware documentation, this repository includes a representative dataset generated using the SDL pipeline implemented here. The dataset was produced fully autonomously, and is used as an example to demonstrate how the design-build-test-learn (DBTL) workflow operates in practice, from experimental design through surrogate model training.
We use the optimization of a glucose oxidase (GOx) enzyme assay, which are common in wet-lab environments and are well suited as a representative example. 

---

## Repository Structure

Use this diagram to navigate the repository:

```
├── Hands-on Tutorial/           # AL and SDL tutorial notebooks and example data
│   ├── Notebook 1 - AL Tutorial.ipynb    # Principles of Active Learning
│   ├── Notebook 2 - SDL Tutorial.ipynb   # Real-world SDL application (enzyme assay)
│   ├── GOx Assay data_bank_renamed.csv  # Example GOx assay data for tutorials
│   ├── Older Iterations/               # Previous notebook versions
│   ├── requirements.txt                # Python dependencies for tutorials
│   └── README.md
│
├── Pen Plotter Liquid Handler/   # AxiDraw-based liquid handler (hardware + software)
│   ├── Pen Plotter SDL Software/ # GUI and automation software
│   │   ├── data/
│   │   │   └── GOx Assay data_bank.csv  # Validation dataset example
│   │   ├── src/
│   │   │   ├── sdlgui.py         # Main GUI layout and logic
│   │   │   ├── guifunctions.py   # Experiment setup, seed library, dispensing
│   │   │   ├── liquidhandler.py  # AxiDraw and syringe pump control
│   │   │   ├── dataprocessing.py # Data extraction, ML/AL integration
│   │   │   └── sdlvariables.py   # Global variables and experiment state
│   │   ├── resources/            # Icons and assets
│   │   ├── main.py               # Entry point (runs the GUI)
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── SDL CAD Files/                  # 3D printable parts (STL, STEP)
│   ├── Pen Plotter LH Build Guide.pdf  # Building instructions for hardware
│   ├── SDL_SOP.pdf                     # Standard operating procedure
│   └── README.md
│
├── Pipette Liquid Handler/       # Wireless autopipette-based liquid handler (External Repo)
│   ├── Pipette SDL Software/    # GUI and automation software
│   │   ├── data/
│   │   │   └── GOx Assay data_bank.csv  # Validation dataset example
│   │   ├── src/
│   │   │   ├── sdlgui.py         # Main GUI layout and logic
│   │   │   ├── guifunctions.py   # Experiment setup, seed library, dispensing
│   │   │   ├── liquidhandler.py  # Pipette and pump control
│   │   │   ├── dataprocessing.py # Data extraction, ML/AL integration
│   │   │   └── sdlvariables.py   # Global variables and experiment state
│   │   ├── resources/            # Icons, tip positions, RemoteControl
│   │   ├── main.py               # Entry point (runs the GUI)
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── Config Files/            # G-code and controller config (config.g, homeall.g, etc.)
│   ├── Pipette CAD Files/       # 3D design files (needle holder, vial rack, etc.)
│   ├── WIP Controller Program/   # Work-in-progress controller script alternatives
│   ├── Pipette Liquid Handler Building Guide.pdf
│   ├── Pipette Liquid Handler Parts List.xlsx
│   └── README.md
│
├── Images/                      # Figures for README and docs
├── index.html                    # Project homepage
├── LICENSE
└── README.md                     # This file
```

**Quick links:** [Tutorial Notebook 1 (AL)](/Hands-on%20Tutorial/Notebook%201%20-%20AL%20Tutorial.ipynb) · [Tutorial Notebook 2 (SDL)](/Hands-on%20Tutorial/Notebook%202%20-%20SDL%20Tutorial.ipynb) · [Pen Plotter Liquid Handler](/Pen%20Plotter%20Liquid%20Handler) · [Pipette Liquid Handler](https://github.com/GormleyLab/Pipette-Liquid-Handler)

---

## Tutorial Overview
The goal of the provided tutorial is to walk through the essentials of AL from start to finish, providing both conceptual explanations and hands-on code examples. A basic understanding of ML is assumed. If you're new to ML, we highly recommend reviewing [Part 1 of our User’s Guide series](https://doi.org/10.1021/acspolymersau.2c00037).
This guide is structured into three main tutorial notebooks, each building on the last for a smooth learning experience. Major points of interest from the guide provided are discussed below. 
### [Tutorial Notebook 1: Principles of Active Learning ](/Hands-on%20Tutorial/Notebook%201%20-%20AL%20Tutorial.ipynb)  
We begin with a simple, two-dimensional example of **Bayesian Optimization (BO)** applied to a black-box function. This notebook introduces the foundational concepts of active learning, inccluding data seeding, fitting surrogate models to data, and applying various acquisition functions to choose new sampling points.

### [Tutorial Notebook 2: Application to Real-World Data - Self-Driving Labs](/Hands-on%20Tutorial/Notebook%202%20-%20SDL%20Tutorial.ipynb)
Finally, we apply AL to a real experimental dataset involving enzymes. This notebook demonstrates how active learning can be used to efficiently select informative experiments and accelerate discovery in scientific research. This notebook explores: seed library generation (initial sampling) and closing the loop in the context of an SDL.

---

## Abstract
Experimentation is inherently difficult because most methods require substantial refinement, calibration, and validation before high-quality, reliable data can be collected. In most cases, experimental domains have multiple variables (i.e., high dimensionality), thus requiring their simultaneous optimization for single and multi-objective targets. Traditional experimental approaches rely on trial-and-error methods guided by rational decision making, but these become increasingly inefficient and ineffective as complex interactions between inputs limit our ability to capture underlying trends using conventional statistical approaches. Active learning and machine learning (AL/ML) combined with automation represents an indispensable approach for future laboratory productivity. However, a steep initial learning curve and high costs of instrumentation pose substantial barriers to adopt this powerful approach. To democratize access, we provide a comprehensive tutorial covering both the computational skills and hardware implementation necessary for self-driven experimental workflows. The accompanying open-source, low-cost liquid handling platforms offer practical templates for researchers adopting self-driving lab (SDL) methodologies.

---

## Authors:
Apostolos Maroulis, Dylan Waynor<br>
<img width="300" src="/Images/gllogo.png">

Last Updated 01/29/2026
Version 1.0.0
