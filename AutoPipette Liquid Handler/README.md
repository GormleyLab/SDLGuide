# Self-Driving Lab based on ACRO System and Integra AutoPipette

<p align="center">
  <img width="400" alt="liquibot" src="/Images/liquibot.png" />
</p>

This project provides a graphical user interface (GUI) for automating enzyme assay experiments using a custom-designed liquid handler based on the ACRO system and Integra autopipette, as well as a SpectraMax plate reader. It enables users to design, dispense, and analyze experimental formulations with minimal manual intervention.

## Features

- GUI for experiment setup and control
- Automated liquid handling via OpenBuilds ACRO system
- Seed library generation using Latin Hypercube Sampling
- Automated data collection and analysis
- Machine learning-driven experiment optimization (Gaussian Process, Expected Improvement)
- CSV import/export for formulations and results
- Potential integration with SpectraMax plate reader

## Directory Structure

```
main.py
README.md
requirements.txt
data/
    GOx Assay data_bank.csv
resources/
    gl_logo.ico
    Tip Positions.csv
src/
    dataprocessing.py
    guifunctions.py
    liquidhandler.py
    sdlgui.py
    sdlvariables.py
```

## Getting Started

### Prerequisites

- Python 3.8+
- [requirements.txt](requirements.txt) dependencies
- Basic tools and comfort with mechanical assembly
- SpectraMax M2 plate reader and SDK (Optional)
- Chemyx Syringe Pump SDK
- Integra AutoPipette and a computer with Bluetooth capability

---
## Hardware Setup

For hardware assembly and configuration, please refer to the [AutoPipette Liquid Handler.pdf](../AutoPipette%20Liquid%20Handler%20Building%20Guide.pdf) found under the AutoPipette Liquid Handler directory. This guide provides step-by-step instructions for building and setting up the AutoPipette liquid handler.
<p align="center">
<img width="400" alt="Picture9" src="/Images/liquibot%20cad.png" />
</p>

---

### Installation

```sh
git clone <repo-url>
cd AutoPipette Liquid Handler/AutoPipette SDL Software
pip install -r requirements.txt
```

### Running the Program

```sh
python main.py
```

## Usage
There is no SOP for this software yet, but it is based on the original Pen Plotter Liquid Handler design. The Pen Plotter SDL SOP can be found [here](https://github.com/Pasta1107/Users-Guide-to-SDL/blob/main/Pen%20Plotter%20Liquid%20Handler/SDL_SOP.pdf).
- **Home Tab:** Input reagent names, concentrations, pH, and group assignments. Import custom CSVs if needed.
- **Enzyme Assay Tab:** Configure experiment parameters, bounds, and desired activity. Start automated DBTL (Design-Build-Test-Learn) cycles.

---

## Main Modules

- [`src/sdlgui.py`](src/sdlgui.py): Main GUI logic and layout
- [`src/guifunctions.py`](src/guifunctions.py): Experiment setup, seed library generation, dispensing instructions
- [`src/liquidhandler.py`](src/liquidhandler.py): Control the ACRO system, Integra pipette, and ChemyX Syringe pump controller
- [`src/dataprocessing.py`](src/dataprocessing.py): Data extraction, analysis, machine learning
- [`src/sdlvariables.py`](src/sdlvariables.py): Global variables and experiment state

## License

MIT License

## Authors:
Apostolos Maroulis, Dylan Waynor<br>
<img width="300" src="/Images/gllogo.png">

Last Updated ***
