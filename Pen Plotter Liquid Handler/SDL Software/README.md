# Self-Driving Lab based on AxiDraw and Syringe Pump
<p align="center">
<img width="500" alt="axidraw" src="/Images/axidraw.png" />
</p>

This project provides a graphical user interface (GUI) for automating enzyme assay experiments using a custom-built AxiDraw pen plotter liquid handler and a SpectraMax M2plate reader. It enables users to design, dispense, and analyze experimental formulations with minimal manual intervention.

## Features

- GUI for experiment setup and control
- Automated liquid handling via AxiDraw
- Integration with SpectraMax plate reader
- Seed library generation using Latin Hypercube Sampling
- Automated data collection and analysis
- Machine learning-driven experiment optimization (Gaussian Process, Expected Improvement)
- Custom CSV import/export for formulations and results

## Directory Structure

```
main.py
README.md
requirements.txt
data/
    GOx Assay data_bank.csv
resources/
    gl_logo.ico
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
- AxiDraw hardware and drivers
- SpectraMax plate reader and SDK
- ChemyX Syringe Pump SDK

### Installation

```sh
git clone <repo-url>
cd SDL_6_3_25
pip install -r requirements.txt
```

### Running the Program

```sh
python main.py
```

## Usage

- **Home Tab:** Input reagent names, concentrations, pH, and group assignments. Import custom CSVs if needed.
- **Enzyme Assay Tab:** Configure experiment parameters, bounds, and desired activity. Start automated DBTL (Design-Build-Test-Learn) cycles.
- **Liquid Handler Tab:** Manually control AxiDraw for dispensing. (WIP)
- **Manual Tab:** Directly move AxiDraw and adjust pen/syringe settings.

---

### [Standard Operating Procedure (SOP)](SDL_SOP.pdf)

#### Self-Driven Lab: Custom Liquid Handler SOP

This SOP provides relevant information necessary for the overall viability and use of our device. It is reliant on basic understanding of other supplemental information: References to device setup will be attributed to the system build guide. References to ML/AL will be attributed to the tutorial notebooks. Some constraints of the device are in place to balance price effectiveness and system performance to provide a low-cost example of a liquid handler. The following is a quick start guide. It is highly recommended to follow the linked SOP above for an in-depth guide.

---

#### 1. Hardware Setup

To adapt the AxiDraw Pen Plotting machine for liquid handling, several modifications are required.  
Refer to the [Pen Plotter LH Build Guide](https://github.com/Pasta1107/Users-Guide-to-SDL/blob/main/Pen%20Plotter%20Liquid%20Handler/Pen%20Plotter%20LH%20Build%20Guide.pdf) for a detailed walkthrough of the necessary mechanical and hardware changes.  

1. **3D print the following custom parts**:  
   - AxiDraw Needle Holder  
   - AxiDraw Well Plate Platform  
   - Large Reservoir Insert  
2. **Modify** the AxiDraw hardware as outlined in the guide.  
3. **Assemble the fluidic loop** and connect it to both the Chemyx syringe pump and the AxiDraw.  
---

#### 2. Software Setup

- **AxiDraw:**  
  [AxiDraw Python API](https://axidraw.com/doc/py_api/#introduction)  
  If you haven't already, install:  
  `python -m pip install https://cdn.evilmadscientist.com/dl/ad/public/AxiDraw_API.zip`
- **Chemyx Syringe Pump:**  
  [Chemyx Python Setup Guide](https://chemyx.com/resources/knowledge-base/general-syringe-pump-info/computer-control-programs/python-program-installation-set-up-for-chemyx-fusion-200x-syringe-pump/)

---

#### 3. Preparation For Experimentation

**3.1 Calibration Protocol:**  
- Syringe Pump Calibration: Dispense triplicate known volumes, measure mass, and calibrate for loss.
- X/Y calibration: Use Manual UI to ensure coordinates for wells, waste, and reagents are accurate.
---

**3.2 Initial Setup**

- Fill the loop with water, minimizing air bubbles.
- Align the device with the Spectramax M2 by first raising the device to the appropriate height, and then ensuring the drawer opening just clears the cutout in the AxiDraw platform.
- Fill reservoirs 1 and 3 with DI water, leaving reservoir 2 (the middle) empty for waste.

---

**3.3 Experimental Setup**

- **Experiment Configuration:**
    - Prepare reagent list with concentrations and pHs.
    - Set mixing order and concentration bounds.
    - Define desired activity (ΔOD).
    - Ensure total dispensing volume per reagent is under 1.6ml.
    - Group reagents as needed; buffers can have different types and pHs.

---

**4.3 Notes on Experimentation**

- **Reagent Order:**  
  Dispense substrate last to control reaction start.
- **Locations:**  
  Log positions for reagents, waste, storage, and cleaning troughs in the script, as well as the pen down position to be at or near the bottom of each well without putting pressure.
- **Compatibility:**  
  Check tubing and luer compatibility for solvents and biologics. Adjust calibration for viscous materials.
- **Cleaning Protocol:**  
  Flush system after use, replace disposable components, and reprime as needed.

---

Thank you for trying to use our device as a platform for a SDL.

---

## Main Modules

- [`src/sdlgui.py`](src/sdlgui.py): Main GUI logic and layout
- [`src/guifunctions.py`](src/guifunctions.py): Experiment setup, seed library generation, dispensing instructions
- [`src/liquidhandler.py`](src/liquidhandler.py): AxiDraw and ChemyX pump control
- [`src/dataprocessing.py`](src/dataprocessing.py): Data extraction, analysis, machine learning
- [`src/sdlvariables.py`](src/sdlvariables.py): Global variables and experiment state

## License

MIT License

## Authors:
Apostolos Maroulis,
Dylan Waynor

<img width="500" height="238" alt="image-asset" src="https://github.com/user-attachments/assets/dc089951-90d0-4818-8117-935856d4576b" />

Last Updated ***
