> **Note**: The widgets in the user interface for notebook 2 do not render in the GitHub Preview. To view, you must download and run locally, or view the notebook in [Google Colab](https://colab.research.google.com/drive/1bBl2bTz1-Y4bTuPBKBZ19CbYj2Hosa-n?usp=sharing)
## Overview

The goal of this tutorial is to walk through the essentials of **Active Learning (AL)** from start to finish, providing both conceptual explanations and hands-on code examples. A basic understanding of **Machine Learning (ML)** is assumed. If you're new to ML, we highly recommend reviewing:

- **A User's Guide to Machine Learning**: https://doi.org/10.1021/acspolymersau.2c00037  
- **Accompanying ML Colab notebook**: https://www.gormleylab.com/MLcolab

### Tutorial Overview

This guide is structured into three main notebooks, each building on the last:

---

#### Tutorial Notebook 1: Principles of Active Learning  
We begin with a simple, two-dimensional example of **Bayesian Optimization (BO)** applied to a black-box function. This notebook introduces the foundational concepts of active learning, including:  
- Data seeding strategies
- Fitting a surrogate model (e.g., Gaussian process, random forest, neural network) to observed data  
- Using different acquisition functions to choose new sampling points

---

#### Tutorial Notebook 2: Application to Real-World Data - Self-Driving Labs

Here, we apply active learning to a **real experimental dataset** involving enzymes. This notebook demonstrates how active learning can be used to efficiently select informative experiments and accelerate discovery in scientific research. This notebook explores:

- Alternative data seeding strategies
- Structure of Self-Driving Labs
- Closing the Loop

---

## Authors:
Apostolos Maroulis, Dylan Waynor, Quinn Gallagher, Roshan Patel, Matthew Tamasi<br>
<img width="300" src="/Images/gllogo.png">

Last Updated 02/06/2026
Version 1.0.0
