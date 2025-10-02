# Older Iterations of Active Learning Hands-On Tutorial

These notebooks can be used to supplement some information and skills that can be used in developing your **Active Learning (AL)** campaign. They are a slower paced, more in depth guide on active learning, with the SDL tutorial remaining the same.

## Overview

The goal of this tutorial is to walk through the essentials of AL from start to finish, providing both conceptual explanations and hands-on code examples. A basic understanding of **Machine Learning (ML)** is assumed. If you're new to ML, we highly recommend reviewing:

- **A User's Guide to Machine Learning**: https://doi.org/10.1021/acspolymersau.2c00037  
- **Accompanying ML Colab notebook**: https://www.gormleylab.com/MLcolab

### Tutorial Overview

This guide is structured into three main notebooks, each building on the last:

---

#### [Tutorial Notebook 1: Bayesian Optimization in 1D](https://github.com/GormleyLab/SDLGuide/blob/main/Hands-on%20Tutorial/Notebook%201%20AL%20Tutorial.ipynb)  
We begin with a simple, one-dimensional example of **Bayesian Optimization (BO)** applied to a black-box function. This notebook introduces the foundational concepts of active learning, including:  
- Fitting a Gaussian Process (GP) to observed data  
- Using acquisition functions to choose new sampling points

---

#### [Tutorial Notebook 2: Deeper dive into Active Learning](https://github.com/GormleyLab/SDLGuide/blob/main/Hands-on%20Tutorial/Notebook%202%20AL%20Tutorial.ipynb)

Next, we extend the BO framework to higher-dimensional spaces. This notebook explores:

- Acquisition functions in complex domains
- The trade-off between exploitation (sampling where predictions are high) and exploration (sampling where uncertainty is high)
- Alternative ML models and their effects on BO
- Alternative BO algorithms and their effects on optimization efficiency

---

#### [Tutorial Notebook 3: Application to Real-World Data - Self-Driving Labs](https://github.com/GormleyLab/SDLGuide/blob/main/Hands-on%20Tutorial/Notebook%203%20AL%20Tutorial.ipynb)

Finally, we apply active learning to a **real experimental dataset** involving enzymes. This notebook demonstrates how active learning can be used to efficiently select informative experiments and accelerate discovery in scientific research. This notebook explores:

- Seed Library Generation (Initial sampling)
- Structure of Self-Driving Labs
- Closing the Loop

---