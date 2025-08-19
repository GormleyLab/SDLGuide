<p align="center">
  <img width="900" alt="readme home" src="https://github.com/user-attachments/assets/a6531e94-a821-4ddb-b5ea-00947e8dab23" />
</p>

# Overview:
This repository contains the instructions/code/files for the assembly and operations of a [Pen Plotter Driven Liquid Handler](/Pen%20Plotter%20Liquid%20Handler) and 
[Pipette Driven Liquid Handler](https://github.com/Pasta1107/Pipette-Liquid-Handler) as well as a collection of hands-on tutorial notebooks. Both devices are designed as low-cost automation tools to produce self-driving labs (SDLs). The pen plotter-driven liquid handler is created using rigid pre-built spatial components that are modified to fit a user-developed fluidic system. The pipette-driven liquid handler is designed using modular components to provide the most flexibility, paired with a wireless autopipette.  

---

## Tutorial Overview
The goal of the provided tutorial is to walk through the essentials of AL from start to finish, providing both conceptual explanations and hands-on code examples. A basic understanding of ML is assumed. If you're new to ML, we highly recommend reviewing [Part 1 of our User’s Guide series](https://doi.org/10.1021/acspolymersau.2c00037).
This guide is structured into three main tutorial notebooks, each building on the last for a smooth learning experience. Major points of interest from the guide provided are discussed below. 
### [Tutorial Notebook 1: Bayesian Optimization in 1D](/Hands-on%20Tutorial/Section%201%20AL%20Tutorial_8_5_25.ipynb)  
We begin with a simple, one-dimensional example of BO applied to a black-box function. This notebook introduces the foundational concepts of AL, including but not limited to: fitting a GP to observed data and using acquisition functions to choose new sampling points.
### [Tutorial Notebook 2: Deeper dive into Active Learning](/Hands-on%20Tutorial/Section%202%20AL%20Tutorial_8_5_25.ipynb)
Next, we extend the BO framework to higher-dimensional spaces. This notebook explores: acquisition functions in complex domains, the trade-off between exploitation (sampling where predictions are high) and exploration (sampling where uncertainty is high), alternative ML models and their effects on BO, alternative BO algorithms, and their effects on optimization efficiency.
### [Tutorial Notebook 3: Application to Real-World Data - Self-Driving Labs](/Hands-on%20Tutorial/Section%203%20AL%20Tutorial_8_5_25.ipynb)
Finally, we apply AL to a real experimental dataset involving enzymes. This notebook demonstrates how active learning can be used to efficiently select informative experiments and accelerate discovery in scientific research. This notebook explores: seed library generation (initial sampling) and closing the loop in the context of an SDL.

---

## Abstract
Experimentation is inherently difficult because most methods require optimization before final data can confidently be collected. In most cases, experiment design spaces have multiple variables (i.e., high dimensionality), thus requiring their simultaneous multi-objective  optimization optimization for single and multi-objective targets. Trial-and-error experimentation guided by rational decision making is the main process for optimizing experiments, but this method becomes increasingly difficult as complex interactions between inputs limit our ability to capture underlying trends using traditional statistical approaches. To empower researchers with tools that accelerate productivity, we believe active learning/machine learning (AL/ML) combined with automation will be indispensable tools in future labs. However, a steep initial learning curve and high instrument costs pose significant barriers for early adopters of these powerful tools. To democratize access, here we provide Part 2 of our User’s Guide Series with a comprehensive tutorial for the development of AL/ML skills with insight geared towards implementation of self-driven workflows. The provided open-source, low-cost liquid handling platforms act as templates for researchers beginning to incorporate self-driving lab (SDL) methodologies into their workflows.

---

## Authors:
Apostolos Maroulis, Dylan Waynor<br>
<img width="300" src="/Images/gllogo.png">

Last Updated ***
