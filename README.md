# DrivAerNet
DrivAerNet: A Parametric Car Dataset for Data-driven Aerodynamic Design and Graph-Based Drag Prediction

## Introduction
DrivAerNet is a large-scale, high-fidelity CFD dataset of 3D industry-standard car shapes designed for data-driven aerodynamic design. It comprises 4000 high-quality 3D car meshes and their corresponding aerodynamic performance coefficients, alongside full 3D flow field information.

## Dataset Details
- **3D Car Meshes**: 4000 designs with 0.5 million elements each.
- **Aerodynamic Coefficients**: Includes Cd, Cl, Clr, Clf, and Cm.
- **CFD Simulation Data**: Full 3D pressure, velocity fields, and wall-shear stresses computed using 8 million mesh elements.

## Parametric Model 
The DrivAerNet dataset includes a parametric model of the DrivAer fastback, developed using ANSA® software to enable extensive exploration of automotive design variations. This model is defined by 50 geometric parameters, allowing the generation of 4000 unique car designs through Optimal Latin Hypercube sampling and the Enhanced Stochastic Evolutionary Algorithm. 
<p align="center">
  <img src="https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/ed7e825a-db41-4230-ac91-1286c69d61fe" width="400" alt="DrivAerNet Morphing">
</p>

<p align="center">
  <img src="https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/f6af36aa-079b-49d9-8ac7-a6b20595faee" width="400" alt="DrivAerNet Analysis">
</p>


## CFD Data
![Prsentation4-ezgif com-crop](https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/3d5e3b3e-4dcd-490f-9936-2a3dbda1402b)

## Car Designs
The DrivAerNet dataset specifically concentrates on conventional car designs, highlighting the significant role that minor geometric modifications play in aerodynamic efficiency. This focus enables researchers and engineers to explore the nuanced relationship between car geometry and aerodynamic performance, facilitating the optimization of vehicle designs for improved efficiency and performance.

https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/86b8046f-8858-4193-a904-f80cc59544d0

## RegDGCNN: Dynamic Graph Convolutional Neural Network for Regression Tasks


## Usage Instructions
The dataset and accompanying Python scripts for data conversion are available at [GitHub repository link].

## Contributing
We welcome contributions to improve the dataset or project. Please submit pull requests for review.

## Maintenance and Support
Maintained by the DeCoDE Lab at MIT. Report issues via [GitHub issues link].

## License
Distributed under the Creative Commons Attribution (CC BY) license. Full terms [here](https://creativecommons.org/licenses/by/4.0/deed.en).

## Citations
Please cite the DrivAerNet dataset in your publications as: [Citation details].

## Additional Resources
- Tutorials: [Link]
- Technical Documentation: [Link]





