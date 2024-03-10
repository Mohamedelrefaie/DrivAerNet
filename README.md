# DrivAerNet
DrivAerNet: A Parametric Car Dataset for Data-driven Aerodynamic Design and Graph-Based Drag Prediction

## Introduction
DrivAerNet is a large-scale, high-fidelity CFD dataset of 3D industry-standard car shapes designed for data-driven aerodynamic design. It comprises 4000 high-quality 3D car meshes and their corresponding aerodynamic performance coefficients, alongside full 3D flow field information.

## Dataset Details & Contents

The DrivAerNet dataset is meticulously crafted to serve a wide range of applications from aerodynamic analysis to the training of advanced machine learning models for automotive design optimization. It includes:

- **3D Car Meshes**: A total of **4000 designs**, showcasing a variety of conventional car shapes and emphasizing the impact of minor geometric modifications on aerodynamic efficiency. The 3D meshes and aerodynamic coefficients consume about **84GB**,
- **Aerodynamic Coefficients**: Each car model comes with comprehensive **aerodynamic performance coefficients** including drag coefficient (Cd), total lift coefficient (Cl), front lift coefficient (Clf), rear lift coefficient (Clr), and moment coefficient (Cm).
- **CFD Simulation Data**: The raw dataset, including full 3D pressure, velocity fields, and wall-shear stresses, computed using **8 million mesh elements** for each car design has a total size of around **16TB**.
- **Curated CFD Simulations**: For ease of access and use, a **streamlined version of the CFD simulation data** is provided, refined to include key insights and data, reducing the size to approximately **1TB**. 

This rich dataset, with its focus on the nuanced effects of design changes on aerodynamics, provides an invaluable resource for researchers and practitioners in the field.



## Parametric Model 
The DrivAerNet dataset includes a parametric model of the DrivAer fastback, developed using ANSA® software to enable extensive exploration of automotive design variations. This model is defined by 50 geometric parameters, allowing the generation of 4000 unique car designs through Optimal Latin Hypercube sampling and the Enhanced Stochastic Evolutionary Algorithm. 

DrivAerNet dataset incorporates a wide range of geometric modifications, including changes to side mirror and muffler positions, windscreen and rear window dimensions, engine undercover size, front door and fender offsets, hood placement, headlight scale, overall car length and width, upper and underbody scaling, and key angles like the ramp, diffusor, and trunk lid angles, to thoroughly investigate their impacts on car aerodynamics.
![DrivAerNetMorphingNew2-ezgif com-crop](https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/ed7e825a-db41-4230-ac91-1286c69d61fe)

![ezgif-7-2930b4ea0d](https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/f6af36aa-079b-49d9-8ac7-a6b20595faee)


## Car Designs
The DrivAerNet dataset specifically concentrates on conventional car designs, highlighting the significant role that minor geometric modifications play in aerodynamic efficiency. This focus enables researchers and engineers to explore the nuanced relationship between car geometry and aerodynamic performance, facilitating the optimization of vehicle designs for improved efficiency and performance.
<div align="center">
    <video src="https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/86b8046f-8858-4193-a904-f80cc59544d0" width="50%"></video>
</div>


## CFD Data
![Prsentation4-ezgif com-crop](https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/3d5e3b3e-4dcd-490f-9936-2a3dbda1402b)

## RegDGCNN: Dynamic Graph Convolutional Neural Network for Regression Tasks
![RegDGCNN_animationLong-ezgif com-crop](https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/a9a086e7-1e69-45cd-af8d-560b619172a8)

## Computational Efficiency of RegDGCNN
RegDGCNN model is both lightweight, with just 3 million parameters and a 10MB size, and fast, estimating drag for a 500k mesh face car design in only 1.2 seconds on four A100 GPUs. This represents a significant reduction in computational time compared to the 2.3 hours required for a conventional CFD simulation on a system with 128 CPU cores.

## Effect of Training Dataset Size

<table>
<tr>
<td>

- DrivAerNet is 60% larger than the previously available largest public dataset of cars and is the only opensource dataset that also models wheels and underbody, allowing accurate estimation of drag.
- Within the DrivAerNet dataset, expanding the dataset from 560 to 2800 car designs resulted in a 75% decrease in error. A similar trend is observed with the ShapeNet cars dataset, where enlarging the number of training samples from 1270 to 6352 entries yielded a 56% error reduction, further validating the inherent value of large datasets in driving advancements in surrogate modeling.

</td>
<td>

<img src="https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/30443fbb-5fe4-4a50-a9e0-d22af6f1aa2b" width="150%">

</td>
</tr>
</table>


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





