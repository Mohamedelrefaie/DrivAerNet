# DrivAerNet
DrivAerNet: A Parametric Car Dataset for Data-driven Aerodynamic Design and Graph-Based Drag Prediction

## Introduction
DrivAerNet is a large-scale, high-fidelity CFD dataset of 3D industry-standard car shapes designed for data-driven aerodynamic design. It comprises 4000 high-quality 3D car meshes and their corresponding aerodynamic performance coefficients, alongside full 3D flow field information.


## Parametric Model 
The DrivAerNet dataset includes a parametric model of the DrivAer fastback, developed using ANSA® software to enable extensive exploration of automotive design variations. This model is defined by 50 geometric parameters, allowing the generation of 4000 unique car designs through Optimal Latin Hypercube sampling and the Enhanced Stochastic Evolutionary Algorithm. 

DrivAerNet dataset incorporates a wide range of geometric modifications, including changes to side mirrors, muffler positions, windscreen and rear window dimensions, engine undercover size, front door and fender offsets, hood placement, headlight scale, overall car length and width, upper and underbody scaling, and key angles like the ramp, diffusor, and trunk lid angles, to thoroughly investigate their impacts on car aerodynamics.
![DrivAerNetMorphingNew2-ezgif com-crop](https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/ed7e825a-db41-4230-ac91-1286c69d61fe)

![ezgif-7-2930b4ea0d](https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/f6af36aa-079b-49d9-8ac7-a6b20595faee)


## Car Designs
The DrivAerNet dataset specifically concentrates on conventional car designs, highlighting the significant role that minor geometric modifications play in aerodynamic efficiency. This focus enables researchers and engineers to explore the nuanced relationship between car geometry and aerodynamic performance, facilitating the optimization of vehicle designs for improved efficiency and performance.
<div align="center">
    <video src="https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/86b8046f-8858-4193-a904-f80cc59544d0" width="50%"></video>
</div>


## CFD Data
The DrivAerNet dataset provides comprehensive 3D flow field data, including detailed analyses of pressure and velocity fields along the $x$ and $y$ axes, as well as wall-shear stresses, contributing to a deeper understanding of aerodynamic forces. Additionally, it offers key aerodynamic metrics relevant to car geometries, such as the total moment coefficient $C_m$, drag coefficient $C_d$, lift coefficient $C_l$, front lift coefficient $C_{l,f}$, and rear lift coefficient $C_{l,r}$. The dataset also encompasses important parameters like wall-shear stress and the $y^{+}$ metric, essential for mesh quality evaluations, and insights into flow trajectories, enhancing the comprehension of aerodynamic interactions.

![Prsentation4-ezgif com-crop](https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/3d5e3b3e-4dcd-490f-9936-2a3dbda1402b)



## RegDGCNN: Dynamic Graph Convolutional Neural Network for Regression Tasks
In this study, we adapt the Dynamic Graph Convolutional Neural Network (DGCNN) framework, traditionally used for classification tasks, to tackle regression problems, specifically predicting aerodynamic coefficients. Our RegDGCNN model integrates PointNet's spatial encoding with graph CNNs' relational analysis to understand fluid dynamics around objects. It employs edge convolution (EdgeConv) on dynamically updating graphs to capture fluid flow's intricate interactions, presenting a new approach for precise aerodynamic parameter estimation.

RegDGCNN utilizes the extensive DrivAerNet dataset to deliver accurate drag predictions directly from 3D meshes, overcoming conventional challenges like the requirement for 2D image rendering or Signed Distance Fields (SDF) generation.

![RegDGCNN_animationLong-ezgif com-crop](https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/a9a086e7-1e69-45cd-af8d-560b619172a8)

## Computational Efficiency of RegDGCNN
RegDGCNN model is both lightweight, with just 3 million parameters and a 10MB size, and fast, estimating drag for a 540k mesh face car design in only 1.2 seconds on four A100 GPUs. This represents a significant reduction in computational time compared to the 2.3 hours required for a conventional CFD simulation on a system with 128 CPU cores.

## Effect of Training Dataset Size

<table>
<tr>
<td>

- DrivAerNet is 60% larger than the previously available largest public dataset of cars and is the only opensource dataset that also models wheels and underbody, allowing accurate estimation of drag.
- Within the DrivAerNet dataset, expanding the training set from 560 to 2800 car designs resulted in a 75% decrease in error. A similar trend is observed with the ShapeNet cars dataset, where enlarging the number of training samples from 1270 to 6352 entries yielded a 56% error reduction, further validating the inherent value of large datasets in driving advancements in surrogate modeling.

</td>
<td>

<img src="https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/eb38b12a-3301-4358-8e3a-2a791376dc49" width="150%">

</td>
</tr>
</table>


## Dataset Details & Contents

The DrivAerNet dataset is meticulously crafted to serve a wide range of applications from aerodynamic analysis to the training of advanced machine learning models for automotive design optimization. It includes:

- **CFD Simulation Data**: The raw dataset, including full 3D pressure, velocity fields, and wall-shear stresses, computed using **8-16 million mesh elements** has a total size of $\sim$ **16TB**.
- **Curated CFD Simulations**: For ease of access and use, a **streamlined version of the CFD simulation data** is provided, refined to include key insights and data, reducing the size to $\sim$ **1TB**. 
- **3D Car Meshes**: A total of **4000 designs**, showcasing a variety of conventional car shapes and emphasizing the impact of minor geometric modifications on aerodynamic efficiency. The 3D meshes and aerodynamic coefficients $\sim$ **84GB**.
- 2D slices include the car's wake in the $x$-direction and the symmetry plane in the $y$-direction $\sim$ **12GB**.

This rich dataset, with its focus on the nuanced effects of design changes on aerodynamics, provides an invaluable resource for researchers and practitioners in the field.

## Contents of the small subset
- `DrivAerNet_projected_areas.txt`: Contains the projected frontal area for each car, which is essential for aerodynamic coefficient calculations.
- `DrivAerNet_STLs_DoE`: A folder with car designs from the Design of Experiments (DoE) used for the CFD simulations and defining boundary conditions.
- `DrivAerNet_STLs_Combined`: Contains combined STL files where each file represents a complete car model, including body and wheels, which can be directly used in AI models for aerodynamic drag predictions.
- `yNormal`: Includes a normal slice for each car at the symmetry plane saved as VTK format (at y=0).
- `xNormal`: Contains slices in the x-direction at x=4, capturing the wake of the car, saved as VTK.
- `SurfacePressureVTK`: Comprises 3D car models with surface pressure fields saved in VTK format.
- Videos: Two videos showcasing the 4000 designs are provided for better visualization.
- `AeroCoefficients_DrivAerNet_FilteredCorrected.csv`: This file includes the aerodynamic coefficients for each design (drag, lift, frontal lift, and rear lift coefficients).
- `train_val_test_splits`: This folder includes the train/val/test splits.

## Usage Instructions
The dataset and accompanying Python scripts for data conversion are available at [GitHub repository link].

## Contributing
We welcome contributions to improve the dataset or project. Please submit pull requests for review.

## Maintenance and Support
Maintained by the DeCoDE Lab at MIT. Report issues via [GitHub issues](https://github.com/Mohamedelrefaie/DrivAerNet/issues).

## License
The code is distributed under the MIT License. The DrivAerNet dataset is distributed under the Creative Commons Attribution-NonCommercial (CC BY-NC) license. Full terms for the dataset license [here](https://creativecommons.org/licenses/by-nc/4.0/deed.en).

## Additional Resources
- Tutorials: [Link]
- Technical Documentation: [Link]

  
## Citations
To cite this work, please use the following reference:
```bibtex
@article{elrefaie2024drivaernet,
  title={DrivAerNet: A Parametric Car Dataset for Data-Driven Aerodynamic Design and Graph-Based Drag Prediction},
  author={Elrefaie, Mohamed and Dai, Angela and Ahmed, Faez},
  journal={arXiv preprint arXiv:2403.08055},
  year={2024}
}
