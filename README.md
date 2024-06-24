# DrivAerNet++

We present DrivAerNet++, the largest and most comprehensive multimodal dataset for aerodynamic car design. DrivAerNet++ comprises 8,000 diverse car designs modeled with high-fidelity computational fluid dynamics (CFD) simulations. The dataset includes diverse car configurations such as fastback, notchback, and estateback, with different underbody and wheel designs to represent both internal combustion engines and electric vehicles.

## Design Parameters

Design parameters for the generation of the DrivAerNet++ dataset. Several geometric parameters with significant impact on aerodynamics were selected and varied within a specific range. These parameter ranges were chosen to avoid values that are either difficult to manufacture or not aesthetically pleasing. 

<img src="https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/7d5e016e-d2e5-4e7a-b5eb-cac5e0727009" />


## Dataset Contents

- **CFD Simulation Data**: High-fidelity CFD simulation data for each car design, including 3D flow fields and surface pressure distributions.
- **3D Car Meshes**: Detailed 3D meshes of each car design, suitable for various machine learning applications.
- **Parametric Models**: Parametric models created using ANSA® software, allowing extensive exploration of automotive design variations.
- **Aerodynamic Coefficients**: Key aerodynamic metrics such as drag coefficient (Cd), lift coefficient (Cl), and more.
- **Flow and Surface Field Data**: Detailed flow and surface field data, including velocity and pressure fields.
- **Segmented Parts**: Segmented parts of the car models for classification tasks.
- **Point Cloud Data**: Point cloud data for each car design.

## Applications

DrivAerNet++ supports a wide array of machine learning applications, including but not limited to:
- Data-driven design optimization
- Generative AI
- Surrogate model training
- CFD simulation acceleration
- Geometric classification

## How to Access

The dataset and accompanying Python scripts for data conversion are available at [].

## Usage Instructions

To use the dataset, follow these steps:
1. Clone the repository: `git clone https://github.com/yourusername/DrivAerNet.git`
2. Navigate to the dataset directory: `cd DrivAerNet/dataset`
3. Load the dataset using the provided Python scripts.

## Contributing

We welcome contributions to improve the dataset or project. Please submit pull requests for review.

## Maintenance and Support

Maintained by the DeCoDE Lab at MIT. Report issues via [GitHub issues](https://github.com/yourusername/DrivAerNet/issues).

## License

The code is distributed under the MIT License. The DrivAerNet++ dataset is distributed under the Creative Commons Attribution-NonCommercial (CC BY-NC) license. Full terms for the dataset license [here](https://creativecommons.org/licenses/by-nc/4.0/deed.en).

## Additional Resources

- Tutorials: [Link]
- Technical Documentation: [Link]

## Previous Version

To replicate the code and experiments from the first version of DrivAerNet, please refer to the folder: [DrivAerNet_v1](https://github.com/Mohamedelrefaie/DrivAerNet/tree/main/DrivAerNet_v1). This includes 4,000 car designs based on the fastback category. 


## Citations

To cite this work, please use the following reference:

```bibtex
@article{elrefaie2024drivaernetplusplus,
  title={DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks},
  author={Elrefaie, Mohamed and Dai, Angela and Ahmed, Faez},
  journal={arXiv preprint arXiv:2406.09624},
  year={2024}
}
```

To cite the first version of DrivAerNet, please use the following reference:
```bibtex
@article{elrefaie2024drivaernet,
  title={DrivAerNet: A Parametric Car Dataset for Data-Driven Aerodynamic Design and Graph-Based Drag Prediction},
  author={Elrefaie, Mohamed and Dai, Angela and Ahmed, Faez},
  journal={arXiv preprint arXiv:2403.08055},
  year={2024}
}
