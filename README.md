# DrivAerNet++

We present DrivAerNet++, the largest and most comprehensive multimodal dataset for aerodynamic car design. DrivAerNet++ comprises 8,000 diverse car designs modeled with high-fidelity computational fluid dynamics (CFD) simulations. The dataset includes diverse car configurations such as fastback, notchback, and estateback, with different underbody and wheel designs.

## Design Parameters

Design parameters for the generation of the DrivAerNet++ dataset. Several geometric parameters with significant impact on aerodynamics were selected and varied within a specific range. These parameter ranges were chosen to avoid values that are either difficult to manufacture or not aesthetically pleasing. 

<img src="https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/7d5e016e-d2e5-4e7a-b5eb-cac5e0727009" />

## Shape Variation

DrivAerNet++ covers all conventional car designs. The dataset encompasses various underbody and wheel designs to represent both internal combustion engine (ICE) and electric vehicles (EV). This extensive coverage allows for comprehensive studies on the impact of geometric variations on aerodynamic performance. By including a diverse set of car shapes, DrivAerNet++ facilitates the exploration of aerodynamic effects across different vehicle types, supporting both academic research and industrial applications.

<table>
  <tr>
    <td><img src="https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/98064523-1a12-4ab3-9be4-8b745d1d1072" width="100%"></td>
    <td><img src="https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/0fc97e2a-f06c-4036-a9de-8d9d1c5e6a91" width="100%"></td>
  </tr>
</table>

Each 3D car geometry is parametrized with 26 parameters that completely describe the design. To create diverse car designs, we used two main morphing methods: morphing boxes and direct morphing. For a detailed description of the design parameters, their ranges, lower and upper bounds, please refer to the paper.

![DrivAerNet_params-ezgif com-crop](https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/8a2408de-a920-4326-8433-9b8b9b231ffb)



## Dataset Contents

- **CFD Simulation Data**: High-fidelity CFD simulation data for each car design, including 3D flow fields.
- **3D Car Meshes**: Detailed 3D meshes of each car design, suitable for various machine learning applications.
- **Parametric Models**: Parametric models with tabular data, allowing extensive exploration of automotive design variations.
- **Aerodynamic Coefficients**: Key aerodynamic metrics such as drag coefficient (Cd), lift coefficient (Cl), and more.
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
1. Clone the repository: `git clone https://github.com/Mohamedelrefaie/DrivAerNet.git`
2. Navigate to the dataset directory: `cd DrivAerNet/dataset`
3. Load the dataset using the provided Python scripts.

## Computational Cost

Running the high-fidelity CFD simulations for DrivAerNet++ required substantial computational resources. The simulations were conducted on the MIT Supercloud, leveraging parallelization across 60 nodes, totaling 2880 CPU cores, with each CFD case using 256 cores and 1000 GBs of memory. The full dataset requires **39 TB** of storage space. The simulations took approximately **3 × 10⁶ CPU-hours** to complete.

## Contributing

We welcome contributions to improve the dataset or project. Please submit pull requests for review.

## Maintenance and Support

Maintained by the DeCoDE Lab at MIT. Report issues via [GitHub issues](https://github.com/yourusername/DrivAerNet/issues).

## License

The code is distributed under the MIT License. The DrivAerNet++ dataset is distributed under the Creative Commons Attribution-NonCommercial (CC BY-NC) license. Full terms for the dataset license [here](https://creativecommons.org/licenses/by-nc/4.0/deed.en).

## Additional Resources

- Tutorials: [Link](https://github.com/Mohamedelrefaie/DrivAerNet/tree/main/tutorials)
- Technical Documentation: [Link]

## Previous Version

To replicate the code and experiments from the first version of DrivAerNet, please refer to the folder: [DrivAerNet_v1](https://github.com/Mohamedelrefaie/DrivAerNet/tree/main/DrivAerNet_v1). This includes 4,000 car designs based on the fastback category. 


## Citations

To cite this work, please use the following reference:

```bibtex
@article{elrefaie2024drivaernet++,
  title={DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks},
  author={Elrefaie, Mohamed and Morar, Florin and Dai, Angela and Ahmed, Faez},
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
