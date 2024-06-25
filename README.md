# DrivAerNet++

Our new preprint: DrivAerNet++ paper [here](https://www.researchgate.net/publication/381470334_DrivAerNet_A_Large-Scale_Multimodal_Car_Dataset_with_Computational_Fluid_Dynamics_Simulations_and_Deep_Learning_Benchmarks)

DrivAerNet Paper: [here](https://www.researchgate.net/publication/378937154_DrivAerNet_A_Parametric_Car_Dataset_for_Data-Driven_Aerodynamic_Design_and_Graph-Based_Drag_Prediction)

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
- **Parametric Models**: Parametric models with tabular data, allowing extensive exploration of automotive design variations.
- **Point Cloud Data**: Point cloud data for each car design.
- **3D Car Meshes**: Detailed 3D meshes of each car design, suitable for various machine learning applications.
- **CFD Simulation Data**: High-fidelity CFD simulation data for each car design, including 3D volumetric fields, surface fields, and streamlines.
- **Aerodynamic Coefficients**: Key aerodynamic metrics such as drag coefficient (Cd), lift coefficient (Cl), and more.
  
![DatasetContents](https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/424a1aac-fe9b-4e4f-ba14-20f466311224)

## Dataset Annotations
In addition to the CFD simulation data, our dataset includes detailed annotations for various car components (29 labels), such as wheels, side mirrors, and doors. These annotations are instrumental for a range of machine learning tasks, including classification, semantic segmentation, and object detection. The comprehensive labeling can also facilitate automated CFD meshing processes by providing precise information about different car components. By incorporating these labels, our dataset enhances the
utility for developing and testing advanced algorithms in automotive design and analysis.

![DrivAerNet_ClassLabels_new](https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/18833c92-6be9-437a-be10-4c52f9ed105f)


## Applications

DrivAerNet++ supports a wide array of machine learning applications, including but not limited to:
- Data-driven design optimization
- Generative AI
- Surrogate model training
- CFD simulation acceleration
- Geometric classification

## Results Replication

To replicate the results based on the parametric tabular data, please refer to the [`ParametricModels`](https://github.com/Mohamedelrefaie/DrivAerNet/tree/main/ParametricModels) directory and the code in `AutoML_parametric.py`.

To replicate the results based on geometric deep learning methods, please refer to the code saved in the [`DeepSurrogates`](https://github.com/Mohamedelrefaie/DrivAerNet/tree/main/DeepSurrogates) directory.

### Training Using Point Clouds
You can access the preprocessed point clouds for training at the following link:  
[DrivAerNet++ Processed Point Clouds (100k)](https://www.dropbox.com/work/decode_lab/Datasets/Public%20Documents/DrivAerNet%2B%2B/DrivAerNetPlusPlus_Processed_Point_Clouds_100k)

### Training Using STL Files / Graphs
For training using the STL files (for graph-based methods or point sampling at a different resolution), please download the STLs from the following link:  
[DrivAerNet++ STL Files](https://www.dropbox.com/work/decode_lab/Datasets/Public%20Documents/DrivAerNet%2B%2B/DrivAerNetPlusPlus_STLs_Combined)


The data will be released after the review process on Harvard Dataverse.


## Computational Cost

Running the high-fidelity CFD simulations for DrivAerNet++ required substantial computational resources. The simulations were conducted on the MIT Supercloud, leveraging parallelization across 60 nodes, totaling 2880 CPU cores, with each CFD case using 256 cores and 1000 GBs of memory. The full dataset requires **39 TB** of storage space. The simulations took approximately **3 × 10⁶ CPU-hours** to complete.

## Contributing

We welcome contributions to improve the dataset or project. Please submit pull requests for review.

## Maintenance and Support
Maintained by the DeCoDE Lab at MIT. Report issues via [GitHub issues](https://github.com/Mohamedelrefaie/DrivAerNet/issues).

## License

The code is distributed under the MIT License. The DrivAerNet++ dataset is distributed under the Creative Commons Attribution-NonCommercial (CC BY-NC) license. Full terms for the dataset license [here](https://creativecommons.org/licenses/by-nc/4.0/deed.en).

## Additional Resources

- Tutorials: [Link](https://github.com/Mohamedelrefaie/DrivAerNet/tree/main/tutorials)
- Technical Documentation: [Link]

## Previous Version

To replicate the code and experiments from the first version of DrivAerNet, please refer to the folder: [DrivAerNet_v1](https://github.com/Mohamedelrefaie/DrivAerNet/tree/main/DrivAerNet_v1). This includes 4,000 car designs based on the fastback category. The dataset can be accessed [here](https://www.dropbox.com/scl/fo/mnnnogykaoixdus9j0wr7/AO0PywI7VAZ1_ZL0e2HcjaA?rlkey=iox7cd8510800iinkqalph9zx&dl=0).



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
