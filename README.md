# DrivAerNet++

> Update (19.11.2024): DrivAerNet++ has been accepted to NeurIPS 2024! The full dataset is now released on [Harvard Dataverse](https://dataverse.harvard.edu/dataverse/DrivAerNet). Please note the (CC BY-NC 4.0) license terms, as outlined in the [License section](#license). 

> Update (11.09.2024): Due to the overwhelming interest and numerous inquiries from industry partners, we are excited to announce that we are now offering commercial licensing options for the DrivAerNet and DrivAerNet++ datasets. Please refer to the [DrivAerNet/DrivAerNet++ Commercial Inquiry](#drivaernetdrivaernet-commercial-inquiry) section.

> Update (15.08.2024): DrivAerNet has been utilized in the [IJCAI 2024 competition](https://aistudio.baidu.com/projectdetail/7459168?channelType=0&channel=0). All competitors' code and results are available in the open-source repository: [Rapid Aerodynamic Drag Prediction for Arbitrary Vehicles in 3D Space](https://github.com/PaddlePaddle/PaddleScience/tree/develop/jointContribution/IJCAI_2024).

> Update (11.07.2024): DrivAerNet is now integrated into NVIDIA Modulus [FIGConvUNet](https://github.com/NVIDIA/modulus/tree/main/examples/cfd/figconvnet) and [AeroGraphNet](https://github.com/NVIDIA/modulus/tree/main/examples/cfd/aero_graph_net)


Our new preprint: DrivAerNet++ paper [here](https://www.researchgate.net/publication/381470334_DrivAerNet_A_Large-Scale_Multimodal_Car_Dataset_with_Computational_Fluid_Dynamics_Simulations_and_Deep_Learning_Benchmarks)

Video summary of DrivAerNet++ paper [here](https://youtu.be/Y2-s0R_yHpo?si=E9B4BzDzcJebAMsC)

DrivAerNet paper: [here](https://www.researchgate.net/publication/378937154_DrivAerNet_A_Parametric_Car_Dataset_for_Data-Driven_Aerodynamic_Design_and_Graph-Based_Drag_Prediction)

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


## Importance of Dataset Diversity 

Dataset diversity and shape variation are crucial for developing robust deep learning models in aerodynamic car design. By providing a wide range of car shapes and configurations with high-fidelity CFD, DrivAerNet++ enables models to generalize better, supports exploration of unconventional designs, and enhances understanding of how geometric features impact aerodynamic performance.

![DrivAerNet_Demo_cropped](https://github.com/user-attachments/assets/1fa8a865-9e26-4985-b807-245d0227c610)

## Dataset Contents & Modalities
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


## Computational Cost

Running the high-fidelity CFD simulations for DrivAerNet++ required substantial computational resources. The simulations were conducted on the MIT Supercloud, leveraging parallelization across 60 nodes, totaling 2880 CPU cores, with each CFD case using 256 cores and 1000 GBs of memory. The full dataset requires **39 TB** of storage space. The simulations took approximately **3 × 10⁶ CPU-hours** to complete.

## Applications

DrivAerNet++ supports a wide array of machine learning applications, including but not limited to:

- 🚀 **Data-driven design optimization**: Optimize car designs based on aerodynamic performance.
- 🧠 **Generative AI**: Train generative models to create new car designs based on performance or aesthetics.
- 🎯 **Surrogate models**: Predict aerodynamic performance without full CFD simulations.
- 🔥 **CFD simulation acceleration**: Speed up simulations using machine learning and multi-GPU techniques.
- 📉 **Reduced Order Modeling**: Create data-driven reduced-order models for efficient & fast aerodynamic simulations.
- 💾 **Large-Scale Data Handling**: Efficiently store and manage large datasets from high-fidelity simulations.
- 🗜️ **Data Compression**: Implement high-performance lossless compression techniques.
- 🌐 **Part and shape classification**: Classify car categories or components to enhance design analysis.
- 🔧 **Automated CFD meshing**: Automate the meshing process based on car components to streamline simulations.


## Results Replication

DrivAerNet++ serves as a valuable benchmark dataset due to its size and diversity. It provides extensive coverage of various car designs and configurations, making it ideal for testing and validating machine learning models in aerodynamic design. We provide the train, test, and validation splits in the following folder: [train_val_test_splits](https://github.com/Mohamedelrefaie/DrivAerNet/tree/main/train_val_test_splits).

To replicate the results based on the parametric tabular data, please refer to the [`ParametricModels`](https://github.com/Mohamedelrefaie/DrivAerNet/tree/main/ParametricModels) directory and the code in `AutoML_parametric.py`.

To replicate the results based on geometric deep learning methods, please refer to the code saved in the [`DeepSurrogates`](https://github.com/Mohamedelrefaie/DrivAerNet/tree/main/DeepSurrogates) directory.

![image](https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/37dbd178-8fc1-46f0-a006-a873e0825bf1)

Drag values for the 8k car designs can be found [Here](https://www.dropbox.com/scl/fi/cfuv2zyaoxp87r7zgr0si/DrivAerNetPlusPlus_Drag_8k.csv?rlkey=3fd4c3c5a38plue3ir2r8kx7b&dl=0)


## Datasets Comparison

DrivAerNet++ stands out as the largest and most comprehensive dataset in the field of car design.

![image](https://github.com/Mohamedelrefaie/DrivAerNet/assets/86707575/3c5b33d5-7163-4c33-b734-eeffbb4fb1f0)


## Contributing

We welcome contributions to improve the dataset or project. Please submit pull requests for review.

## Maintenance and Support

Maintained by the DeCoDE Lab at MIT. Report issues via [GitHub issues](https://github.com/Mohamedelrefaie/DrivAerNet/issues).


## Additional Resources

- Tutorials: [Link](https://github.com/Mohamedelrefaie/DrivAerNet/tree/main/tutorials)


## Previous Version

To replicate the code and experiments from the first version of DrivAerNet, please refer to the folder: [DrivAerNet_v1](https://github.com/Mohamedelrefaie/DrivAerNet/tree/main/DrivAerNet_v1). 

## License

**Strict Licensing Notice**: DrivAerNet/DrivAerNet++ is released under the Creative Commons Attribution-NonCommercial 4.0 International License [(CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/deed.en) and is exclusively for non-commercial research and educational purposes. Any commercial use—including, but not limited to, training machine learning models, developing generative AI tools, creating software products, running new simulations using the provided geometries or any derived geometries, or other commercial R&D applications—is strictly prohibited. Unauthorized commercial use of DrivAerNet/DrivAerNet++, or any derived data, will result in enforcement by the MIT Technology Licensing Office (MIT TLO) and may carry legal consequences. The code is distributed under the MIT License.


## DrivAerNet/DrivAerNet++ Commercial Inquiry

If you are interested in the commercial use of the DrivAerNet or DrivAerNet++ datasets, please contact Mohamed Elrefaie (mohamed.elrefaie@mit.edu) and Faez Ahmed (faez@mit.edu) with the subject line: "DrivAerNet Commercial Inquiry".


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
