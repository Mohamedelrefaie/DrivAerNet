# DrivAerNet++: High-Fidelity Computational Fluid Dynamics & Deep Learning Benchmarks

<p align="center">
  <img src="assets/drivaernet_banner.png" alt="DrivAerNet++ Banner" width="100%">
</p>

<p align="center">
  <a href="https://neurips.cc/virtual/2024/poster/97586"><img src="https://img.shields.io/badge/NeurIPS-2024-blue.svg" alt="NeurIPS 2024"></a>
  <a href="https://arxiv.org/abs/2406.09624"><img src="https://img.shields.io/badge/arXiv-2406.09624-b31b1b.svg" alt="arXiv"></a>
  <a href="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZQLJJL"><img src="https://img.shields.io/badge/Dataset-Harvard%20Dataverse-orange.svg" alt="Dataset"></a>
  <a href="https://creativecommons.org/licenses/by-nc/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg" alt="License"></a>
  <a href="https://github.com/DeCoDE-Lab/DrivAerNet/stargazers"><img src="https://img.shields.io/github/stars/DeCoDE-Lab/DrivAerNet?style=social" alt="GitHub stars"></a>
</p>

<p align="center">
  <b>The largest and most comprehensive multimodal dataset for aerodynamic car design</b>
</p>

We present **DrivAerNet++**, comprising **8,000 diverse car designs** modeled with high-fidelity computational fluid dynamics (CFD) simulations, covering configurations such as fastback, notchback, and estateback.

---

## 📢 Latest News

| Date | News |
|------|------|
| 🆕 **2024** | **CarBench Released** — A unified benchmark for high-fidelity 3D car aerodynamics and generalization testing |

- 🏆 **Leaderboard:** [CarBench Leaderboard](https://decode.mit.edu/carbench)
- 📄 **Paper:** [CarBench Paper](https://arxiv.org/abs/2505.00000)

---

## 🔗 Quick Links

| Resource | Description | Link |
|----------|-------------|------|
| DrivAerNet++ Paper | NeurIPS 2024 Full Paper | [arXiv](https://arxiv.org/abs/2406.09624) |
| Dataset Download | Hosted on Harvard Dataverse | [Access Data](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZQLJJL) |
| Leaderboard | Submit models & compare results | [DrivAerNet++ Leaderboard](https://decode.mit.edu/drivaernetpp) |
| Video Summary | Overview of the project | [YouTube](https://www.youtube.com/watch?v=example) |
| Podcasts | Deep dive discussions | [DrivAerNet++](https://example.com/podcast) |

---

## 🏎️ Design & Shape Variation

<p align="center">
  <img src="assets/design_variation.png" alt="Design Variation" width="80%">
</p>

### Design Parameters

Several geometric parameters with significant impact on aerodynamics were selected and varied within a specific range. These parameter ranges were chosen to avoid values that are either difficult to manufacture or not aesthetically pleasing.

### Shape Variation

DrivAerNet++ covers **all conventional car designs**. The dataset encompasses various underbody and wheel designs to represent both:
- **Internal Combustion Engine (ICE)** vehicles
- **Electric Vehicles (EV)**

> 💡 Each 3D car geometry is parametrized with **26 parameters** that completely describe the design.

### Importance of Diversity

By providing a wide range of car shapes and configurations with high-fidelity CFD, DrivAerNet++ enables:
- ✅ Models to **generalize better**
- ✅ Exploration of **unconventional designs**
- ✅ Enhanced understanding of how **geometric features impact aerodynamic performance**

---

## 📦 Dataset Contents & Modalities

### ✅ Available Modalities

| Modality | Description |
|----------|-------------|
| **Parametric Models** | Structured tabular design parameters |
| **Volumetric Fields** | Full 3D CFD (pressure, velocity, turbulence) |
| **Surface Fields** | Coefficient of pressure (Cp) and Wall Shear Stress (WSS) |
| **Streamlines** | Flow visualization data illustrating streamlines |
| **Point Clouds** | Dense and sparse point cloud representations |
| **Meshes** | High-resolution 3D surface triangulations |
| **Aerodynamic Coefficients** | Drag (Cd), Lift (Cl), and moment coefficients |
| **Annotations** | Per-part semantic labels |
| **Renderings** | High-quality photorealistic 2D renderings |
| **Sketches** | Hand-drawn style sketches (Canny edge & CLIPasso) |

### 🚧 Coming Soon

- 📐 **2D Slices:** Planar field extractions
- 📊 **Signed Distance Fields (SDF):** For occupancy modeling
- 💥 **Deformations:** Simulation outputs under crash/pressure conditions

### Dataset Annotations

The dataset includes detailed annotations for various car components (**29 labels**), such as wheels, side mirrors, and doors. These are instrumental for:
- Classification
- Semantic segmentation
- Automated meshing

<p align="center">
  <img src="assets/annotations.png" alt="Dataset Annotations" width="70%">
</p>

---

## ✏️ Sketch-to-Design Extension

We bridge the gap between **conceptual creativity** and **computational design** with 2D hand-drawn sketches and photorealistic renderings.

<p align="center">
  <img src="assets/sketch_to_design.png" alt="Sketch to Design" width="80%">
</p>

---

## 💾 Dataset Access & Download

The dataset is hosted on **Harvard Dataverse** ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)).

| Specification | Value |
|--------------|-------|
| **Total Size** | 39 TB |
| **Subsets** | 3D Meshes, Pressure, Wall Shear Stress, Full CFD Domain |

We provide instructions on how to use [Globus](https://www.globus.org/) to download the dataset efficiently.

### Performance Data

| Data | Download |
|------|----------|
| Drag Values | [Download CSV](https://example.com/drag_values.csv) |
| Frontal Projected Areas | [Download CSV](https://example.com/frontal_areas.csv) |

---

## 🏆 Leaderboard & Comparisons

DrivAerNet++ serves as a valuable benchmark for testing and validating machine learning models in aerodynamic design. Researchers can submit their models to compare against state-of-the-art baselines.

👉 **[View DrivAerNet++ Leaderboard](https://decode.mit.edu/drivaernetpp)**

### Datasets Comparison

<p align="center">
  <img src="assets/dataset_comparison.png" alt="Dataset Comparison" width="80%">
</p>

> DrivAerNet++ stands out as the **largest and most comprehensive dataset** in the field.

---

## 📚 Related Research & Extensions

### TripOptimizer

A fully differentiable deep-learning framework for rapid aerodynamic analysis and shape optimization on industry-standard car designs.

📄 [Read Paper](https://arxiv.org/abs/example)

### AI Agents in Engineering Design

A multi-agent framework leveraging VLMs and LLMs to accelerate the car design process—from concept sketching to CAD modeling, meshing, and simulation.

📄 [Read Paper](https://arxiv.org/abs/example)

### RegDGCNN

We have open-sourced the RegDGCNN pipeline for surface field prediction on 3D car meshes.

🔗 [View Code](https://github.com/DeCoDE-Lab/RegDGCNN) | 📄 [Benchmarking Paper](https://arxiv.org/abs/example)

---

## 🛠️ Framework Integrations

DrivAerNet++ is integrated into leading Scientific Machine Learning (SciML) frameworks:

### NVIDIA Modulus

- [FIGConvUNet Example](https://docs.nvidia.com/modulus/examples/figconvunet)
- [AeroGraphNet Example](https://docs.nvidia.com/modulus/examples/aerographnet)

### PaddleScience (Baidu)

- [IJCAI 2024 Competition](https://example.com/ijcai2024)
- [DrivAerNet++ Example](https://paddlescience.readthedocs.io/drivaernetpp)

---

## 💻 Computational Cost & Applications

### Resources Used

| Resource | Specification |
|----------|--------------|
| **Infrastructure** | MIT Supercloud (60 nodes, 2880 CPU cores) |
| **Cost** | Approx. 3 × 10⁶ CPU-hours |

### Applications

| Application | Description |
|-------------|-------------|
| 🚀 Data-driven design optimization | Optimize car designs using ML |
| 🧠 Generative AI for car design | Generate novel car geometries |
| 🎯 Surrogate modeling | Fast approximations of CFD |
| 🔥 CFD simulation acceleration | Speed up simulations with ML |
| 📉 Reduced Order Modeling | Compress high-dimensional data |
| 🔧 Automated CFD meshing | Automate mesh generation |

---

## ⚖️ License & Commercial Use

### Strict Licensing Notice

> ⚠️ **DrivAerNet/DrivAerNet++** is released under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

| Usage | Status |
|-------|--------|
| ✅ Non-commercial research | **Allowed** |
| ✅ Educational purposes | **Allowed** |
| ❌ Commercial use | **Prohibited** |
| ❌ Model training for commercial tools | **Prohibited** |
| ❌ Commercial R&D | **Prohibited** |

**Code License:** [MIT License](LICENSE)

### Commercial Inquiry

For commercial licensing, please contact:

📧 **Mohamed Elrefaie** — [mohamed.elrefaie@mit.edu](mailto:mohamed.elrefaie@mit.edu)  
📧 **Faez Ahmed** — [faez@mit.edu](mailto:faez@mit.edu)

**Subject:** `"DrivAerNet Commercial Inquiry"`

---

## 📖 Citations

### DrivAerNet++ (NeurIPS 2024)

```bibtex
@inproceedings{NEURIPS2024_013cf29a,
    author    = {Elrefaie, Mohamed and Morar, Florin and Dai, Angela and Ahmed, Faez},
    booktitle = {Advances in Neural Information Processing Systems},
    editor    = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
    pages     = {499--536},
    publisher = {Curran Associates, Inc.},
    title     = {DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks},
    url       = {https://proceedings.neurips.cc/paper_files/paper/2024/file/013cf29a9e68e4411d0593040a8a1eb3-Paper-Datasets_and_Benchmarks_Track.pdf},
    volume    = {37},
    year      = {2024}
}
```

<details>
<summary><b>Click to see citations for DrivAerNet (v1)</b></summary>

#### Journal of Mechanical Design

```bibtex
@article{elrefaie2025drivaernet,
    title     = {DrivAerNet: A Parametric Car Dataset for Data-Driven Aerodynamic Design and Prediction},
    author    = {Elrefaie, Mohamed and Dai, Angela and Ahmed, Faez},
    journal   = {Journal of Mechanical Design},
    volume    = {147},
    number    = {4},
    year      = {2025},
    publisher = {American Society of Mechanical Engineers Digital Collection}
}
```

#### IDETC-CIE 2024

```bibtex
@proceedings{10.1115/DETC2024-143593,
    author = {Elrefaie, Mohamed and Dai, Angela and Ahmed, Faez},
    title  = {DrivAerNet: A Parametric Car Dataset for Data-Driven Aerodynamic Design and Graph-Based Drag Prediction},
    volume = {Volume 3A: 50th Design Automation Conference (DAC)},
    series = {International Design Engineering Technical Conferences and Computers and Information in Engineering Conference},
    pages  = {V03AT03A019},
    year   = {2024},
    month  = {08},
    doi    = {10.1115/DETC2024-143593},
    url    = {https://doi.org/10.1115/DETC2024-143593}
}
```

</details>

---

## 🔧 Maintenance & Support

<p align="center">
  Maintained by the <a href="https://decode.mit.edu"><b>DeCoDE Lab</b></a> at MIT
</p>

- 🐛 **Report Issues:** [GitHub Issues](https://github.com/DeCoDE-Lab/DrivAerNet/issues)
- 📚 **View Tutorials:** [Documentation](https://decode.mit.edu/drivaernetpp/tutorials)
- 📦 **Original V1 Code:** [DrivAerNet_v1](https://github.com/DeCoDE-Lab/DrivAerNet_v1)

---

<p align="center">
  <img src="assets/mit_logo.png" alt="MIT Logo" height="50">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/decode_logo.png" alt="DeCoDE Lab Logo" height="50">
</p>

<p align="center">
  Made with ❤️ at MIT
</p>
