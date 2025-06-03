# Model-Guided Network with Cluster-Based Operators for Spatio-Spectral Super-Resolution

[![arXiv](https://img.shields.io/badge/arXiv-2409.02675-B31B1B.svg)](https://arxiv.org/abs/2505.24605)

This repository contains the implementation and additional resources for the paper:

**Model-Guided Network with Cluster-Based
Operators for Spatio-Spectral Super-Resolution**  
*Ivan Pereira-Sánchez, Julia Navarro, Ana Belén Petro and Joan Duran*  
Submmited to the IEEE Journal of Selected Topics on Signal Proceessing

---

## 📄 Abstract
This paper addresses the problem of reconstructing a high-resolution hyperspectral image from a low-resolution multispectral observation. While spatial super-resolution and spectral super-resolution have been extensively studied, joint spatio-spectral super-resolution remains relatively explored. We propose an end-to-end model-driven framework that explicitly decomposes the joint spatio-spectral super-resolution problem into spatial super-resolution, spectral super-resolution and fusion tasks. Each sub-task is addressed by unfolding a variational-based approach, where the operators involved in the proximal gradient iterative scheme are replaced with tailored learnable modules. In particular, we design an upsampling operator for spatial super-resolution based on classical back-projection algorithms, adapted to handle arbitrary scaling factors. Spectral reconstruction is performed using learnable cluster-based upsampling and downsampling operators. For image fusion, we integrate low-frequency estimation and high-frequency injection modules to combine the spatial and spectral information from spatial super-resolution and spectral super-resolution outputs. Additionally, we introduce an efficient nonlocal post-processing step that leverages image self-similarity by combining a multi-head attention mechanism with residual connections. Extensive evaluations on several datasets and sampling factors demonstrate the effectiveness of our approach.

---

## 📚 arXiv Preprint

The paper is currently under revision, and the first preprint is available on [arXiv](https://arxiv.org/abs/2505.24605).


---

## ⚙️ Setup

To begin, create an .env file in the project root directory and define the `DATASET_PATH` variable, pointing to the directory where your dataset is stored.

We provide an example DataModule using CAVE images. This module requires the data to generate the multispectral low and high resolution and the hyperspectral low and high resolution and stored as cropped .h5 files.

Also, you can adapt the dataset class accordingly how you have the data stored. Please note that we are unable to share the dataset used for training due to data access restrictions.

---
## Train

Run the following command:
   ```bash
   python train.py 
   ```
---
## Test

A specific script is still pending, but the training script includes a testing section that can be used as a reference
---
## 📌 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{pereirasánchez2025model,
      title={Model-Guided Network with Cluster-Based Operators for Spatio-Spectral Super-Resolution}, 
      author={Ivan Pereira-Sánchez and Julia Navarro and Ana Belén Petro and Joan Duran},
      year={2025},
      eprint={2505.24605},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2505.24605}, 
}
```