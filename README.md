# Fit Pixels, Get Labels: Metalearned Implicit Networks for Image Segmentation `(MetaSeg)`
Repository for Fit Pixels, Get Labels: Meta learned implicit networks for medical image segmentation (MICCAI'25 ORAL presentation). 

[Project page](https://kushalvyas.github.io/metaseg.html) | [Paper](https://link.springer.com/chapter/10.1007/978-3-032-04947-6_19) | [Openreview](https://papers.miccai.org/miccai-2025/0340-Paper3113.html) | [Demo (coming soon!)](https://colab.research.google.com/drive/1C9xon4HPBtXA_GTxPuSNRlTIU853qbIt?usp=sharing)

## Installation instructions.
Please install the following packages for running MetaSeg. You can find then in requirements.txt. Feel free to create a seperate virtual venv/conda environment and then install specific python packages.

    pip install -r requirements.txt

Our codebase (MetaSeg) also depends on the [Alpine](https://github.com/kushalvyas/alpine/) INR library. Please install that as well. 

    git clone https://github.com/kushalvyas/alpine/
    cd alpine
    pip install .


## Instructions to run code:

Each __experiment__ is its own jupyter notebook. 

1. __2D segmentation (5 class):__ run metaseg_2d_5classes.ipynb
2. __2D segmentation (24 class):__ run metaseg_2d_24classes.ipynb
3. __3D Segmentation (5 class):__ run metaseg_3d_5classes.ipynb 

__For visualization:__
We also provide a script to visualize the principal components of learned MetaSeg features. Please find that in *metaseg_vis_pca.ipynb*.

## Dataset:

We use the [OASIS-MRI neurite](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md) dataset. This is part of the bigger [OASIS-MRI dataset](https://sites.wustl.edu/oasisbrains/). For 2D segmentation, images are preprocessed to remain the full size of 192 x 192, while for 3D, we downsample our volumes to 80 x 80 x 100 for computational feasibilty.

## Baselines:

We use the U-Net proposed by [Buda et.al. from PyTorch Hub](https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/) for 2D MRI segmentation and [SegRestNet proposed by Myronenko et.al](https://arxiv.org/pdf/1810.11654) using the [MONAI package](https://docs.monai.io/en/0.8.0/index.html). Additionally, for 3D INR segmentation, we also use the [NISF baseline](https://github.com/niloide/implicit_segmentation) proposed by Stolt-Ansó et.al. Please refer to the respective codebases to run any baselines.

## Citation

If you find our code or work useful, please consider citing us!

    @InProceedings{vyas2025metaseg,
      author="Vyas, Kushal
        and Veeraraghavan, Ashok
        and Balakrishnan, Guha",
      title="Fit Pixels, Get Labels: Meta-learned Implicit Networks for Image Segmentation",
      booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025",
      year="2026",
      publisher="Springer Nature Switzerland",
      pages="194--203",
      isbn="978-3-032-04947-6"
    }

