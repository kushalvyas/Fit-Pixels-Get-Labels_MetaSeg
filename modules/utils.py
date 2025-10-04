from matplotlib import pyplot as plt
import numpy as np
import os, os.path as osp
import torch

def show_image_subplot(images : list [np.ndarray], num_rows:int, num_cols :int , titles : list[str] = [], axis:str ='off' , dpi:int = 100) -> None:
    """ shows a matplotlib subplot for multiple images"""
    assert num_rows > 1 or num_cols > 1 , "Please ensure that you have more than 1 row or col"
    assert len(images) == int(num_rows*num_cols), "Please ensure that number of images provided match rows x cols product"
    titles = titles + ["No Title" for _ in range(len(images)-len(titles))] if len(titles) < len(images) else titles[:len(images)]
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, dpi=dpi)
    axs_flat = axs.flatten()
    for (_ax, _im, _title) in zip(axs_flat, images, titles):
        _ax.imshow(_im)
        _ax.axis(axis)
        _ax.set_title(_title)
    plt.tight_layout()
    plt.show()

def convert_tensor_to_onehot(x, num_classes):
    x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    onehot = torch.zeros(size=(x.shape[0], x.shape[1], num_classes), dtype=torch.int32)
    for c in range(num_classes):
        onehot[..., c] = (x == c)
    return onehot