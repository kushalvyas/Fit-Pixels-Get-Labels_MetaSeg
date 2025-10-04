from matplotlib import pyplot as plt
import numpy as np
import os, os.path as osp



def plot_result_row(imgs, titles, save=None, show=False):
    fig, axs = plt.subplots(1, len(imgs))
    for  (_ax, img, title) in zip(axs.flatten(), imgs, titles):
        _ax.axis('off')
        _ax.imshow(img)
        _ax.set_title(title)
    plt.tight_layout()

    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()
    
    
