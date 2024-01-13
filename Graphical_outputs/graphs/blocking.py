import matplotlib.pyplot as plt
import numpy as np

save_fig = True

type_ = ""

if type_ == "blocking":
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), num="blocking")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 6), num="blocking")

    # Vytvoření heatmapy
    im = plt.imread("blocking.png")
    h, w = im.shape[:2]

    ax2.vlines(51.5, -1, h, color='red', linewidth=2)
    ax2.vlines(71.5, -1, 55.5, color='red', linewidth=2)
    ax2.vlines(91.5, -1, 55.5, color='red', linewidth=2)
    # ax2.vlines(19.5, -1, h, color='red', linewidth=2)
    # ax2.vlines(35.5, -1, h, color='red', linewidth=2)

    ax2.hlines(55.5, -1, w, color='red', linewidth=2)
    ax2.hlines(15.5, 51.5, w, color='red', linewidth=2)
    ax2.hlines(34.5, 51.5, w, color='red', linewidth=2)

    for ax in (ax1, ax2):
        ax.imshow(im)

        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor("none")
        ax.axis('off')

    fig.set_facecolor("none")
    fig.tight_layout()

    if save_fig:
        plt.savefig("blocking.pdf", format="pdf", dpi=700, bbox_inches='tight')

    plt.show()
else:
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), num="blocking")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), num="blocking")

    # Vytvoření heatmapy
    im1 = plt.imread("zoomed_b.jpg")[100:-60, :]
    im2 = plt.imread("lined_c.jpg")[150:-150, 80:-120]
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    for ax, im in zip((ax1, ax2), (im1, im2)):
        ax.imshow(im, extent=[0, w1, 0, h1])

        ax.set_facecolor("none")
        ax.axis('off')
        ax.set_aspect('equal', adjustable='box')

    [ax1.vlines(i * 90 + 100, 0, h1, color='black', linewidth=2) for i in range(11)]

    ax2.vlines(0, 0, h1, color='none', linewidth=0)

    fig.set_facecolor("none")
    fig.tight_layout()

    if save_fig:
        plt.savefig("spots.pdf", format="pdf", dpi=700, bbox_inches='tight')

    plt.show()
