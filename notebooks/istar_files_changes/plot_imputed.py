import sys

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch


from utils import load_pickle, save_image, read_lines, load_image
# from visual import cmap_turbo_truncated


def plot_super(
        x, outfile, underground=None, truncate=None):

    x = x.copy()
    mask = np.isfinite(x)

    if truncate is not None:
        x -= np.nanmean(x)
        x /= np.nanstd(x) + 1e-12
        x = np.clip(x, truncate[0], truncate[1])

    x -= np.nanmin(x)
    x /= np.nanmax(x) + 1e-12

    cmap = plt.get_cmap('turbo')
    # cmap = cmap_turbo_truncated
    if underground is not None:
        under = underground.mean(-1, keepdims=True)
        under -= under.min()
        under /= under.max() + 1e-12

    img = cmap(x)[..., :3]
    if underground is not None:
        img = img * 0.5 + under * 0.5
    img[~mask] = 1.0
    img = (img * 255).astype(np.uint8)
    save_image(img, outfile)


def main():

    prefix = sys.argv[1]  # e.g. 'data/her2st/B1/'
    gene_names = read_lines(f'{prefix}gene-names.txt')
    mask = load_image(f'{prefix}mask-small.png') > 0

    os.makedirs(f"{prefix}cnts-super-plots", exist_ok=True)
    os.makedirs(f"{prefix}ct-super-plots", exist_ok=True)

    # ---------------------------
    # 1) Gene super-resolution
    # ---------------------------
    for gn in gene_names:
        cnts = load_pickle(f'{prefix}cnts-super/{gn}.pickle')
        cnts[~mask] = np.nan
        plot_super(cnts, f'{prefix}cnts-super-plots/{gn}.png')

    # ---------------------------
    # 2) Cell-type super-resolution
    # ---------------------------
    ct_names_file = f"{prefix}celltype-names.txt"
    if os.path.exists(ct_names_file):
        ct_names = read_lines(ct_names_file)

        ct_arrays = []
        for ct in ct_names:
            arr = load_pickle(f'{prefix}ct-super/{ct}.pickle')
            arr[~mask] = np.nan
            plot_super(arr, f'{prefix}ct-super-plots/{ct}.png')
            ct_arrays.append(arr)

        # --------------------------------------
        # 3) Argmax cell-type map + legend
        # --------------------------------------
        if len(ct_arrays) > 0:
            ct_stack = np.stack(ct_arrays, axis=-1)  # (H, W, n_ct)

            # Use -inf where NaN so they don't dominate argmax
            ct_stack_arg = np.where(np.isfinite(ct_stack), ct_stack, -np.inf)
            ct_idx = np.argmax(ct_stack_arg, axis=-1)  # (H, W)

            # Mask out background
            ct_idx[~mask] = -1  # background

            n_ct = len(ct_names)
            cmap = plt.get_cmap("tab20", n_ct)

            # Build RGB image with white background
            rgb = np.ones((*ct_idx.shape, 3), dtype=float)
            for i in range(n_ct):
                color = cmap(i)[:3]
                rgb[ct_idx == i] = color

            # Save raw image
            img = (rgb * 255).astype(np.uint8)
            save_image(img, f"{prefix}ct-super-plots/ct_argmax_map.png")

            # Also save a figure with a legend
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(rgb)
            ax.axis("off")

            handles = [Patch(color=cmap(i), label=ct_names[i]) for i in range(n_ct)]
            ax.legend(
                handles=handles,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                borderaxespad=0.,
                fontsize=6,
            )
            fig.tight_layout()
            fig.savefig(f"{prefix}ct-super-plots/ct_argmax_map_with_legend.png", dpi=300)
            plt.close(fig)



if __name__ == '__main__':
    main()
