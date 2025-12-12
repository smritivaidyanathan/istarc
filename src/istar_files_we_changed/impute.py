import argparse
import multiprocessing
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import numpy as np
import scanpy as sc
import pandas as pd
import os
from pathlib import Path
from scipy.spatial import cKDTree

# Assuming these are available in your environment
from impute_by_basic import get_gene_counts, get_embeddings, get_locs
from utils import read_lines, read_string, save_pickle
from image import get_disk_mask
from train import get_model as train_load_model
from visual import plot_matrix, plot_spot_masked_image


class FeedForward(nn.Module):

    def __init__(
            self, n_inp, n_out,
            activation=None, residual=False):
        super().__init__()
        self.linear = nn.Linear(n_inp, n_out)
        if activation is None:
            # TODO: change activation to LeakyRelu(0.01)
            activation = nn.LeakyReLU(0.1, inplace=True)
        self.activation = activation
        self.residual = residual

    def forward(self, x, indices=None):
        if indices is None:
            y = self.linear(x)
        else:
            weight = self.linear.weight[indices]
            bias = self.linear.bias[indices]
            y = nn.functional.linear(x, weight, bias)
        y = self.activation(y)
        if self.residual:
            y = y + x
        return y


class ELU(nn.Module):

    def __init__(self, alpha, beta):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta

    def forward(self, x):
        return self.activation(x) + self.beta


import torch.nn.functional as F


class ForwardSumModel(pl.LightningModule):

    def __init__(
            self, lr, n_inp, n_out,
            n_cell_types=None, ct_loss_weight=1.0, fake_weight=0.1):
        super().__init__()
        self.lr = lr
        self.n_cell_types = n_cell_types
        self.ct_loss_weight = ct_loss_weight
        self.fake_weight = fake_weight

        # Latent network: per-pixel embedding
        self.net_lat = nn.Sequential(
            FeedForward(n_inp, 256),
            FeedForward(256, 256),
            FeedForward(256, 256),
            FeedForward(256, 256),
        )

        # Gene output head: per-pixel gene expression
        self.net_out = FeedForward(
            256,
            n_out,
            activation=ELU(alpha=0.01, beta=0.01),
        )

        # Optional cell-type head: per-pixel logits over cell types from latent
        if n_cell_types is not None:
            self.ct_head = nn.Sequential(
                FeedForward(256, 128),
                FeedForward(128, n_cell_types, activation=nn.Identity()),
            )
        else:
            self.ct_head = None

        self.save_hyperparameters()

    def inp_to_lat(self, x):
        return self.net_lat.forward(x)

    def lat_to_out(self, z, indices=None):
        y = self.net_out.forward(z, indices)
        return y

    def celltype_logits_from_latent(self, z):
        if self.ct_head is None:
            raise RuntimeError("Cell-type head requested but n_cell_types was None.")
        return self.ct_head(z)


    def forward(self, x, indices=None):
        z = self.inp_to_lat(x)
        y_pred_pix = self.lat_to_out(z, indices)

        if self.ct_head is None:
            return y_pred_pix

        ct_pred_pix = self.celltype_logits_from_latent(z)
        return y_pred_pix, ct_pred_pix


    def training_step(self, batch, batch_idx):
        """
        Batch may be:
            (x, y_mean, is_real)
            (x, y_mean, ct_spot, is_real)
        """
        if len(batch) == 3:
            x, y_mean, is_real = batch
            ct_spot = None
        else:
            x, y_mean, ct_spot, is_real = batch

        # make sure boolean
        is_real = is_real.bool()

        # Forward pass
        if self.ct_head is None:
            y_pred_pix = self.forward(x)
            ct_pred_pix = None
        else:
            y_pred_pix, ct_pred_pix = self.forward(x)

        # Gene loss (spot-level)
        y_mean_pred = y_pred_pix.mean(dim=1)
        diff2 = (y_mean_pred - y_mean) ** 2          # (B, n_genes)
        gene_loss_per_spot = diff2.mean(dim=1)       # (B,)

        # weight fake spots lower
        weights = torch.where(is_real, torch.ones_like(gene_loss_per_spot),
                              torch.full_like(gene_loss_per_spot, self.fake_weight))

        gene_loss = (weights * gene_loss_per_spot).mean()
        self.log("rmse", torch.sqrt(gene_loss), prog_bar=True)

        loss = gene_loss


        # Cell-type loss only on real spots
        if ct_spot is not None and ct_pred_pix is not None:
            ct_pred_spot = ct_pred_pix.mean(dim=1)  # (B, C)

            real_mask = is_real
            if real_mask.any():
                ct_diff2 = (ct_pred_spot[real_mask] - ct_spot[real_mask]) ** 2
                ct_loss = ct_diff2.mean()
                loss = loss + self.ct_loss_weight * ct_loss
                self.log("ct_loss", ct_loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

class SpotDataset(Dataset):

    def __init__(self, x_all, y, locs, radius, ct_props=None, is_real=None):
        super().__init__()
        mask = get_disk_mask(radius)
        x = get_patches_flat(x_all, locs, mask)
        isin = np.isfinite(x).all((-1, -2))
        
        self.x = x[isin]
        self.y = y[isin]
        self.locs = locs[isin]
        self.size = x_all.shape[:2]
        self.radius = radius
        self.mask = mask

        if ct_props is not None:
            ct_props = np.asarray(ct_props, dtype=np.float32)
            assert ct_props.shape[0] == len(y)
            self.ct_props = ct_props[isin]
        else:
            self.ct_props = None
            
        if is_real is not None:
            is_real = np.asarray(is_real, dtype=bool)
            assert is_real.shape[0] == len(y)
            self.is_real = is_real[isin]
        else:
            # if not provided, treat everything as real
            self.is_real = np.ones(len(self.y), dtype=bool)


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.ct_props is None:
            return self.x[idx], self.y[idx], self.is_real[idx]
        else:
            return self.x[idx], self.y[idx], self.ct_props[idx], self.is_real[idx]

    def show(self, channel_x, channel_y, prefix):
        mask = self.mask
        size = self.size
        locs = self.locs
        xs = self.x
        ys = self.y

        plot_spot_masked_image(
                locs=locs, values=xs[:, :, channel_x], mask=mask, size=size,
                outfile=f'{prefix}x{channel_x:04d}.png')

        plot_spot_masked_image(
                locs=locs, values=ys[:, channel_y], mask=mask, size=size,
                outfile=f'{prefix}y{channel_y:04d}.png')


def get_disk(img, ij, radius):
    i, j = ij
    patch = img[i-radius:i+radius, j-radius:j+radius]
    disk_mask = get_disk_mask(radius)
    patch[~disk_mask] = 0.0
    return patch


def get_patches_flat(img, locs, mask):
    shape = np.array(mask.shape)
    center = shape // 2
    r = np.stack([-center, shape-center], -1)  # offset
    x_list = []
    for s in locs:
        patch = img[
                s[0]+r[0][0]:s[0]+r[0][1],
                s[1]+r[1][0]:s[1]+r[1][1]]
        if mask.all():
            x = patch
        else:
            x = patch[mask]
        x_list.append(x)
    x_list = np.stack(x_list)
    return x_list


def add_coords(embs):
    coords = np.stack(np.meshgrid(
            np.linspace(-1, 1, embs.shape[0]),
            np.linspace(-1, 1, embs.shape[1]),
            indexing='ij'), -1)
    coords = coords.astype(embs.dtype)
    mask = np.isfinite(embs).all(-1)
    coords[~mask] = np.nan
    embs = np.concatenate([embs, coords], -1)
    return embs


# Heat diffusion + fake spots
def load_heatdiff_maps(
    hd_dir="../data/heat_diffused_data",
    maps_fname="visium_mouse_brain_pixel_gene_maps.npy",
    genes_fname="visium_mouse_brain_pixel_gene_maps_genes.txt",
):
    """
    Load heat-diffused pixel-level gene maps and gene names.
    """
    maps_path = Path(hd_dir) / maps_fname
    genes_path = Path(hd_dir) / genes_fname

    hd_maps = np.load(maps_path)   # (H, W, G)
    hd_genes = np.array(read_lines(str(genes_path)))
    return hd_maps, hd_genes


def align_genes_between_cnts_and_heat(cnts, hd_genes):
    """
    Make sure we only use genes that appear in BOTH:
      - cnts columns (istar HVGs)
      - hd_genes (heat-diff output)
    """
    cnt_genes = np.array(cnts.columns, dtype=str)

    # Intersection & consistent ordering
    common = sorted(set(cnt_genes).intersection(set(hd_genes)))
    if len(common) == 0:
        raise ValueError(
            "[heatdiff] No overlapping genes between cnts.tsv and heat-diff gene list. "
            "Double-check that heat diffusion was run using the same gene_names.txt as iStar."
        )

    common = np.array(common, dtype=str)

    gene_to_idx_cnt = {g: i for i, g in enumerate(cnt_genes)}
    gene_to_idx_hd = {g: i for i, g in enumerate(hd_genes)}

    idx_cnt = np.array([gene_to_idx_cnt[g] for g in common], dtype=int)
    idx_hd = np.array([gene_to_idx_hd[g] for g in common], dtype=int)

    return common, idx_cnt, idx_hd


def make_fake_spot_locs(locs_real, image_shape, radius, n_fake_per_real=1, min_dist_mult=2.0):
    """
    Make 'fake' spot centers roughly between existing spots using midpoint heuristic.
    """
    H, W = image_shape
    locs_real = np.asarray(locs_real, dtype=int)
    N = locs_real.shape[0]
    if N < 2:
        return np.zeros((0, 2), dtype=int)

    # Precompute pairwise distances roughly
    tree = cKDTree(locs_real)
    min_dist = radius * min_dist_mult

    locs_fake = []
    rng = np.random.default_rng(0)

    target_fake = N * n_fake_per_real
    attempts = 0
    max_attempts = target_fake * 20

    while len(locs_fake) < target_fake and attempts < max_attempts:
        attempts += 1
        i = rng.integers(0, N)
        # nearest neighbor that is not i
        dists, idxs = tree.query(locs_real[i], k=3)
        # idxs[0] == i; take next neighbour that exists
        j = None
        for idx in idxs[1:]:
            if idx < N:
                j = idx
                break
        if j is None:
            continue

        mid = np.round((locs_real[i] + locs_real[j]) / 2.0).astype(int)
        r, c = mid
        if r < 0 or r >= H or c < 0 or c >= W:
            continue

        # far enough from any real spot
        dist, _ = tree.query(mid, k=1)
        if dist < min_dist:
            continue

        locs_fake.append(mid)

    if len(locs_fake) == 0:
        return np.zeros((0, 2), dtype=int)

    locs_fake = np.unique(np.stack(locs_fake, axis=0), axis=0)
    return locs_fake.astype(int)


def compute_fake_y_from_heat(hd_maps_common, fake_locs, radius):
    """
    For each fake spot center, compute the mean heat-diffused GE per gene inside the disk.
    """
    if fake_locs.shape[0] == 0:
        return np.zeros((0, hd_maps_common.shape[-1]), dtype=np.float32)

    H, W, G = hd_maps_common.shape
    disk_mask = get_disk_mask(radius)  # shape (Dh, Dw), center at middle
    Dh, Dw = disk_mask.shape
    center = np.array([Dh // 2, Dw // 2])
    
    # Generate relative coordinates for the disk mask
    offsets = np.stack(np.meshgrid(
        np.arange(Dh) - center[0],
        np.arange(Dw) - center[1],
        indexing="ij"
    ), axis=-1)  # (Dh, Dw, 2)

    y_list = []

    for (r0, c0) in fake_locs:
        # Absolute coordinates of all pixels in the square patch
        rr = offsets[..., 0] + r0
        cc = offsets[..., 1] + c0

        # Boolean mask: is inside image bounds AND inside the disk mask
        valid = (
            (rr >= 0) & (rr < H) &
            (cc >= 0) & (cc < W) &
            disk_mask
        )

        if not np.any(valid):
            y_list.append(np.zeros((G,), dtype=np.float32))
            continue

        # Extract values at valid locations
        vals = hd_maps_common[rr[valid], cc[valid], :]   # (N_pix, G)
        
        # Mean across the pixels for each gene
        y_list.append(vals.mean(axis=0).astype(np.float32))

    y_fake = np.stack(y_list, axis=0)
    return y_fake


def get_data(prefix):
    gene_names = read_lines(f'{prefix}gene-names.txt')
    cnts = get_gene_counts(prefix)
    cnts = cnts[gene_names]
    embs = get_embeddings(prefix)
    locs = get_locs(prefix, target_shape=embs.shape[:2])
    return embs, cnts, locs


def get_model_kwargs(kwargs):
    return get_model(**kwargs)


def get_model(
        x, y, locs, radius, prefix, batch_size, epochs, lr,
        load_saved=False, device='cuda',
        ct_props=None, is_real=None,
        ct_loss_weight=1.0, fake_weight=0.1):

    print('x:', x.shape, ', y:', y.shape)

    x = x.copy()

    # Pass is_real to the dataset
    dataset = SpotDataset(x, y, locs, radius, ct_props=ct_props, is_real=is_real)

    dataset.show(
            channel_x=0, channel_y=0,
            prefix=f'{prefix}training-data-plots/')
    model = train_load_model(
            model_class=ForwardSumModel,
            model_kwargs=dict(
                n_inp=x.shape[-1],
                n_out=y.shape[-1],
                lr=lr,
                n_cell_types=None if ct_props is None else ct_props.shape[1],
                ct_loss_weight=ct_loss_weight,
                fake_weight=fake_weight,
            ),
            dataset=dataset, prefix=prefix,
            batch_size=batch_size, epochs=epochs,
            load_saved=load_saved, device=device)
    model.eval()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return model, dataset


def normalize(embs, cnts):

    embs = embs.copy()
    cnts = cnts.copy()

    embs_mean = np.nanmean(embs, (0, 1))
    embs_std = np.nanstd(embs, (0, 1))
    embs -= embs_mean
    embs /= embs_std + 1e-12

    cnts_min = cnts.min(0)
    cnts_max = cnts.max(0)
    cnts -= cnts_min
    cnts /= (cnts_max - cnts_min) + 1e-12

    return embs, cnts, (embs_mean, embs_std), (cnts_min, cnts_max)


def show_results(x, names, prefix):
    for name in ['CD19', 'MS4A1', 'ERBB2', 'GNAS']:
        if name in names:
            idx = np.where(names == name)[0][0]
            plot_matrix(x[..., idx], prefix+name+'.png')


def predict_single_out(model, z, indices, names, y_range):
    z = torch.tensor(z, device=model.device)
    y = model.lat_to_out(z, indices=indices)
    y = y.cpu().detach().numpy()
    # y[y < 0.01] = 0.0
    # y[y > 1.0] = 1.0
    y *= y_range[:, 1] - y_range[:, 0]
    y += y_range[:, 0]
    return y


def predict_single_lat(model, x):
    x = torch.tensor(x, device=model.device)
    z = model.inp_to_lat(x)
    z = z.cpu().detach().numpy()
    return z


def predict(
        model_states, x_batches, name_list, y_range, prefix,
        device='cuda', ct_names=None):

    batch_size_outcome = 100

    model_states = [mod.to(device) for mod in model_states]


    # Latent embeddings per pixel
    z_states_batches = [
        [predict_single_lat(mod, x_bat) for mod in model_states]
        for x_bat in x_batches
    ]

    # Save median latent embedding as before
    z_point = np.concatenate([
        np.median(z_states, 0)
        for z_states in z_states_batches
    ])
    z_dict = dict(cls=z_point.transpose(2, 0, 1))
    save_pickle(z_dict, prefix + 'embeddings-gene.pickle')
    del z_point


    # Gene predictions (unchanged)
    idx_list = np.arange(len(name_list))
    n_groups_outcome = len(idx_list) // batch_size_outcome + 1
    idx_groups = np.array_split(idx_list, n_groups_outcome)

    os.makedirs(f'{prefix}cnts-super', exist_ok=True)

    for idx_grp in idx_groups:
        name_grp = name_list[idx_grp]
        y_ran = y_range[idx_grp]
        y_grp = np.concatenate([
            np.median([
                predict_single_out(mod, z, idx_grp, name_grp, y_ran)
                for mod, z in zip(model_states, z_states)
            ], 0)
            for z_states in z_states_batches
        ])
        for i, name in enumerate(name_grp):
            save_pickle(y_grp[..., i], f'{prefix}cnts-super/{name}.pickle')


    # Cell-type probabilities per pixel
    print(ct_names)
    print(getattr(model_states[0], "ct_head", None))
    if ct_names is not None and getattr(model_states[0], "ct_head", None) is not None:
        ct_names = list(ct_names)
        n_ct = len(ct_names)
        os.makedirs(f'{prefix}ct-super', exist_ok=True)

        # For each batch of z (across rows of the image), compute median probability per pixel
        ct_probs_batches = []
        for z_states in z_states_batches:
            # z_states: list over models, each (B, W, 256)
            ct_state_list = []
            for mod, z in zip(model_states, z_states):
                z_tensor = torch.tensor(z, device=mod.device)
                logits = mod.celltype_logits_from_latent(z_tensor)  # (B, W, n_ct)
                probs = torch.softmax(logits, dim=-1)
                probs = probs.detach().cpu().numpy()
                ct_state_list.append(probs)
            # median across models: (B, W, n_ct)
            ct_median = np.median(ct_state_list, axis=0)
            ct_probs_batches.append(ct_median)

        # Stitch along rows: (H, W, n_ct)
        ct_probs_full = np.concatenate(ct_probs_batches, axis=0)

        # Save one pickle per cell type
        for k, ct_name in enumerate(ct_names):
            save_pickle(ct_probs_full[..., k], f'{prefix}ct-super/{ct_name}.pickle')


def impute(
        embs, cnts, locs, radius, epochs, batch_size, prefix,
        n_states=1, load_saved=False, device='cuda', n_jobs=1,
        ct_props=None, ct_names=None, is_real=None,
        ct_loss_weight=1.0, fake_weight=0.1):
    """
    ct_props: optional spot-level cell-type proportions (n_spots, n_cell_types)
    ct_names: optional list of cell-type names (len = n_cell_types)
    is_real: boolean flag (N_all_spots,) for real vs fake
    """

    names = cnts.columns
    cnts = cnts.to_numpy()
    cnts = cnts.astype(np.float32)

    if ct_props is not None:
        ct_props = np.asarray(ct_props, dtype=np.float32)
        assert ct_props.shape[0] == cnts.shape[0], (
            f"ct_props has {ct_props.shape[0]} spots, "
            f"but cnts has {cnts.shape[0]}"
        )

    # The normalization here is on the already combined (and reduced) data
    __, cnts, __, (cnts_min, cnts_max) = normalize(embs, cnts)
    __, cnts, __, (cnts_min, cnts_max) = normalize(embs, cnts)

    kwargs_list = [
        dict(
            x=embs,
            y=cnts,
            locs=locs,
            radius=radius,
            batch_size=batch_size,
            epochs=epochs,
            lr=1e-4,
            prefix=f'{prefix}states/{i:02d}/',
            load_saved=load_saved,
            device=device,
            ct_props=ct_props,
            is_real=is_real,
            ct_loss_weight=ct_loss_weight,
            fake_weight=fake_weight,
        )
        for i in range(n_states)
    ]

    if n_jobs is None or n_jobs < 1:
        n_jobs = n_states
    if n_jobs == 1:
        out_list = [get_model_kwargs(kwargs) for kwargs in kwargs_list]
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            out_list = pool.map(get_model_kwargs, kwargs_list)

    model_list = [out[0] for out in out_list]
    dataset_list = [out[1] for out in out_list]
    mask_size = dataset_list[0].mask.sum()

    cnts_range = np.stack([cnts_min, cnts_max], -1)
    cnts_range /= mask_size

    batch_size_row = 50
    n_batches_row = embs.shape[0] // batch_size_row + 1
    embs_batches = np.array_split(embs, n_batches_row)
    del embs

    print(ct_names)

    predict(
        model_states=model_list,
        x_batches=embs_batches,
        name_list=names,
        y_range=cnts_range,
        prefix=prefix,
        device=device,
        ct_names=ct_names,
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--epochs', type=int, default=None)  # e.g. 400
    parser.add_argument('--n-states', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--load-saved', action='store_true')
    parser.add_argument('--data-dir', type=str, default='..',
                        help='Base directory where heat diffused data is stored.')
    parser.add_argument('--use-ct-supervision', action='store_true',
                        help='Use cell-type supervision from AnnData.')
    parser.add_argument('--use-fake-hd-spots', action='store_true',
                        help='Use heat-diffused maps to generate fake spots.')
    parser.add_argument('--hd-loss-weight', type=float, default=0.1,
                        help='Loss weight for fake (heat-diffused) spots.')
    parser.add_argument('--ct-loss-weight', type=float, default=1.0,
                        help='Cell-type loss weight.')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(f"Cell-type supervision: {'enabled' if args.use_ct_supervision else 'disabled'} "
          f"(ct_loss_weight={args.ct_loss_weight})")
    print(f"Fake heat-diffused spots: {'enabled' if args.use_fake_hd_spots else 'disabled'} "
          f"(hd_loss_weight={args.hd_loss_weight})")
    print(f"Training device: {args.device}, states: {args.n_states}")

    embs, cnts, locs = get_data(args.prefix)  # cnts is a DataFrame

    factor = 16
    radius = int(read_string(f'{args.prefix}radius.txt'))
    radius = radius / factor

    ct_props_real = None
    ct_names = None

    if args.use_ct_supervision:
        print("Using cell-type supervision from ../results/st.h5ad")
        adata = sc.read_h5ad("../results/st.h5ad")

        rows_star = adata.obs["array_row"].astype(int).to_numpy()
        cols_star = adata.obs["array_col"].astype(int).to_numpy()
        spot_star = np.array([f"{r}x{c}" for r, c in zip(rows_star, cols_star)])

        spot_model = cnts.index.to_numpy() if "spot" not in cnts.columns else cnts["spot"].to_numpy()

        if np.array_equal(spot_star, spot_model):
            ct_props_real = adata.obsm["qc_m"].astype("float32")
        else:
            print("Spot orders differ between Starfysh AnnData and cnts; realigning by spot IDs.")
            if set(spot_model) != set(spot_star):
                missing_in_star = set(spot_model) - set(spot_star)
                missing_in_model = set(spot_star) - set(spot_model)
                raise ValueError(
                    f"Spot sets do not match.\n"
                    f"Missing in Starfysh: {len(missing_in_star)}\n"
                    f"Missing in model: {len(missing_in_model)}"
                )
            ct_df = pd.DataFrame(adata.obsm["qc_m"], index=spot_star)
            ct_df = ct_df.loc[spot_model]
            ct_props_real = ct_df.to_numpy(dtype="float32")

        print("ct_props_real shape:", ct_props_real.shape)

        if "cell_types" in adata.uns:
            ct_names = np.array(adata.uns["cell_types"])
        else:
            ct_names = np.array([f"ct_{i}" for i in range(ct_props_real.shape[1])])

        with open(f"{args.prefix}celltype-names.txt", "w") as f:
            for name in ct_names:
                f.write(str(name) + "\n")
    else:
        print("Cell-type supervision disabled; skipping ct_props loading.")

    # Heat Diffusion (fake-spots)
    if args.use_fake_hd_spots:
        print("Generating fake spots from heat-diffused maps.")
        hd_maps, hd_genes = load_heatdiff_maps(hd_dir=f"{args.data_dir}") 
        common_genes, idx_cnt, idx_hd = align_genes_between_cnts_and_heat(cnts, hd_genes)

        cnts_common = cnts.iloc[:, idx_cnt].copy()
        hd_maps_common = hd_maps[..., idx_hd]  # (H, W, G_common)

        locs_real = locs.copy()
        print(f"{locs_real.shape} real spots")
        H, W = embs.shape[:2]
        locs_fake = make_fake_spot_locs(locs_real, image_shape=(H, W),
                                        radius=radius, n_fake_per_real=1,
                                        min_dist_mult=1.0)
        print(f"[fake spots] made {locs_fake.shape} fake spots")

        y_fake = compute_fake_y_from_heat(hd_maps_common, locs_fake, radius=radius)
        print("y_fake shape:", y_fake.shape)

        y_real = cnts_common.to_numpy(dtype=np.float32)

        cnts_all = np.concatenate([y_real, y_fake], axis=0)  # (N_real + N_fake, G_common)
        locs_all = np.concatenate([locs_real, locs_fake], axis=0)

        if args.use_ct_supervision and ct_props_real is not None:
            n_ct = ct_props_real.shape[1]
            zeros_fake = np.zeros((locs_fake.shape[0], n_ct), dtype=np.float32)
            ct_props_all = np.concatenate([ct_props_real, zeros_fake], axis=0)
        else:
            ct_props_all = None

        is_real = np.concatenate([
            np.ones(locs_real.shape[0], dtype=bool),
            np.zeros(locs_fake.shape[0], dtype=bool),
        ])

        cnts_for_training = pd.DataFrame(cnts_all, columns=common_genes)
        locs_for_training = locs_all
        ct_props_for_training = ct_props_all
        is_real_flag = is_real
    else:
        print("Skipping fake heat-diffused spots; training on real spots only.")
        cnts_for_training = cnts
        locs_for_training = locs
        ct_props_for_training = ct_props_real if args.use_ct_supervision else None
        is_real_flag = None

    n_train = cnts_for_training.shape[0]
    batch_size = min(128, max(1, n_train // 16))

    impute(
        embs=embs,
        cnts=cnts_for_training,
        locs=locs_for_training,
        radius=radius,
        epochs=args.epochs,
        batch_size=batch_size,
        n_states=args.n_states,
        prefix=args.prefix,
        load_saved=args.load_saved,
        device=args.device,
        n_jobs=args.n_jobs,
        ct_props=ct_props_for_training,
        ct_names=ct_names if args.use_ct_supervision else None,
        is_real=is_real_flag,
        fake_weight=args.hd_loss_weight,
        ct_loss_weight=args.ct_loss_weight,
    )


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
