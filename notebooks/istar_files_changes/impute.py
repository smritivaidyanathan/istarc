import argparse
import multiprocessing

import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import numpy as np

import scanpy as sc
import pandas as pd
import os

from impute_by_basic import get_gene_counts, get_embeddings, get_locs
from utils import read_lines, read_string, save_pickle
from image import get_disk_mask
from train import get_model as train_load_model
# from reduce_dim import reduce_dim
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

    def __init__(self, lr, n_inp, n_out, n_cell_types=None, ct_loss_weight=1.0):
        """
        lr: learning rate
        n_inp: input feature dimension per pixel
        n_out: number of genes (output dimension per pixel)
        n_cell_types: number of cell types for the classification head (optional)
        ct_loss_weight: weight for the cell-type loss term
        """
        super().__init__()
        self.lr = lr
        self.n_cell_types = n_cell_types
        self.ct_loss_weight = ct_loss_weight

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
        """
        x: (B, Npix, n_inp)
        returns z: (B, Npix, 256)
        """
        return self.net_lat.forward(x)

    def lat_to_out(self, z, indices=None):
        """
        z: (B, Npix, 256)
        indices: optional subset of gene indices (for weight slicing in predict_single_out)
        returns y_pred: (B, Npix, n_out or len(indices))
        """
        y = self.net_out.forward(z, indices)
        return y

    def celltype_logits_from_latent(self, z):
        """
        z: (B, Npix, 256)
        returns logits: (B, Npix, n_cell_types)
        """
        if self.ct_head is None:
            raise RuntimeError("Cell-type head requested but n_cell_types was None.")
        return self.ct_head(z)


    def forward(self, x, indices=None):
        """
        Always returns:
            y_pred_pix                    # (B, Npix, n_genes_sel)

        If n_cell_types is provided, also returns:
            ct_pred_pix                   # (B, Npix, n_cell_types)
        """
        z = self.inp_to_lat(x)

        y_pred_pix = self.lat_to_out(z, indices)

        if self.ct_head is None:
            return y_pred_pix

        ct_pred_pix = self.celltype_logits_from_latent(z)
        return y_pred_pix, ct_pred_pix


    def training_step(self, batch, batch_idx):
        """
        Batch may be:
            (x, y_mean)
            (x, y_mean, ct_spot)
        """
        if len(batch) == 2:
            x, y_mean = batch
            ct_spot = None
        else:
            x, y_mean, ct_spot = batch

        # Forward pass
        if self.ct_head is None:
            y_pred_pix = self.forward(x)
            ct_pred_pix = None
        else:
            y_pred_pix, ct_pred_pix = self.forward(x)

        # Gene loss (per-pixel -> spot-level)
        y_mean_pred = y_pred_pix.mean(dim=1)          # (B, n_genes)
        gene_loss = ((y_mean_pred - y_mean) ** 2).mean()

        loss = gene_loss
        self.log("rmse", gene_loss ** 0.5, prog_bar=True)

        # Cell Type loss (per-pixel -> spot-level)
        if ct_spot is not None and ct_pred_pix is not None:
            ct_pred_spot = ct_pred_pix.mean(dim=1)    # (B, C)
            ct_loss = ((ct_pred_spot - ct_spot) ** 2).mean()
            loss = loss + self.ct_loss_weight * ct_loss
            self.log("ct_loss", ct_loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer


class SpotDataset(Dataset):

    def __init__(self, x_all, y, locs, radius, ct_props=None):
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

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.ct_props is None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx], self.y[idx], self.ct_props[idx]

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


# def reduce_embeddings(embs):
#     # cls features
#     cls, __ = reduce_dim(embs[..., :192], 0.99)
#     # sub features
#     sub, __ = reduce_dim(embs[..., 192:-3], 0.90)
#     rgb = embs[..., -3:]
#     embs = np.concatenate([cls, sub, rgb], -1)
#     return embs


def get_data(prefix):
    gene_names = read_lines(f'{prefix}gene-names.txt')
    cnts = get_gene_counts(prefix)
    cnts = cnts[gene_names]
    embs = get_embeddings(prefix)
    # embs = embs[..., :192]  # use high-level features only
    # embs = reduce_embeddings(embs)
    locs = get_locs(prefix, target_shape=embs.shape[:2])
    # embs = add_coords(embs)
    return embs, cnts, locs


def get_model_kwargs(kwargs):
    return get_model(**kwargs)


def get_model(
        x, y, locs, radius, prefix, batch_size, epochs, lr,
        load_saved=False, device='cuda',
        ct_props=None):

    print('x:', x.shape, ', y:', y.shape)

    x = x.copy()

    dataset = SpotDataset(x, y, locs, radius, ct_props=ct_props)

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

    # TODO: check if adjsut_weights in extract_features can be skipped
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


# def cluster_lat(x, prefix, device='cuda'):
#     x_minor = x
#     x_major = smoothen(
#             x_minor, size=8, method='cnn', mode='mean',
#             device=device)
#     labels = cluster_hierarchical(
#             x_major.transpose(2, 0, 1), x_minor.transpose(2, 0, 1),
#             method='km', n_clusters=10)
#     # x = reduce_dim(x, method='pca', n_components=0.95)[0]
#     # labels_raw = cluster(
#     #         x.transpose(2, 0, 1), method='km', n_clusters=10)[0]
#     # labels_cls = relabel_small_connected(labels_raw, min_size=1000)
#     # labels_con = cluster_connected(labels_cls)
#     # labels = np.stack([labels_cls, labels_con], -1)
#     plot_labels(labels[..., :2], prefix+'clusters-genes.png')
#     save_pickle(labels, prefix+'clusters-genes.pickle')
#     return labels


def predict(
        model_states, x_batches, name_list, y_range, prefix,
        device='cuda', ct_names=None):
    """
    model_states: list of trained ForwardSumModel instances
    x_batches: list of x batches (chunks of the embedding image)
    name_list: list/array of gene names
    y_range: per-gene min/max range (cnts_range)
    prefix: output prefix (directory)
    device: 'cuda' or 'cpu'
    ct_names: optional list/array of cell type names (len = n_cell_types)
    """

    batch_size_outcome = 100

    model_states = [mod.to(device) for mod in model_states]

    # --------------------------------
    # 1) Latent embeddings per pixel
    # --------------------------------
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

    # --------------------------------
    # 2) Gene predictions (unchanged)
    # --------------------------------
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

    # --------------------------------
    # 3) Cell-type probabilities per pixel
    # --------------------------------
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
        ct_props=None, ct_names=None):
    """
    ct_props: optional spot-level cell-type proportions (n_spots, n_cell_types)
    ct_names: optional list of cell-type names (len = n_cell_types)
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
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    embs, cnts, locs = get_data(args.prefix)  # cnts is a DataFrame

    # ----------------------------
    # Load Starfysh cell-type proportions
    # ----------------------------
    # 1) Load Starfysh AnnData
    adata = sc.read_h5ad("../results/st.h5ad")

    # 2) Rebuild Starfysh spot IDs in the same "rowxcol" format
    rows_star = adata.obs["array_row"].astype(int).to_numpy()
    cols_star = adata.obs["array_col"].astype(int).to_numpy()
    spot_star = np.array([f"{r}x{c}" for r, c in zip(rows_star, cols_star)])

    # 3) Get the spot order used by your model (from cnts)
    #    This assumes cnts has a "spot" column, like in your TSV writer.
    spot_model = cnts.index.to_numpy() if "spot" not in cnts.columns else cnts["spot"].to_numpy()

    # 4) OPTIONAL: simple check if orders are exactly identical
    if np.array_equal(spot_star, spot_model):
        # Fast path: same order, just take qc_m as-is
        ct_props = adata.obsm["qc_m"].astype("float32")
    else:
        # Safe path: align by spot IDs
        print("Spot orders differ between Starfysh AnnData and cnts; realigning by spot IDs.")

        if set(spot_model) != set(spot_star):
            missing_in_star = set(spot_model) - set(spot_star)
            missing_in_model = set(spot_star) - set(spot_model)
            raise ValueError(
                f"Spot sets do not match.\n"
                f"Missing in Starfysh: {len(missing_in_star)}\n"
                f"Missing in model: {len(missing_in_model)}"
            )

        # Build DataFrame indexed by Starfysh spot IDs
        ct_df = pd.DataFrame(adata.obsm["qc_m"], index=spot_star)

        # Reindex to model's spot order
        ct_df = ct_df.loc[spot_model]
        ct_props = ct_df.to_numpy(dtype="float32")

    print("ct_props shape:", ct_props.shape)


    # Cell type names from Starfysh
    if "cell_types" in adata.uns:
        ct_names = np.array(adata.uns["cell_types"])
    else:
        ct_names = np.array([f"ct_{i}" for i in range(ct_props.shape[1])])

    # Optional: save for the plotting script
    with open(f"{args.prefix}celltype-names.txt", "w") as f:
        for name in ct_names:
            f.write(str(name) + "\n")

    factor = 16
    radius = int(read_string(f'{args.prefix}radius.txt'))
    radius = radius / factor

    n_train = cnts.shape[0]
    batch_size = min(128, n_train // 16)

    impute(
        embs=embs, cnts=cnts, locs=locs, radius=radius,
        epochs=args.epochs, batch_size=batch_size,
        n_states=args.n_states, prefix=args.prefix,
        load_saved=args.load_saved,
        device=args.device, n_jobs=args.n_jobs,
        ct_props=ct_props,
        ct_names=ct_names,
    )



if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
