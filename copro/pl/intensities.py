import warnings
from functools import partial
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import seaborn as sns
import anndata as ad
import math
import os
from scipy import sparse 

from .utils import _resolve_color_scheme
from copro.pp.var import is_log_transformed

def peptide_intensities(
    adata,
    protein_ids=None,
    protein_id_key='protein_id', # Implement
    group_by=None,
    group_by_order=None,
    color=None,
    log_transform=None,
    z_transform=False,
    show_zeros=True,
    xlab_rotation=0,
    group_by_label_rotation=0,
    figsize=(15,6),
    show=True,
    save=None,
    ax=False,
    color_scheme=None,
    ):
    '''
    Args:
        log_transform (float, optional): Base for log transformation of the data. 1 will
            be added to each value before transformation.
        z_transform (float, optional): Transform values to have 0-mean and 1-variance
            along the peptide axis. Always uses zeros instead of NaNs if present, even
            if show_zeros=False.
        show_zeros (bool, optional): Don't display zeros if False.
    Returns:
    '''

    # Check input
    if isinstance(protein_ids, str):
        protein_ids = [protein_ids]

    assert protein_id_key in adata.var.columns

    # Format input
    var_cols = [protein_id_key]

    if color:
        var_cols.append(color)

    var = adata.var[var_cols].copy()
    var = var.reset_index().rename(columns={'index': 'var_index'})

    obs = adata.obs[[group_by]].copy()
    obs = obs.reset_index().rename(columns={'index': 'obs_index'})

    if is_categorical_dtype(obs[group_by]):
        obs[group_by] = obs[group_by].astype('category')

    X = adata.to_df().copy()

    X_zeros = X  # backup
    if not show_zeros:
        X = X.replace(0, np.nan)

    if log_transform:
        X = X.apply(lambda x: np.log(x+1) / np.log(log_transform))

    if z_transform:
        arr_zeros = X_zeros.to_numpy()
        arr_z = (
            (arr_zeros - np.mean(arr_zeros, axis=0, keepdims=True)) / 
            np.std(arr_zeros, axis=0, keepdims=True)
            )

        if not show_zeros:
            arr_z = np.where((arr_zeros == 0) & (arr_z != 0), np.nan, arr_z)
                         
        X = pd.DataFrame(arr_z, columns=X.columns, index=X.index)

    X = X.reset_index().rename(columns={'index': 'obs_index'})

    df = X.melt(id_vars='obs_index', var_name='var_index', value_name='intensity')
    df = pd.merge(df, var, on='var_index', how='left')
    df = pd.merge(df, obs, on='obs_index', how='left')

    # Explicitly order the x axis observations to group by group_by
    cat_index_map = {
        cat: sorted(obs[obs[group_by] == cat]['obs_index'].to_list())
        for cat in obs[group_by].cat.categories
        }

    if group_by_order:
        obs_index_ordered = [
            idx
            for cat in group_by_order
            for idx in cat_index_map[cat]
            ]

    else:
        obs_index_ordered = [
            idx
            for idx_list in cat_index_map.values()
            for idx in idx_list
            ]

    df['obs_index'] = pd.Categorical(
        df['obs_index'],
        categories=obs_index_ordered,
        ordered=True)

    axes = []

    if save and len(protein_ids) > 1:
        save_path = save if save.endswith('.pdf') else f'{save}.pdf'
        pdf_pages = PdfPages(save_path)

    for prot_id in protein_ids:
        sub_df = df[df[protein_id_key] == prot_id]
        fig, _ax = plt.subplots(figsize=figsize)

        if sub_df.empty:
            warnings.warn(f'No data found for protein: {prot_id}')
            _ax.text(
                0.5,
                0.5,
                f'No data found for protein: {prot_id}',
                ha='center', va='center', transform=_ax.transAxes,
                fontsize=14,
                color='gray'
                )
            _ax.set_xlim(0, 1)
            _ax.set_ylim(0, 1)
            _ax.set_xticks([])
            _ax.set_yticks([])

        else:

            #sub_df = sub_df.sort_values(by=group_by)

            if color:
                sns.lineplot(
                    data=sub_df,
                    x='obs_index',
                    y='intensity',
                    hue=color,
                    style=group_by,
                    units='var_index',
                    estimator=None,
                    errorbar=None,
                    marker='o',
                    dashes=False,
                    palette='Set2',
                    legend='brief',
                    ax=_ax,
                    )
            else:
                sns.lineplot(
                    data=sub_df,
                    x='obs_index',
                    y='intensity',
                    hue='var_index',
                    style=group_by,
                    marker='o',
                    dashes=False,
                    palette='Set2',
                    legend='brief',
                    ax=_ax
                    )

            # Legend
            handles, labels = _ax.get_legend_handles_labels()

            # Determine which labels correspond to the hue only (ignore style)
            if color:
                hue_values = sub_df[color].unique().astype(str)
            else:
                hue_values = sub_df['var_index'].unique().astype(str)

            # Keep only legend entries whose label matches a hue value
            new_handles_labels = [(h, l) for h, l in zip(handles, labels) if l in hue_values]

            if new_handles_labels:
                handles, labels = zip(*new_handles_labels)  # unzip back into separate lists
                _ax.legend(
                    handles,
                    labels,
                    bbox_to_anchor=(1.01, 1),
                    loc='upper left',
                    title=color if color else 'Peptide',
                    )

            # Add group separator lines
            obs_idxpos_map = {obs: i for i, obs in enumerate(obs_index_ordered)}
            cats_ordered = group_by_order if group_by_order else list(cat_index_map.keys())

            for cat in cats_ordered[:-1]:
                last_obs_in_cat = cat_index_map[cat][-1]

                _ax.axvline(
                    x=obs_idxpos_map[last_obs_in_cat] + 0.5,
                    ymin=0.02,
                    ymax=0.95,
                    color='#D8D8D8',
                    linestyle='--'
                    )

            # Add group labels above each group section
            for cat in cats_ordered:
                group_obs = cat_index_map[cat]

                if not group_obs:
                    continue

                # Determine x-axis group regions
                start = obs_idxpos_map[group_obs[0]]
                end = obs_idxpos_map[group_obs[-1]]
                mid = (start + end) / 2

                rot = group_by_label_rotation if group_by_label_rotation else 0
                ha_for_rot = 'center' if (rot % 360 == 0) else 'left'

                # Determine padded y-axis limits
                ymax = sub_df['intensity'].max()
                ymin = sub_df['intensity'].min()
                ypad_top = (ymax - ymin) * 0.15
                ypad_bottom = (ymax - ymin) * 0.10
                _ax.set_ylim(ymin - ypad_bottom, ymax + ypad_top)

                _ax.text(
                    x=mid,
                    y=ymax + ypad_top * 0.4,
                    s=cat,
                    ha=ha_for_rot,
                    va='bottom',
                    fontsize=12,
                    fontweight='bold',
                    rotation=rot,
                    rotation_mode='anchor',
                    )
 
        plt.xticks(rotation=xlab_rotation, ha='right')
        _ax.set_title(prot_id)
        _ax.set_xlabel('Sample')
        _ax.set_ylabel('Intensity')

        plt.tight_layout()

        if ax:
            axes.append(_ax)

            if show:
                plt.show()


        elif save:

            if len(protein_ids) == 1:
                fig.savefig(save, bbox_inches='tight', dpi=300)

            else:
                pdf_pages.savefig(fig, bbox_inches='tight')

            if show:
                plt.show()

            plt.close(fig)

        elif show:
            plt.show()
            plt.close(fig)

        else:
            print("Warning: Plot created but not displayed, saved, or returned")
            plt.close(fig)

    if save and len(protein_ids) > 1:
        pdf_pages.close()

    if ax:
        return axes[0] if len(axes) == 1 else axes


proteoform_intensities = partial(
    peptide_intensities,
    color = 'proteoform_id',
    )


def intensity_distribution_per_obs(
    adata,
    group_by=None,
    group_by_order=None,
    zero_to_na=False,
    log_transform = False,
    z_transform = False,
    ylabel='Intensity',
    xlabel_rotation=90,
    group_by_label_rotation=0,
    show=True,
    ax=False,
    save=False,
    figsize=(8,5),
    color_scheme=None,
    ):
    
    df = adata.to_df()

    if zero_to_na:
        df = df.replace(0, np.nan)

    if log_transform:
        df = df.apply(lambda x: np.log(x+1) / np.log(log_transform))

    if z_transform:
        df = df.apply(lambda row: (row - row.mean(skipna=True)) / row.std(skipna=True), axis=1)

    df = pd.melt(
        df.reset_index(names='obs'),
        id_vars='obs',
        var_name='var',
        value_name='intensity'
    )
    df = df[~df['intensity'].isna()]

    # Merge group info
    if group_by and group_by != 'obs':
        obs = adata.obs[[group_by]].reset_index(names='obs')
        df = pd.merge(df, obs, on='obs')
    if group_by == 'obs':
        df[group_by] = df['obs']
    if not group_by:
        group_by = 'all'
        df[group_by] = 'all'

    # Determine x-axis order
    obs_df = adata.obs.reset_index().rename(columns={'index': 'obs'})
    if group_by in obs_df.columns and isinstance(obs_df[group_by].dtype, pd.CategoricalDtype):
        obs_df[group_by] = obs_df[group_by].astype('category')

    if group_by_order:
        cat_index_map = {cat: sorted(obs_df[obs_df[group_by] == cat]['obs'].to_list())
                         for cat in group_by_order}
    else:
        cat_index_map = {cat: sorted(obs_df[obs_df[group_by] == cat]['obs'].to_list())
                         for cat in obs_df[group_by].unique()}

    x_ordered = [obs for cat in cat_index_map.values() for obs in cat]
    df['obs'] = pd.Categorical(df['obs'], categories=x_ordered, ordered=True)

    # Assign colors per group
    df[group_by] = df[group_by].astype(str)
    unique_groups = list(cat_index_map.keys())
    color_scheme
    colors = _resolve_color_scheme(color_scheme, unique_groups)
    color_map = {str(grp): colors[i] for i, grp in enumerate(unique_groups)}

    sample_palette = {obs: color_map[df.loc[df['obs'] == obs, group_by].iloc[0]] for obs in x_ordered}

    fig, _ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=df,
        x='obs',
        hue='obs',
        y='intensity',
        palette=sample_palette,
        ax=_ax
    )

    plt.setp(_ax.get_xticklabels(), rotation=xlabel_rotation, ha='right')
    _ax.set_xlabel('')
    _ax.set_ylabel(ylabel)

    # Add group labels above groups
    obs_idx_map = {obs: i for i, obs in enumerate(x_ordered)}
    ymax = df['intensity'].max()
    for cat, obs_list in cat_index_map.items():
        if not obs_list:
            continue
        start_idx = obs_idx_map[obs_list[0]]
        end_idx = obs_idx_map[obs_list[-1]]
        mid_idx = (start_idx + end_idx) / 2
        _ax.text(
            x=mid_idx,
            y=ymax * 1.05,
            s=cat,
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
            rotation=group_by_label_rotation
        )

    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    if ax:
        return _ax

def intensity_hist_imputed(
    adata,
    layer: str | None = None,
    bool_layer: str = "bool_imputed",
    bins: int = 60,
    density: bool = True,
    kde: bool = False,
    figsize=(7, 4),
    palette: dict | None = None,      # {'Measured': '#4C78A8', 'Imputed': '#F58518'}
    alpha: float = 0.6,
    title: str | None = None,
    legend_loc: str = "upper right",
    per_sample: bool = False,
    samples: list | None = None,      # list of obs names or integer indices (subset/order)
    ncols: int = 4,
    sharex: bool = True,
    sharey: bool = True,
    show: bool = True,
    save: bool | str | os.PathLike = False,
):
    """
    Plot histogram(s) of intensities in log2 scale colored by imputation status.

    - Auto-detects log; if raw, plots in log2 (<=0 -> NA).
    - Colors: 'Measured' vs 'Imputed' from layers[bool_layer].
    - If per_sample=False: one combined histogram.
      per_sample=True : grid of per-sample subplots (shared bins & one legend).

    Parameters
    ----------
    show : bool
        If True, call plt.show() at the end.
    save : bool | str | Path
        If True, save to a default filename.
        If str/Path, save to that path. If False, do not save.
    """
    # --- pull data ---
    Xsrc = adata.layers[layer] if layer is not None else adata.X
    X = Xsrc.toarray() if sparse.issparse(Xsrc) else np.asarray(Xsrc, dtype=float)

    if bool_layer not in adata.layers:
        raise KeyError(f"'{bool_layer}' not found in adata.layers")
    Bsrc = adata.layers[bool_layer]
    B = Bsrc.toarray() if sparse.issparse(Bsrc) else np.asarray(Bsrc)
    if B.shape != X.shape:
        raise ValueError(f"Shape mismatch: data {X.shape} vs {bool_layer} {B.shape}")

    # log2 for plotting (if needed)
    is_log, _ = is_log_transformed(adata, layer=layer)
    Y = X.copy()
    if not is_log:
        Y[~np.isfinite(Y) | (Y <= 0)] = np.nan
        Y = np.log2(Y)
    else:
        Y[~np.isfinite(Y)] = np.nan

    # palette & order
    if palette is None:
        palette = {"Measured": "#4C78A8", "Imputed": "#F58518"}
    hue_order = ["Measured", "Imputed"]

    # ------- Single (combined) histogram -------
    if not per_sample:
        vals = Y.ravel()
        flags = B.astype(bool).ravel()
        m = np.isfinite(vals)
        vals = vals[m]
        flags = flags[m]
        if vals.size == 0:
            raise ValueError("No finite values to plot after preprocessing.")

        status = np.where(flags, "Imputed", "Measured")
        df = pd.DataFrame({"intensity_log2": vals, "status": status})

        bin_edges = np.histogram_bin_edges(df["intensity_log2"].to_numpy(), bins=bins)

        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(
            data=df,
            x="intensity_log2",
            hue="status",
            hue_order=[h for h in hue_order if (df["status"] == h).any()],
            bins=bin_edges,
            stat=("density" if density else "count"),
            multiple="layer",
            common_norm=False,
            palette=palette,
            alpha=alpha,
            edgecolor=None,
            ax=ax,
            legend=False,
        )

        if kde:
            for k, g in df.groupby("status"):
                if len(g) > 1:
                    sns.kdeplot(g["intensity_log2"], ax=ax, color=palette.get(k), lw=1.5)

        ax.set_xlabel("Intensity (log2)")
        ax.set_ylabel("Density" if density else "Count")
        ax.set_title(title or "Intensity histogram (log2; colored by imputation)")

        present = [h for h in hue_order if (df["status"] == h).any()]
        handles = [Patch(facecolor=palette[h], edgecolor="none", alpha=alpha, label=h) for h in present]
        ax.legend(handles=handles, title="Status", loc=legend_loc, frameon=False)

        # save/show
        if save:
            path = save if isinstance(save, (str, os.PathLike)) else "intensity_hist_imputed.png"
            fig.savefig(path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return  # nothing else returned

    # ------- Per-sample small multiples -------
    # select samples
    if samples is None:
        idx = np.arange(adata.n_obs)
        labels = adata.obs_names.to_numpy()
    else:
        idx, labels = [], []
        for s in samples:
            if isinstance(s, (int, np.integer)):
                idx.append(int(s)); labels.append(adata.obs_names[int(s)])
            else:
                where = np.where(adata.obs_names == str(s))[0]
                if where.size == 0:
                    raise KeyError(f"Sample '{s}' not in adata.obs_names")
                idx.append(int(where[0])); labels.append(str(s))
        idx = np.asarray(idx, dtype=int)
        labels = np.asarray(labels, dtype=object)

    # global bins across all selected samples
    all_vals = []
    for i in idx:
        vi = Y[i, :]
        m = np.isfinite(vi)
        if m.any():
            all_vals.append(vi[m])
    if len(all_vals) == 0:
        raise ValueError("No finite values to plot after preprocessing across selected samples.")
    bin_edges = np.histogram_bin_edges(np.concatenate(all_vals), bins=bins)

    n = len(idx)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharex=sharex, sharey=sharey)

    present_any = set()
    for k, i in enumerate(idx):
        r, c = divmod(k, ncols)
        ax = axes[r, c]

        vi = Y[i, :]
        bi = B[i, :].astype(bool)

        m = np.isfinite(vi)
        vi = vi[m]
        bi = bi[m]
        if vi.size == 0:
            ax.set_visible(False)
            continue

        status = np.where(bi, "Imputed", "Measured")
        df_i = pd.DataFrame({"intensity_log2": vi, "status": status})
        present = [h for h in hue_order if (df_i["status"] == h).any()]
        present_any.update(present)

        sns.histplot(
            data=df_i,
            x="intensity_log2",
            hue="status",
            hue_order=present,
            bins=bin_edges,
            stat=("density" if density else "count"),
            multiple="layer",
            common_norm=False,
            palette=palette,
            alpha=alpha,
            edgecolor=None,
            ax=ax,
            legend=False,
        )

        if kde:
            for lab in present:
                g = df_i[df_i["status"] == lab]
                if len(g) > 1:
                    sns.kdeplot(g["intensity_log2"], ax=ax, color=palette.get(lab), lw=1.2)

        ax.set_title(str(labels[k]))
        if r == nrows - 1:
            ax.set_xlabel("Intensity (log2)")
        else:
            ax.set_xlabel("")
        if c == 0:
            ax.set_ylabel("Density" if density else "Count")
        else:
            ax.set_ylabel("")

    # hide any extra axes
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].set_visible(False)

    # global legend (figure-level unless user asked for 'best')
    present_any = [h for h in hue_order if h in present_any]
    handles = [Patch(facecolor=palette[h], edgecolor="none", alpha=alpha, label=h) for h in present_any]
    if legend_loc == "best":
        axes[0, 0].legend(handles=handles, title="Status", loc="best", frameon=False)
    else:
        fig.legend(handles=handles, title="Status", loc=legend_loc, frameon=False)

    plt.suptitle(title or "Intensity histograms per sample (log2; colored by imputation)", y=0.995, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # save/show
    if save:
        path = save if isinstance(save, (str, os.PathLike)) else "intensity_hist_imputed-per-sample.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return  # nothing returned
