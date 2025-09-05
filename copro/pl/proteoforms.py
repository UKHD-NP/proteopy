import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
from .peptides import peptide_intensities


def proteoform_scores(
    adata,
    adj=True,
    pval_threshold=None,
    score_threshold=None,
    ax=False,
    ):

    if adj:
        pval_col = 'proteoform_score_pval_adj'
        ylabel = '-log10(adj. p-value)'
    else:
        pval_col = 'proteoform_score_pval'
        ylabel = '-log10(p-value)'

    var = adata.var[['proteoform_score', pval_col]].copy()
    var = var.drop_duplicates() # pval and proteoform_score are repeated 
                             # for peptides of same protein

    mask_nonan = var[pval_col].notna()
    df = var.loc[mask_nonan,['proteoform_score', pval_col]]
    df['neg_log10_pval'] = -np.log10(df[pval_col].replace(0, np.nan))

    # Mask for pval and score thresholds
    if pval_threshold and score_threshold:
        mask = (
            (df['proteoform_score'] > score_threshold) &
            (df['neg_log10_pval'] > -np.log10(pval_threshold))
            )
    elif score_threshold:
        mask = df['proteoform_score'] > score_threshold
    elif pval_threshold:
        mask = df['neg_log10_pval'] > -np.log10(pval_threshold)
    else:
        mask = pd.Series(False, index=df.index)

    df['is_above_threshold'] = mask

    # Rel plot
    g = sns.relplot(
        data=df,
        x='proteoform_score',
        y='neg_log10_pval',
        hue='is_above_threshold',
        palette={
            np.True_: '#008A1D',  # green
            np.False_: '#BDBDBD'   # grey
        },
        alpha=0.5,
        edgecolor=None,
        aspect=1.2,
        s=30,
        legend=False,
    )
    ax = g.ax

    # Add threshold lines
    if pval_threshold:
        ax.axhline(
            y=-np.log10(pval_threshold),
            color='#A2A2A2',    # grey
            linestyle='--',
            label=pval_threshold)

    if score_threshold:
        ax.axvline(
            x=score_threshold,
            color='#A2A2A2',    # grey
            linestyle='--',
            label=score_threshold)

    ax.set_xlabel('Proteoform Score')
    ax.set_ylabel(ylabel)
    g.tight_layout()

    if ax:
        return ax
    else:
        plt.show()
        return
