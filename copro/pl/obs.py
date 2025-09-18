import warnings

import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype, is_categorical_dtype
from .utils import _resolve_color_scheme

def n_samples_by_category(
    adata,
    category_cols,
    ignore_na=False,
    sort_by_counts=True,
    x_label_rotation=45,
    show=True,
    save=False,
    ax=False,
    color_scheme=None,
    ):
    if isinstance(category_cols, str):
        category_cols = [category_cols]

    first_cat_col = category_cols[0]

    obs = adata.obs[category_cols].copy()

    for col in category_cols:
        if not (is_string_dtype(obs[col]) or is_categorical_dtype(obs[col])):
            obs[col] = obs[col].astype(str)

    cats_order = (
        obs[first_cat_col].cat.categories.tolist()
        if is_categorical_dtype(obs[first_cat_col])
        else obs[first_cat_col].unique().tolist()
        )

    for n_col, col in enumerate(category_cols):
        if not ignore_na and obs[col].isna().any():
            obs[col] = obs[col].cat.add_categories(['missing'])
            obs[col] = obs[col].fillna('missing')


    # Plot
    if len(category_cols) == 1:
        freq = obs[first_cat_col].value_counts()

        if not sort_by_counts:
            freq = freq[cats_order]

        colors = _resolve_color_scheme(color_scheme, freq.index)
        _ax = freq.plot(kind='bar', color=colors)

    elif len(category_cols) == 2:
        df = obs.groupby(category_cols, observed=False).size().unstack(fill_value=0)

        if sort_by_counts:
            new_order = df.sum(axis=1).sort_values(ascending=False).index.tolist()
            df = df.loc[new_order]

        colors = _resolve_color_scheme(color_scheme, df.columns)
        _ax = df.plot(kind='bar', stacked=True, color=colors)
        _ax.legend(loc='center right', bbox_to_anchor=(2,0.5))
        
    else:
        print('nr of categories > 2 not implemented yet.')

    _ax.set_xlabel(first_cat_col)
    _ax.set_ylabel('#')

    ha = (
        'right' if x_label_rotation > 0
        else 'left' if x_label_rotation < 0
        else 'center'
        )
    plt.setp(_ax.get_xticklabels(), rotation=x_label_rotation, ha=ha)

    if show:
        plt.show()
    if save:
        _ax.figure.savefig(save, dpi=300, bbox_inches='tight')
    if ax:
        return _ax
    if not show and not save and not ax:
        warnings.warn((
            'Function does not do anything. Set at least one argument to True:'
            ' show, save, ax'
            ))
