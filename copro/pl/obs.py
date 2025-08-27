import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype, is_categorical_dtype

def n_samples_by_category(
    adata,
    category_cols,
    ignore_na=False,
    sort_by_counts=True,
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

        freq.plot(kind='bar')
        plt.show()

    elif len(category_cols) == 2:
        df = obs.groupby(category_cols, observed=False).size().unstack(fill_value=0)

        if sort_by_counts:
            new_order = df.sum(axis=1).sort_values(ascending=False).index.tolist()
            df = df.loc[new_order]

        df.plot(kind='bar', stacked=True)
        plt.legend(loc='center right', bbox_to_anchor=(2,0.5))
        
    else:
        print('nr of categories > 2 not implemented yet.')
