import warnings
import pandas as pd

def obs_merge(
    adata,
    df,
    obs_on,
    df_on,
    suffix = '_annotated',
    sort_obs_by_ann = False,
    ):
    '''Annotate AnnData.obs with df on a specific column.

    Args:
        adata (AnnData)
        df (pd.DataFrame)
        obs_on (pd.DataFrame)
        df_on (pd.DataFrame)

    Returns:
        AnnData with extended .obs
    '''
    adata = adata.copy()
    obs = adata.obs.copy().reset_index()

    # Check input
    if not obs_on in obs.columns:
        raise ValueError()

    if not df_on in df.columns:
        raise ValueError()

    df[df_on] = df[df_on].astype(str)
    df.drop_duplicates(keep='first', inplace=True)

    if len(df[df_on].unique()) != len(df):
        df.drop_duplicates(subset=df_on, keep='first', inplace=True)
        warnings.warn(f'Rows with repeated values in df_on were collapsed to the first occurance.')

    diff_idx1 = set(df[df_on]).difference(set(obs[obs_on]))
    if diff_idx1:
        warnings.warn(
            f'There are {len(diff_idx1)} unique values in df_on '
            f'which are not found in obs_on. They were ignored.'
            )

    diff_idx2 = set(obs[obs_on]).difference(set(df[df_on]))
    if diff_idx2:
        warnings.warn(
            f'There are {len(diff_idx2)} values in obs_on '
            f'which are not found in df_on. They were filled with nan.'
            )

    new_obs = pd.merge(
        obs,
        df,
        left_on=obs_on,
        right_on=df_on,
        how='left',
        suffixes=('', suffix),
        )

    new_obs = new_obs.set_index('index')
    new_obs.index.name = None

    if obs_on != df_on:
        new_obs = new_obs.drop(columns=[df_on])

    if obs_on != 'index':
        new_obs[obs_on] = new_obs[obs_on].astype('category')

    adata.obs = new_obs

    if sort_obs_by_ann:
        idx = [i for i in df[df_on] if i in obs[obs_on].values]
        idx.extend(list(diff_idx2))
        adata = adata[idx,]

    return adata
