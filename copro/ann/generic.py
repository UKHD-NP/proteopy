import warnings
import pandas as pd

def obs_merge(
    adata,
    df,
    obs_on,
    df_on,
    suffix = '_annotated',
    ):
    '''Annotate AnnData.var with df on a specific column.

    Args:
        adata (AnnData)
        df (pd.DataFrame)
        obs_on (pd.DataFrame)
        df_on (pd.DataFrame)

    Returns:
        AnnData with extended .obs
    '''
    adata = adata.copy()
    obs = adata.obs.copy()
    obs = obs.reset_index()

    df[df_on] = df[df_on].astype(str)

    if len(df[df_on].unique()) != len(df):
        df = df.drop_duplicates(subset=df_on, keep='first')
        warnings.warn(f'Rows with repeated obs[obs_on] values were collapsed to the first occurance.')

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

    return adata
