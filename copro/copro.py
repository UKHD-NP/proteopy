from sklearn.cluster import AgglomerativeClustering

def cluster_peptides(df,
                     column_map: dict = {'peptide_a': 'pepA',
                                         'peptide_b': 'pepB',
                                         'correlation_values': 'PCC',
                                         'protein_id': 'protein_id'},
                     method: str = 'agglomerative-hierarchical-clustering') -> dict:
    '''
    Perform peptide clustering grouped by protein annotation.


    Parameters:
    ----------
    df : pandas.DataFrame
        Data frame with pairwise correlations annotated with the protein they belong to.]
    column_map: dict

    method : str
        Which clustering method to apply.

    Returns:
    -------
    dict
        Dictionary with clustering method output.
        - 'agglomerative-hierarchical-clustering' => {protein_id: {'labels': list, 'height': list, 'merge': list(list)}}
    '''

    assert all(df.index == df.columns)

    model = AgglomerativeClustering(n_clusters=None,
                                    metric='precomputed',
                                    linkage='average',
                                    distance_threshold=0,
                                    compute_distances=True)

    model.fit(df)

    # pylint: disable=no-member
    dendogram = {
        'labels': model.feature_names_in_.tolist(),
        'height': model.distances_.tolist(),
        'merge': model.children_.tolist()
    }
    # pylint: enable=no-member

    return dendogram
