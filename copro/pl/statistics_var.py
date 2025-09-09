from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns

def n_elements_per_category(
    adata,
    category_col,
    elements_col = None,
    bin_width = None,
    bin_range = None,
    ):
    var = adata.var.copy()
    cats = [category_col]

    if elements_col:
        cats.append(elements_col)
    else:
        elements_col = 'index'
        var.reset_index()
        cats.append('index')

    var = var.drop_duplicates(var, keep='first')
    counts = var.groupby(category_col, observed=False).size()

    sns.histplot(
        counts,
        binwidth=bin_width if bin_width else None,
        binrange=bin_range if bin_range else None,
        )

    plt.show()

n_peptides_per_gene = partial(
    n_elements_per_category,
    category_col = 'protein_id',
    bin_width = 5,
    )

n_proteoforms_per_gene = partial(
    n_elements_per_category,
    elements_col = 'proteoform_id',
    category_col = 'protein_id',
    )
