"""Small, self-contained simulations of peptide intensities."""

from collections.abc import Sequence
from numbers import Integral, Real

import anndata as ad
import numpy as np
import pandas as pd

from proteopy.pp.quantification import quantify_proteins
from proteopy.utils.anndata import check_proteodata


_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _integer(name, value, minimum=1):
    """Validate an integer without accepting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return int(value)


def _number(name, value):
    """Validate a finite real number without accepting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _vector(name, value):
    """Accept common one-dimensional sequence containers."""
    if isinstance(value, (str, bytes)) or not isinstance(
        value, (Sequence, np.ndarray, pd.Series, pd.Index)
    ):
        raise TypeError(f"{name} must be a one-dimensional sequence.")
    if isinstance(value, np.ndarray) and value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return list(value)


def _strings(name, value, size=None, unique=False):
    """Validate labels without silently converting missing values."""
    values = _vector(name, value)
    if not all(isinstance(v, str) for v in values):
        raise TypeError(f"{name} must contain only strings.")
    if not values or any(not v.strip() for v in values):
        raise ValueError(f"{name} must contain nonempty strings.")
    if size is not None and len(values) != size:
        raise ValueError(f"{name} must contain exactly {size} values.")
    if unique and len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique values.")
    return values


def _normal(name, value):
    """Normalize a Gaussian mean/standard-deviation pair."""
    values = _vector(name, value)
    if len(values) != 2:
        raise ValueError(f"{name} must contain [mean, std].")
    mean, std = (_number(name, v) for v in values)
    if std < 0:
        raise ValueError(f"{name} standard deviation must be >= 0.")
    return np.array([mean, std])


def _option(name, value, choices):
    """Validate a named option."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}.")


def _peptide_counts(value, n_proteins):
    """Resolve constant or protein-aligned peptide counts."""
    if isinstance(value, Integral):
        return np.full(
            n_proteins, _integer("n_peps_per_protein", value), dtype=int
        )
    values = _vector("n_peps_per_protein", value)
    if len(values) != n_proteins:
        raise ValueError("n_peps_per_protein must have one value per protein.")
    return np.array(
        [_integer("n_peps_per_protein", v) for v in values], dtype=int
    )


def _sample_groups(n_samples, groups, sample_ids):
    """Resolve sample count and labels without guessing explicit groups."""
    if n_samples is not None:
        n_samples = _integer("n_samples", n_samples)
    if groups is not None:
        groups = _strings("groups", groups)
    if sample_ids is not None:
        sample_ids = _strings("sample_ids", sample_ids, unique=True)
    if n_samples is None:
        lengths = [len(v) for v in (groups, sample_ids) if v is not None]
        n_samples = lengths[0] if lengths else 10
    for name, values in (("groups", groups), ("sample_ids", sample_ids)):
        if values is not None and len(values) != n_samples:
            raise ValueError(
                f"{name} must contain exactly {n_samples} values "
                "to match the sample count."
            )
    if groups is None:
        n_control = (n_samples + 1) // 2
        groups = ["control"] * n_control + ["treatment"] * (
            n_samples - n_control
        )
    return n_samples, groups, sample_ids


def _effect_input(value, n_contrasts, n_targets):
    """Validate explicit effects and resolve their broadcasting."""
    if value is None:
        return None
    if n_contrasts == 0:
        raise ValueError("A single group requires effect_size=None.")
    values = np.asarray(value, dtype=object)
    shape = (n_contrasts, n_targets)
    if values.ndim > 2:
        raise ValueError("effect_size must be a scalar, vector, or matrix.")
    if values.ndim == 1 and values.shape != (n_targets,):
        raise ValueError(
            f"effect_size must have {n_targets} target-aligned values."
        )
    if values.ndim == 2 and values.shape != shape:
        raise ValueError(f"effect_size matrix must have shape {shape}.")
    numeric = np.array(
        [_number("effect_size", v) for v in values.flat]
    ).reshape(values.shape)
    return np.broadcast_to(numeric, shape).copy()


def _identifiers(counts, groups, supplied):
    """Validate custom identifiers and supply deterministic defaults."""
    n_proteins = len(counts)
    n_peptides = int(counts.sum())
    proteins = supplied["protein_ids"]
    if proteins is None:
        proteins = [f"MAP{i + 1:03d}" for i in range(n_proteins)]
    proteins = _strings("protein_ids", proteins, n_proteins, unique=True)
    if any(any(c in p for c in " ,;") for p in proteins):
        raise ValueError("protein_ids must not contain spaces, ',' or ';'.")

    genes = supplied["gene_ids"]
    if genes is None:
        genes = [f"{p}{chr(65 + i % 26)}_gene" for i, p in enumerate(proteins)]
    genes = _strings("gene_ids", genes, n_proteins)
    samples = supplied["sample_ids"]
    if samples is None:
        samples = [f"sample{i + 1:02d}" for i in range(len(groups))]

    peptides = supplied["peptide_ids"]
    if peptides is not None:
        peptides = _strings("peptide_ids", peptides, n_peptides, unique=True)
        if any(set(p) - set(_AMINO_ACIDS) for p in peptides):
            raise ValueError(
                "peptide_ids must be unmodified uppercase sequences "
                "using the 20 canonical amino acids."
            )
    return proteins, genes, samples, peptides


def _sequences(rng, size):
    """Draw unique synthetic sequences from the canonical amino acids."""
    alphabet = list(_AMINO_ACIDS)
    peptides, seen = [], set()
    while len(peptides) < size:
        length = int(rng.integers(8, 21))
        peptide = "".join(rng.choice(alphabet, size=length))
        if peptide not in seen:
            seen.add(peptide)
            peptides.append(peptide)
    return peptides


def _categorical(values):
    """Keep the explicit simulation order available to plotting code."""
    return pd.Categorical(
        values, categories=list(dict.fromkeys(values)), ordered=True
    )


def _annotations(groups, ids, protein_index):
    """Build annotation columns and matching string indices."""
    proteins, genes, samples, peptides = ids
    obs = pd.DataFrame(
        {
            "sample_id": _categorical(samples),
            "group": _categorical(groups),
        },
        index=pd.Index(samples),
    )
    var = pd.DataFrame(
        {
            "protein_id": _categorical(np.array(proteins)[protein_index]),
            "peptide_id": _categorical(peptides),
            "gene_id": _categorical(np.array(genes)[protein_index]),
        },
        index=pd.Index(peptides),
    )
    return obs, var


def _raw_intensities(log2_values):
    """Exponentiate without silently introducing infinities or zeros."""
    try:
        if not np.isfinite(log2_values).all():
            raise FloatingPointError("Nonfinite simulated log2 intensity.")
        with np.errstate(over="raise", under="raise", invalid="raise"):
            return np.exp2(log2_values)
    except FloatingPointError as error:
        raise ValueError(
            "The simulation cannot represent these raw intensities as "
            "finite positive float64 values; reduce the log2 parameter "
            "magnitudes or standard deviations."
        ) from error


def _protein_output(adata, proteins, genes):
    """Summarize observed peptides and restore configured ID order."""
    out = quantify_proteins(adata, func="median", inplace=False)
    positions = out.var["protein_id"].tolist()
    order = pd.Index(positions).get_indexer(proteins)
    out = out[:, order].copy()
    out.var = pd.DataFrame(
        {
            "protein_id": _categorical(proteins),
            "gene_id": _categorical(genes),
        },
        index=pd.Index(proteins),
    )
    check_proteodata(out)
    return out


def toy_peptides(
    *,
    n_samples: int | None = None,
    n_proteins: int = 20,
    n_peps_per_protein: int | Sequence[int] = 5,
    protein_ids: Sequence[str] | None = None,
    gene_ids: Sequence[str] | None = None,
    peptide_ids: Sequence[str] | None = None,
    sample_ids: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    level: str = "peptide",
    effect_size: float | Sequence | np.ndarray | None = None,
    effect_level: str = "protein",
    differential_fraction: float = 0.2,
    differential_magnitude: float = 1.5,
    protein_abundance: Sequence[float] = (20.0, 2.0),
    peptide_response: Sequence[float] = (0.0, 1.0),
    biological_variation: Sequence[float] = (0.0, 0.3),
    loading_factor: Sequence[float] = (0.0, 0.3),
    measurement_noise: Sequence[float] = (0.0, 0.2),
    missing_rate: float = 0.05,
    log2_detection_limit: float | None = 15.0,
    random_state: int | None = 0,
) -> ad.AnnData:
    r"""Simulate a small, reproducible proteomics dataset offline.

    Generate peptide intensities with shared protein abundance,
    biological variation, peptide response, sample loading, and
    measurement noise. The model is additive in log2 space; ``.X``
    contains raw intensities and ``np.nan`` for missing observations.

    Parameters
    ----------
    n_samples : int or None
        Number of samples. ``None`` infers the count from ``groups``,
        otherwise from ``sample_ids``, and uses 10 if neither is given.
        If supplied, both vectors must match this count. With no
        ``groups``, samples are split as evenly as possible into
        control and treatment, with an odd extra sample in control.
    n_proteins : int
        Number of proteins, in generated or supplied protein ID order.
    n_peps_per_protein : int or sequence of int
        Constant peptide count, or one positive count per protein.
        Peptides occur in protein blocks, in the supplied protein order.
    protein_ids : sequence of str or None
        Unique IDs aligned to proteins. ``None`` generates ``MAP001``,
        ``MAP002``, etc. Spaces, commas, and semicolons are not allowed.
    gene_ids : sequence of str or None
        Gene IDs aligned to proteins; multiple proteins may share a
        gene. ``None`` appends a cycling A-Z letter and ``_gene`` to
        each protein ID, e.g. ``MAP001A_gene``.
    peptide_ids : sequence of str or None
        Globally unique, uppercase, unmodified canonical amino-acid
        sequences aligned to peptide columns. ``None`` generates
        synthetic 8-20-residue sequences by sampling uniformly from
        the 20 canonical amino acids, without cleavage constraints.
    sample_ids : sequence of str or None
        Unique IDs aligned to samples. ``None`` generates ``sample01``,
        ``sample02``, etc. The length must match the number of samples
        resolved by ``n_samples`` and ``groups``.
    groups : sequence of str or None
        One group label per sample, retaining sample order. ``None``
        creates balanced control and treatment groups, with control
        first and an odd extra sample assigned to control. The
        zero-argument dataset has five samples in each group. Explicit
        labels allow arbitrary group sizes. The first encountered
        group is the reference; subsequent groups follow encounter
        order. A single group has no differential effects.
    level : {'peptide', 'protein'}
        Output level. Protein output is the median of observed raw
        peptide intensities per protein and sample, after missingness.
        Entirely missing proteins remain missing.
    effect_size : float, sequence, array, or None
        Signed log2 effects relative to the reference group. A scalar
        affects every target and contrast. A vector has one value per
        target and is reused across non-reference groups. A matrix has
        shape ``(n_groups - 1, n_targets)``, with rows in non-reference
        group order. ``effect_level`` determines target alignment.
        ``None`` samples effects using ``differential_fraction`` and
        ``differential_magnitude``. Single-group input requires
        ``None``.
    effect_level : {'protein', 'peptide'}
        Whether effect targets are proteins or individual peptides.
        Protein effects are shared by all peptides of each protein.
    differential_fraction : float
        Probability of a nonzero effect when ``effect_size=None``.
        Positive and negative effects have equal probability; draws
        are independent across targets and contrasts. Explicit effects
        override this setting.
    differential_magnitude : float
        Nonnegative absolute log2 effect when sampling differential
        targets. Explicit effects override this setting.
    protein_abundance : sequence of float
        ``[mean, std]`` for baseline protein log2 abundance, sampled
        once per protein and shared across its peptides and samples.
    peptide_response : sequence of float
        ``[mean, std]`` for a peptide's fixed log2 response, sampled
        once per peptide and shared across samples.
    biological_variation : sequence of float
        ``[mean, std]`` for log2 biological deviations, sampled per
        protein-sample pair and shared by that protein's peptides.
    loading_factor : sequence of float
        ``[mean, std]`` for additive log2 loading differences, sampled
        once per sample and shared by all peptides in that sample.
    measurement_noise : sequence of float
        ``[mean, std]`` for independent log2 measurement noise per
        peptide-sample observation. All normal parameters accept lists,
        tuples, or arrays; the second value is a standard deviation,
        not a variance. Zero standard deviation disables random scatter.
    missing_rate : float
        Independent probability of deleting each observation (MCAR),
        between zero and one. This is not the final missing fraction
        when detection-limit censoring is also enabled.
    log2_detection_limit : float or None
        Censor observations strictly below this log2 intensity (MNAR).
        Values equal to the limit are retained unless randomly deleted.
        ``None`` disables censoring. Both mechanisms produce ``np.nan``.
    random_state : int or None
        Nonnegative seed for local NumPy generators. ``None`` requests
        fresh randomness; the realized seed entropy is stored for
        replay. Global NumPy random state is not changed.

    Returns
    -------
    AnnData
        Dense float64 raw intensities, samples by peptides or proteins.
        ``.obs`` contains ``sample_id`` and ``group``. Peptide ``.var``
        contains ``protein_id``, ``peptide_id``, and ``gene_id``;
        protein output omits ``peptide_id``. IDs and groups are ordered
        categoricals in configured order. Missing annotation labels
        are rejected. ``.uns['toy_peptides']`` retains compact truth.

    Raises
    ------
    TypeError
        If argument types are invalid, including boolean numbers.
    ValueError
        If ranges, dimensions, identifiers, or raw intensities are
        invalid. Fully missing samples and variables are retained.

    Notes
    -----
    For protein :math:`p`, peptide :math:`j`, and sample :math:`s`:

    .. math::

        Y_{pjs} = \mu_p + \beta_{t,g(s)} + u_{ps}
                  + \alpha_{pj} + l_s + \epsilon_{pjs},
        \qquad I_{pjs} = 2^{Y_{pjs}}.

    Here :math:`t=p` for protein effects and :math:`t=(p,j)` for
    peptide effects; :math:`\beta_{t,g_0}=0` for the reference group.
    The five distribution parameters respectively control
    :math:`\mu_p`, :math:`\alpha_{pj}`, :math:`u_{ps}`, :math:`l_s`,
    and :math:`\epsilon_{pjs}`. Their underlying draws are independent;
    observations become dependent through shared terms. No draws are
    centered or clipped. Fixed responses preserve expected peptide
    ranks, but measurement noise can swap nearby peptides.

    With differential fraction :math:`f` and magnitude :math:`d`,
    automatic effects are :math:`0,+d,-d` with probabilities
    :math:`1-f,f/2,f/2`. The defaults produce 20 percent differential
    targets on average, not an exact proportion in each small dataset.

    Constant log2 variance :math:`v` gives lognormal raw intensities:

    .. math::

        \mathrm{Var}(I) = \mathrm{E}(I)^2
        \left[\exp\left((\ln 2)^2v\right)-1\right].

    Thus raw variance increases with abundance. Evaluate this within
    groups before censoring, which can alter the observed relationship.
    The baseline log2 median is near 20. The settings are illustrative,
    rather than an instrument calibration or a fixed intensity range.
    Sequence composition is independent of intensity. Peptide effects
    can create proteoform-like groups, but there are no independent
    proteoform abundances or contributions from shared peptides.
    Multi-group profiles do not impose fractionation kinetics.

    Output is linear-scale, like the existing example dataset loaders.
    Some analyses require this scale, including CV calculation, while
    others require log-transformed data. A log2 layer can retain both
    representations; see the example below.

    To create proteoform-like groups, set ``effect_level="peptide"``
    and give subsets of the same protein's peptides distinct effects.
    Repeating an effect within each subset gives those peptides a
    shared condition response. The example below assigns opposite
    responses to two three-peptide groups. Group labels can be added
    as ground truth; they are not inferred or added automatically.

    Each call initializes the same independent child random streams
    from a given ``random_state``. Normal draws repeat when their
    stream, arguments, shapes, and call sequence are unchanged.
    Successive draws from one generator advance its state; different
    components use different streams rather than identical draws.
    Sequence generation and deletion have their own streams, so
    changing either does not shift the model's normal draws.

    Compact truth contains ``parameters``, ``seed_entropy`` (a string),
    ``numpy_version``, ``reference_group``, ``contrast_groups``,
    ``effect_target_ids``, and ``effects`` (contrasts by targets).
    ``proteins``, ``peptides``, and ``samples`` are DataFrames holding
    ``protein_abundance``, ``peptide_response``, and ``loading_factor``
    in log2 units, alongside ID mappings and sample groups. Peptide
    truth is retained for protein output. Full biological deviations,
    noise, uncensored intensities, and missingness masks are omitted.
    Replay an unseeded run with ``random_state=int(seed_entropy)`` and
    the same arguments. Separate random streams keep numeric draws
    unchanged when custom IDs or missingness settings change. Exact
    replay assumes the same NumPy version and simulation implementation.

    See Also
    --------
    proteopy.pp.quantify_proteins : Aggregate peptide intensities.
    proteopy.utils.proteoform_group_simulator : Plant proteoform groups.
    numpy.random.SeedSequence : Reproducible child random streams.

    Examples
    --------
    >>> import proteopy as pr
    >>> adata = pr.datasets.toy_peptides()
    >>> adata.shape
    (10, 100)
    >>> proteins = pr.datasets.toy_peptides(level="protein")
    >>> proteins.shape
    (10, 20)

    Keep a log2 layer for methods that support selecting a layer:

    >>> import numpy as np
    >>> adata.layers["log2"] = np.log2(adata.X)

    Set the sample count explicitly or infer it from supplied IDs:

    >>> adata = pr.datasets.toy_peptides(n_samples=12)
    >>> adata.obs["group"].value_counts().tolist()
    [6, 6]
    >>> adata = pr.datasets.toy_peptides(
    ...     sample_ids=["A", "B", "C", "D"],
    ... )
    >>> adata.obs["group"].tolist()
    ['control', 'control', 'treatment', 'treatment']

    Create two proteoform-like groups with opposing condition effects:

    >>> adata = pr.datasets.toy_peptides(
    ...     n_proteins=1,
    ...     n_peps_per_protein=6,
    ...     effect_level="peptide",
    ...     effect_size=[1.5, 1.5, 1.5, -1.5, -1.5, -1.5],
    ...     missing_rate=0,
    ...     log2_detection_limit=None,
    ...     random_state=42,
    ... )
    >>> adata.var["true_proteoform_id"] = (
    ...     ["MAP001_pf1"] * 3 + ["MAP001_pf2"] * 3
    ... )

    Assign distinct profiles to three groups and two proteins:

    >>> fractions = pr.datasets.toy_peptides(
    ...     n_proteins=2,
    ...     n_peps_per_protein=[3, 5],
    ...     groups=["F1"] * 3 + ["F2"] * 3 + ["F3"] * 3,
    ...     effect_size=[[2, -1], [-1, 2]],
    ... )
    """
    # -- Validate the complete simulation contract before drawing
    n_proteins = _integer("n_proteins", n_proteins)
    counts = _peptide_counts(n_peps_per_protein, n_proteins)
    _option("level", level, ("peptide", "protein"))
    _option("effect_level", effect_level, ("protein", "peptide"))
    if random_state is not None:
        random_state = _integer("random_state", random_state, minimum=0)
    n_samples, groups, sample_ids = _sample_groups(
        n_samples, groups, sample_ids
    )
    group_order = list(dict.fromkeys(groups))
    n_targets = n_proteins if effect_level == "protein" else int(counts.sum())
    effects = _effect_input(effect_size, len(group_order) - 1, n_targets)
    differential_fraction = _number(
        "differential_fraction", differential_fraction
    )
    differential_magnitude = _number(
        "differential_magnitude", differential_magnitude
    )
    missing_rate = _number("missing_rate", missing_rate)
    if not 0 <= differential_fraction <= 1:
        raise ValueError("differential_fraction must be between 0 and 1.")
    if differential_magnitude < 0:
        raise ValueError("differential_magnitude must be >= 0.")
    if not 0 <= missing_rate <= 1:
        raise ValueError("missing_rate must be between 0 and 1.")
    if log2_detection_limit is not None:
        log2_detection_limit = _number(
            "log2_detection_limit", log2_detection_limit
        )
    normals = {
        name: _normal(name, value)
        for name, value in {
            "protein_abundance": protein_abundance,
            "peptide_response": peptide_response,
            "biological_variation": biological_variation,
            "loading_factor": loading_factor,
            "measurement_noise": measurement_noise,
        }.items()
    }
    proteins, genes, samples, peptides = _identifiers(
        counts,
        groups,
        {
            "protein_ids": protein_ids,
            "gene_ids": gene_ids,
            "sample_ids": sample_ids,
            "peptide_ids": peptide_ids,
        },
    )

    # -- Independent streams for effects, model terms, IDs, and deletion
    seed = np.random.SeedSequence(random_state)
    (
        effect_rng,
        abundance_rng,
        response_rng,
        biological_rng,
        loading_rng,
        measurement_rng,
        sequence_rng,
        missing_rng,
    ) = (np.random.default_rng(s) for s in seed.spawn(8))
    if effects is None:
        effects = effect_rng.choice(
            [0.0, differential_magnitude, -differential_magnitude],
            size=(len(group_order) - 1, n_targets),
            p=[
                1 - differential_fraction,
                differential_fraction / 2,
                differential_fraction / 2,
            ],
        )
    protein_index = np.repeat(np.arange(n_proteins), counts)
    n_peptides = len(protein_index)
    mu = abundance_rng.normal(*normals["protein_abundance"], size=n_proteins)
    alpha = response_rng.normal(*normals["peptide_response"], size=n_peptides)
    biological = biological_rng.normal(
        *normals["biological_variation"], size=(n_samples, n_proteins)
    )
    loading = loading_rng.normal(*normals["loading_factor"], size=n_samples)
    noise = measurement_rng.normal(
        *normals["measurement_noise"], size=(n_samples, n_peptides)
    )
    target_index = (
        protein_index if effect_level == "protein" else (np.arange(n_peptides))
    )
    group_index = pd.Categorical(groups, categories=group_order).codes
    all_effects = np.vstack([np.zeros((1, n_targets)), effects])
    with np.errstate(over="ignore", invalid="ignore"):
        log2_values = (
            mu[protein_index][None, :]
            + all_effects[group_index][:, target_index]
            + biological[:, protein_index]
            + alpha[None, :]
            + loading[:, None]
            + noise
        )
    values = _raw_intensities(log2_values)
    missing = missing_rng.random(values.shape) < missing_rate
    if log2_detection_limit is not None:
        missing |= log2_values < log2_detection_limit
    values[missing] = np.nan
    if peptides is None:
        peptides = _sequences(sequence_rng, n_peptides)

    # -- Assemble observations, variables, and compact simulation truth
    obs, var = _annotations(
        groups, (proteins, genes, samples, peptides), protein_index
    )
    adata = ad.AnnData(X=values, obs=obs, var=var)
    adata.uns["toy_peptides"] = {
        "parameters": {
            "n_samples": n_samples,
            "n_proteins": n_proteins,
            "n_peps_per_protein": counts.copy(),
            "level": level,
            "effect_level": effect_level,
            "effect_mode": "automatic" if effect_size is None else "explicit",
            "differential_fraction": differential_fraction,
            "differential_magnitude": differential_magnitude,
            "missing_rate": missing_rate,
            "log2_detection_limit": log2_detection_limit,
            **normals,
        },
        "seed_entropy": str(seed.entropy),
        "numpy_version": np.__version__,
        "reference_group": group_order[0],
        "contrast_groups": np.array(group_order[1:], dtype=str),
        "effect_target_ids": np.array(
            proteins if effect_level == "protein" else peptides, dtype=str
        ),
        "effects": effects,
        "proteins": pd.DataFrame(
            {
                "protein_id": proteins,
                "gene_id": genes,
                "protein_abundance": mu,
            },
            index=pd.Index(proteins),
        ),
        "peptides": pd.DataFrame(
            {
                "peptide_id": peptides,
                "protein_id": np.array(proteins)[protein_index],
                "gene_id": np.array(genes)[protein_index],
                "peptide_response": alpha,
            },
            index=pd.Index(peptides),
        ),
        "samples": pd.DataFrame(
            {
                "sample_id": samples,
                "group": groups,
                "loading_factor": loading,
            },
            index=pd.Index(samples),
        ),
    }
    check_proteodata(adata)
    if level == "protein":
        return _protein_output(adata, proteins, genes)
    return adata
