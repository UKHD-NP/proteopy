from collections.abc import Sequence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

def _resolve_color_scheme(color_scheme, labels):
    """Convert a user-supplied color scheme into a list matching `labels` order."""
    labels_list = list(labels)

    if color_scheme is None:
        if not labels_list:
            return None

        color_cycle = plt.rcParams.get('axes.prop_cycle')

        if color_cycle is None:
            return None

        default_colors = color_cycle.by_key().get('color', [])

        if not default_colors:
            return None

        return [default_colors[i % len(default_colors)] for i in range(len(labels_list))]

    if isinstance(color_scheme, np.ndarray):
        color_scheme = color_scheme.tolist()

    if isinstance(color_scheme, dict):
        colors = []
        missing = []
        for label in labels_list:
            color = color_scheme.get(label)
            if color is None:
                color = color_scheme.get(str(label))
            if color is None:
                missing.append(label)
            colors.append(color)
        if missing:
            raise ValueError(
                f'No color provided for categories: {", ".join(map(str, missing))}.'
            )
        return colors

    if isinstance(color_scheme, str):
        try:
            cmap = plt.get_cmap(color_scheme)
        except ValueError:
            return [color_scheme] * len(labels_list)
        if len(labels_list) == 1:
            return [cmap(0.5)]
        positions = np.linspace(0, 1, len(labels_list))
        return [cmap(pos) for pos in positions]

    if isinstance(color_scheme, Colormap):
        if len(labels_list) == 1:
            return [color_scheme(0.5)]
        positions = np.linspace(0, 1, len(labels_list))
        return [color_scheme(pos) for pos in positions]

    if isinstance(color_scheme, Sequence) and not isinstance(
        color_scheme, (bytes, bytearray)
    ):
        color_list = list(color_scheme)
        if not color_list:
            raise ValueError('color_scheme sequence cannot be empty.')
        if len(color_list) == 1:
            return color_list * len(labels_list)
        if len(color_list) < len(labels_list):
            raise ValueError(
                'color_scheme sequence must include at least as many colors as labels.'
            )
        return color_list[:len(labels_list)]

    if callable(color_scheme):
        positions = np.linspace(0, 1, len(labels_list))
        return [color_scheme(pos) for pos in positions]

    raise TypeError(
        'color_scheme must be a sequence of colors, a Matplotlib colormap, a '
        'callable returning colors, a dictionary mapping categories to colors, '
        'or a named Matplotlib palette.'
    )
