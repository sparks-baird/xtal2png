from importlib.resources import read_text
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120, beta=90, gamma=60)
dummy_structures = [
    Structure(lattice, ["Si", "Si"], coords),
    Structure(lattice, ["Ni", "Ni"], coords),
]

EXAMPLE_CIFS = ["Zn2B2PbO6.cif", "V2NiSe4.cif"]
example_structures = []
for cif in EXAMPLE_CIFS:
    cif_str = read_text("xtal2png.utils", cif)
    example_structures.append(Structure.from_str(cif_str, "cif"))


def element_wise_scaler(
    X: ArrayLike,
    feature_range: Optional[Sequence] = None,
    data_range: Optional[Sequence] = None,
):
    """Scale parameters according to a prespecified min and max (``data_range``).

    ``feature_range`` is preserved from MinMaxScaler

    See Also
    --------
    sklearn.preprocessing.MinMaxScaler : Scale each feature to a given range.

    Parameters
    ----------
    X : ArrayLike
        Features to be scaled element-wise.
    feature_range : Sequence
        The scaled values will span the range of ``feature_range``
    data_range : Sequence
        Expected bounds for the data, e.g. 0 to 117 for periodic elements

    Returns
    -------
    X_scaled
        Element-wise scaled values.

    Examples
    --------
    >>> element_wise_scaler([[1, 2], [3, 4]], feature_range=[1, 4], data_range=[0, 8])
    array([[1.375, 1.75 ],
        [2.125, 2.5  ]])
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if data_range is None:
        data_range = [np.min(X), np.max(X)]
    if feature_range is None:
        feature_range = [np.min(X), np.max(X)]

    data_min, data_max = data_range
    feature_min, feature_max = feature_range
    # following modified from:
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    X_std = (X - data_min) / (data_max - data_min)
    X_scaled = X_std * (feature_max - feature_min) + feature_min
    return X_scaled


def element_wise_unscaler(
    X_scaled: ArrayLike,
    feature_range: Sequence,
    data_range: Sequence,
):
    """Scale parameters according to a prespecified min and max (``data_range``).

    ``feature_range`` is preserved from MinMaxScaler

    See Also
    --------
    sklearn.preprocessing.MinMaxScaler : Scale each feature to a given range.

    Parameters
    ----------
    X : ArrayLike
        Element-wise scaled values.
    feature_range : Sequence
        The scaled values will span the range of ``feature_range``
    data_range : Sequence
        Expected bounds for the data, e.g. 0 to 117 for periodic elements

    Returns
    -------
    X
        Element-wise unscaled values.

    Examples
    --------
    >>> element_wise_unscaler(
    ...     [[1.375, 1.75], [2.125, 2.5]], feature_range=[1, 4], data_range=[0, 8]
    ... )
    array([[1., 2.],
       [3., 4.]])

    """
    if not isinstance(X_scaled, np.ndarray):
        X_scaled = np.array(X_scaled)

    data_min, data_max = data_range
    feature_min, feature_max = feature_range
    # following modified from:
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

    # inverse transform, checked against Mathematica
    X_std = (X_scaled - feature_min) / (feature_max - feature_min)
    X = data_min + (data_max - data_min) * X_std
    return X


def rgb_scaler(
    X: ArrayLike,
    data_range: Optional[Sequence] = None,
):
    """Scale parameters according to RGB scale (0 to 255).

    ``feature_range`` is fixed to [0, 255], ``data_range`` is either specified

    See Also
    --------
    sklearn.preprocessing.MinMaxScaler : Scale each feature to a given range.

    Parameters
    ----------
    X : ArrayLike
        Features to be scaled element-wise.
    data_range : Optional[Sequence]
        Range to use in place of np.min(X) and np.max(X) as in ``MinMaxScaler``.

    Returns
    -------
    X_scaled
        Element-wise scaled values.

    Examples
    --------
    >>> rgb_scaler([[1, 2], [3, 4]], data_range=[0, 8])
    array([[ 32,  64],
        [ 96, 128]], dtype=uint8)
    """
    rgb_range = [0, 255]
    X_scaled = element_wise_scaler(X, data_range=data_range, feature_range=rgb_range)
    X_scaled = np.round(X_scaled).astype(int)
    return X_scaled


def rgb_unscaler(
    X: ArrayLike,
    data_range: Sequence,
):
    """Unscale parameters from their RGB scale (0 to 255).

    ``feature_range`` is fixed to [0, 255], ``data_range`` is either specified or
    calculated based on min and max.

    See Also
    --------
    sklearn.preprocessing.MinMaxScaler : Scale each feature to a given range.

    Parameters
    ----------
    X : ArrayLike
        Element-wise scaled values.
    data_range : Optional[Sequence]
        Range to use in place of np.min(X) and np.max(X) as in ``class:MinMaxScaler``.

    Returns
    -------
    X
        Unscaled features.

    Examples
    --------
    >>> rgb_unscaler([[32, 64], [96, 128]], data_range=[0, 8])
    array([[1, 2],
          [3, 4]])
    """
    rgb_range = [0, 255]
    X_scaled = element_wise_unscaler(X, data_range=data_range, feature_range=rgb_range)
    return X_scaled
