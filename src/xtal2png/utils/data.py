from importlib.resources import read_text
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

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
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html # noqa:E501
    X_std = (X - data_min) / (data_max - data_min)
    X_scaled = X_std * (feature_max - feature_min) + feature_min
    return X_scaled


def element_wise_scalers(
    X: ArrayLike,
    feature_ranges: Optional[Sequence] = None,
    data_ranges: Optional[Sequence] = None,
):
    """Scale parameters according to prespecified sets of min and max (``data_ranges``).

    ``feature_ranges`` is comparable to ``feature_range`` from MinMaxScaler.

    See Also
    --------
    sklearn.preprocessing.MinMaxScaler : Scale each feature to a given range.

    Parameters
    ----------
    X : ArrayLike
        Features to be scaled element-wise.
    feature_ranges : Sequence
        The scaled values will span the range of ``feature_ranges`` on a per X element
        basis (i.e. element_wise_scaler(X[0], feature_ranges[0], data_ranges[0])))
    data_ranges : Sequence
        Expected bounds for the data, e.g. 0 to 117 for periodic elements. The scaled
        values will span the range of ``data_ranges`` on a per X element basis (i.e.
        element_wise_scaler(X[0], feature_ranges[0], data_ranges[0])))

    Returns
    -------
    X_scaled
        Element-wise scaled values.

    Examples
    --------
    >>> element_wise_scalers([[1, 2], [3, 4]], feature_ranges=[[1, 4],[2, 4]],
    ... data_ranges=[[0, 8],[2, 8]])
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    num_samples = X.shape[0]
    if data_ranges is None:
        data_ranges = [None] * num_samples

    if feature_ranges is None:
        feature_ranges = [None] * num_samples

    X_scaled = np.array(
        [
            element_wise_scaler(x, fr, dr)
            for x, dr, fr in zip(X, data_ranges, feature_ranges)
        ]
    )
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
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html # noqa:E501

    # inverse transform, checked against Mathematica
    X_std = (X_scaled - feature_min) / (feature_max - feature_min)
    X = data_min + (data_max - data_min) * X_std
    return X


def element_wise_unscalers(
    X_scaled: ArrayLike,
    feature_ranges: Sequence,
    data_ranges: Sequence,
):
    """Unscale parameters via prespecified sets of min and max (``data_ranges``).

    ``feature_ranges`` is comparable to ``feature_range`` from MinMaxScaler.

    See Also
    --------
    sklearn.preprocessing.MinMaxScaler : Scale each feature to a given range.

    Parameters
    ----------
    X : ArrayLike
        Features to be scaled element-wise.
    feature_ranges : Sequence
        The unscaled values will span the range of ``feature_ranges`` on a per X element
        basis (i.e. element_wise_scaler(X[0], feature_ranges[0], data_ranges[0])))
    data_ranges : Sequence
        Expected bounds for the data, e.g. 0 to 117 for periodic elements. The unscaled
        values will span the range of ``data_ranges`` on a per X element basis (i.e.
        element_wise_scaler(X[0], feature_ranges[0], data_ranges[0])))

    Returns
    -------
    X
        Element-wise unscaled values.

    Examples
    --------
    >>> element_wise_scalers([[1, 2], [3, 4]], feature_ranges=[[1, 4],[2, 4]],
    ... data_ranges=[[0, 8],[2, 8]])
    """
    if not isinstance(X_scaled, np.ndarray):
        X_scaled = np.array(X_scaled)

    X = np.array(
        [
            element_wise_unscaler(x, fr, dr)
            for x, dr, fr in zip(X_scaled, data_ranges, feature_ranges)
        ]
    )
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


def rgb_scalers(
    X: ArrayLike,
    data_ranges: Optional[Sequence] = None,
):
    """Scale parameters according to RGB scale (0 to 255).

    ``feature_ranges`` is fixed to [[0, 255]]*X.shape[0], ``data_range`` is either
    specified or fitted based on min/max.

    See Also
    --------
    sklearn.preprocessing.MinMaxScaler : Scale each feature to a given range.

    Parameters
    ----------
    X : ArrayLike
        Features to be scaled element-wise.
    data_ranges : Optional[Sequence]
        Ranges to use in place of np.min(X) and np.max(X) as in ``MinMaxScaler``.

    Returns
    -------
    X_scaled
        Element-wise scaled values.

    Examples
    --------
    >>> rgb_scaler([[1, 2], [3, 4]], data_ranges=[[0, 8],[1, 8]])
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    rgb_ranges = [0, 255] * X.shape[0]
    X_scaled = element_wise_scalers(
        X, data_ranges=data_ranges, feature_ranges=rgb_ranges
    )
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


def rgb_unscalers(
    X: ArrayLike,
    data_ranges: Sequence,
):
    """Unscale parameters from their RGB scale (0 to 255), (loop of rgb_unscaler).

    ``feature_range`` is fixed to [0, 255], ``data_range`` is either specified or
    calculated based on min and max.

    See Also
    --------
    sklearn.preprocessing.MinMaxScaler : Scale each feature to a given range.

    Parameters
    ----------
    X : ArrayLike
        Element-wise scaled values.
    data_ranges : Optional[Sequence]
        Ranges to use in place of np.min(X) and np.max(X) as in ``class:MinMaxScaler``.

    Returns
    -------
    X
        Unscaled features.

    Examples
    --------
    >>> rgb_unscaler([[32, 64], [32, 64]], data_ranges=[[0, 8],[2,16]])
    array([[1, 2],
          [3, 4]])
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    rgb_ranges = [0, 255] * X.shape[0]
    X_scaled = element_wise_unscalers(
        X, data_ranges=data_ranges, feature_ranges=rgb_ranges
    )
    return X_scaled


def get_image_mode(d: np.ndarray) -> str:
    """Get the image mode (i.e. "RGB" vs. grayscale ("L")) for an image array.

    Parameters
    ----------
    d : np.ndarray
        A NumPy array with 3 dimensions, where the first dimension corresponds to the
      of image channels and the second and third dimensions correspond to the height and
      width of the image.

    Returns
    -------
    mode : str
        "RGB" for 3-channel images and "L" for grayscale images.

    Raises
    ------
    ValueError
        "expected an array with 3 dimensions, received {d.ndim} dims"
    ValueError
        "Expected a single-channel or 3-channel array, but received a {d.ndim}-channel
        array."

    Examples
    --------
    >>> d = np.zeros((1, 64, 64), dtype=np.uint8) # grayscale image
    >>> mode = get_image_mode(d)
    "L"
    """
    if d.ndim != 3:
        raise ValueError("expected an array with 3 dimensions, received {d.ndim} dims")
    if d.shape[0] == 3:
        mode = "RGB"
    elif d.shape[0] == 1:
        mode = "L"
    else:
        raise ValueError(
            f"Expected a single-channel or 3-channel array, but received a {d.ndim}-channel array."  # noqa: E501
        )

    return mode


def unit_cell_converter(
    s: Structure, cell_type: Optional[str] = None, symprec=0.1, angle_tolerance=5.0
):
    """Convert from the original unit cell type to another unit cell via pymatgen.

    Parameters
    ----------
    s : Structure
        a pymatgen Structure.
    cell_type : Optional[str], optional
        The cell type as a str or None if leaving the structure as-is. Possible options
        are "primitive_standard", "conventional_standard", "refined", "reduced", and
        None. By default None

    Returns
    -------
    s : Structure
        The converted Structure.

    Raises
    ------
    ValueError
        "Expected one of 'primitive_standard', 'conventional_standard', 'refined',
        'reduced' or None, got {cell_type}"

    Examples
    --------
    >>> s = unit_cell_converter(s, cell_type="reduced")
    """
    spa = SpacegroupAnalyzer(
        s,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    if cell_type == "primitive_standard":
        s = spa.get_primitive_standard_structure()
    elif cell_type == "conventional_standard":
        s = spa.get_conventional_standard_structure()
    elif cell_type == "refined":
        s = spa.get_refined_structure()
    elif cell_type == "reduced":
        s = s.get_reduced_structure()
    elif cell_type is not None:
        raise ValueError(
            f"Expected one of 'primitive_standard', 'conventional_standard', 'refined', 'reduced' or None, got {cell_type}"  # noqa: E501
        )
    return s
