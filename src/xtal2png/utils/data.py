from importlib.resources import read_text
from typing import Optional, Sequence
from warnings import warn

import numpy as np
from numpy.testing import assert_allclose, assert_equal
from numpy.typing import ArrayLike
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
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


# ToDo: potentially expose tolerance options
def _get_space_group(s: Structure) -> int:
    """Get space group from structure.
    See issue https://github.com/sparks-baird/xtal2png/issues/184
    """
    try:
        return int(np.round(s.get_space_group_info()[1]))
    except TypeError:
        # 0 should be fine as it is not taken
        return 0


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


RGB_TOL = 1 / 255  # should this be 256?
RGB_LOOSE_TOL = 1.5 / 255


def assert_structures_approximate_match(
    example_structures, structures, tol_multiplier=1.0
):
    for i, (s, structure) in enumerate(zip(example_structures, structures)):
        dummy_matcher = StructureMatcher()
        ltol = dummy_matcher.ltol * tol_multiplier
        stol = dummy_matcher.stol * tol_multiplier
        angle_tol = dummy_matcher.angle_tol * tol_multiplier
        sm = StructureMatcher(
            ltol=ltol,
            stol=stol,
            angle_tol=angle_tol,
            comparator=ElementComparator(),
        )
        is_match = sm.fit(s, structure)
        if not is_match:
            warn(
                f"{i}-th original and decoded structures do not match according to StructureMatcher(comparator=ElementComparator()).fit(s, structure).\n\nOriginal (s): {s}\n\nDecoded (structure): {structure}"  # noqa: E501
            )

        spa = SpacegroupAnalyzer(s, symprec=0.1, angle_tolerance=5.0)
        s = spa.get_refined_structure()
        spa = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5.0)
        structure = spa.get_refined_structure()

        sm = StructureMatcher(primitive_cell=False, comparator=ElementComparator())
        s2 = sm.get_s2_like_s1(s, structure)

        a_check = s._lattice.a
        b_check = s._lattice.b
        c_check = s._lattice.c
        angles_check = s._lattice.angles
        atomic_numbers_check = s.atomic_numbers
        frac_coords_check = s.frac_coords
        space_group_check = _get_space_group(s)

        latt_a = s2._lattice.a
        latt_b = s2._lattice.b
        latt_c = s2._lattice.c
        angles = s2._lattice.angles
        atomic_numbers = s2.atomic_numbers
        frac_coords = s2.frac_coords
        space_group = _get_space_group(s)

        assert_allclose(
            a_check,
            latt_a,
            rtol=RGB_LOOSE_TOL * tol_multiplier,
            err_msg="lattice parameter length `a` not all close",
        )

        assert_allclose(
            b_check,
            latt_b,
            rtol=RGB_LOOSE_TOL * tol_multiplier,
            err_msg="lattice parameter length `b` not all close",
        )

        assert_allclose(
            c_check,
            latt_c,
            rtol=RGB_LOOSE_TOL * 2 * tol_multiplier,
            err_msg="lattice parameter length `c` not all close",
        )

        assert_allclose(
            angles_check,
            angles,
            rtol=RGB_LOOSE_TOL * tol_multiplier,
            err_msg="lattice parameter angles not all close",
        )

        assert_allclose(
            atomic_numbers_check,
            atomic_numbers,
            rtol=RGB_LOOSE_TOL * tol_multiplier,
            err_msg="atomic numbers not all close",
        )

        # use atol since frac_coords values are between 0 and 1
        assert_allclose(
            frac_coords_check,
            frac_coords,
            atol=RGB_TOL * tol_multiplier,
            err_msg="atomic numbers not all close",
        )

        assert_equal(
            space_group_check,
            space_group,
            err_msg=f"space groups do not match. Original: {space_group_check}. Decoded: {space_group}.",  # noqa: E501
        )
