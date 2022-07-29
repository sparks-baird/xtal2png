"""Crystal to PNG conversion core functions and scripts."""
import logging
import sys
from functools import lru_cache
from itertools import chain

# from itertools import zip_longest
from os import PathLike, path
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from uuid import uuid4
from warnings import warn

import numpy as np
import numpy.typing as npt
import pandas as pd
from element_coder import decode_many, encode_many
from element_coder.utils import get_range
from numpy.typing import NDArray
from PIL import Image
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from tqdm import tqdm

from xtal2png.utils.data import (
    _get_space_group,
    dummy_structures,
    element_wise_scaler,
    element_wise_unscaler,
    get_image_mode,
    rgb_scaler,
    rgb_unscaler,
    unit_cell_converter,
)

# from sklearn.preprocessing import MinMaxScaler


__author__ = "sgbaird"
__copyright__ = "sgbaird"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via `from xtal2png.core import
# XtalConverter`, when using this Python module as a library.


ATOM_ID = 1
FRAC_ID = 2
A_ID = 3
B_ID = 4
C_ID = 5
ANGLES_ID = 6
NUM_SITES_ID = 7
SPACE_GROUP_ID = 8
DISTANCE_ID = 9

ATOM_KEY = "atom"
FRAC_KEY = "frac"
A_KEY = "a"
B_KEY = "b"
C_KEY = "c"
ANGLES_KEY = "angles"
NUM_SITES_KEY = "num_sites"
SPACE_GROUP_KEY = "space_group"
DISTANCE_KEY = "distance"
LOWER_TRI_KEY = "lower_tri"

SUPPORTED_MASK_KEYS = [
    ATOM_KEY,
    FRAC_KEY,
    A_KEY,
    B_KEY,
    C_KEY,
    ANGLES_KEY,
    NUM_SITES_KEY,
    SPACE_GROUP_KEY,
    DISTANCE_KEY,
    LOWER_TRI_KEY,
]


def construct_save_name(s: Structure) -> str:
    """Construct savename based on formula, space group, and a uid."""
    save_name = f"{s.formula.replace(' ', '')},space-group={_get_space_group(s)},uid={str(uuid4())[0:4]}"  # noqa: E501
    return save_name


@lru_cache(maxsize=None)
def _element_encoding_range_cached(elements, encoding_type):
    return get_range(elements, encoding_type)


class XtalConverter:
    """Convert between pymatgen Structure object and PNG-encoded representation.

    Note that if you modify the ranges to be different than their defaults, you have
    effectively created a new representation. In the future, anytime you use
    :func:`XtalConverter` with a dataset that used modified range(s), you will need to
    specify the same ranges; otherwise, your data will be decoded (unscaled)
    incorrectly. In other words, make sure you're using the same :func:`XtalConverter`
    object for both encoding and decoding.

    We encourage you to use the default ranges, which were carefully selected based on a
    trade-off between keeping the range as low as possible and trying to incorporate as
    much of what's been observed on Materials Project with no more than 52 sites. For
    more details, see the corresponding notebook in the ``notebooks`` directory:
    https://github.com/sparks-baird/xtal2png/tree/main/notebooks

    Parameters
    ----------
    atom_range : Tuple[int, int], optional
        Expected range for atomic number, by default (1, 118)
    frac_range : Tuple[float, float], optional
        Expected range for fractional coordinates, by default (0.0, 1.0)
    a_range : Tuple[float, float], optional
        Expected range for lattice parameter length a, by default (2.0, 15.3)
    b_range : Tuple[float, float], optional
        Expected range for lattice parameter length b, by default (2.0, 15.0)
    c_range : Tuple[float, float], optional
        Expected range for lattice parameter length c, by default (2.0, 36.0)
    angles_range : Tuple[float, float], optional
        Expected range for lattice parameter angles, by default (0.0, 180.0)
    num_sites_range : Tuple[float, float], optional
        Expected range for unit cell num_sites, by default (0, 52)
    space_group_range : Tuple[int, int], optional
        Expected range for space group numbers, by default (1, 230)
    distance_range : Tuple[float, float], optional
        Expected range for pairwise distances between sites, by default (0.0, 25.0)
    max_sites : int, optional
        Maximum number of sites to accomodate in encoding, by default 52
    save_dir : Union[str, 'PathLike[str]']
        Directory to save PNG files via :func:`xtal2png`,
        by default path.join("data", "interim")
    symprec : Union[float, Tuple[float, float]], optional
        The symmetry precision to use when decoding `pymatgen` structures via
        :func:`pymatgen.symmetry.analyzer.SpaceGroupAnalyzer.get_refined_structure`. If
        specified as a tuple, then ``symprec[0]`` applies to encoding and ``symprec[1]``
        applies to decoding. By default 0.1.
    angle_tolerance : Union[float, int, Tuple[float, float], Tuple[int, int]], optional
        The angle tolerance (degrees) to use when decoding `pymatgen` structures via
        :func:`pymatgen.symmetry.analyzer.SpaceGroupAnalyzer.get_refined_structure`. If
        specified as a tuple, then ``angle_tolerance[0]`` applies to encoding and
        ``angle_tolerance[1]`` applies to decoding. By default 5.0.
    encode_cell_type : Optional[str], optional
        Encode structures as-is (None), or after applying a certain tranformation. Uses
        ``symprec`` if ``symprec`` is of type float, else uses ``symprec[0]`` if
        ``symprec`` is of type tuple. Same applies for ``angle_tolerance``. By default
        None
    decode_cell_type : Optional[str], optional
        Decode structures as-is (None), or after applying a certain tranformation. Uses
        ``symprec`` if ``symprec`` is of type float, else uses ``symprec[0]`` if
        ``symprec`` is of type tuple. Same applies for ``angle_tolerance``. By default
        None
    relax_on_decode: bool, optional
        Use m3gnet to relax the decoded crystal structures.
    channels : int, optional
        Number of channels, a positive integer. Typically choices would be 1 (grayscale)
        or 3 (RGB), and are the only compatible choices when using
        :func:`XtalConverter().xtal2png` and :func:`XtalConverter().png2xtal`. For
        positive integers other than 1 or 3, use
        :func:`XtalConverter().structures_to_arrays` and
        :func:`XtalConverter().arrays_to_structures` directly instead.
    verbose: bool, optional
        Whether to print verbose debugging information or not.
    element_encoding : str
        How to encode the element. Can be one of
        `element_coder.data.coding_data._PROPERTY_KEYS` (e.g., `mod_pettifor`, `atomic`,
        `pettifor`, `X`). Defaults to `atomic` (which encodes elements as atomic
        numbers).
    element_decoding_metric: Union[str, callable]
        Metric to measure distance between (noisy) input encoding and tabulated
        encodings.
        If a string, the distance function can be 'braycurtis', 'canberra', 'chebyshev',
        'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard',
        'jensenshannon', 'kulsinski', 'kulczynski1', 'mahalanobis', 'matching',
        'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
        'sokalsneath', 'sqeuclidean', 'yule'. Defaults to "euclidean".
    mask_types : List[str], optional
        List of information types to mask out (assign as 0) from the array/image. values
        are "atom", "frac", "a", "b", "c", "angles", "num_sites", "space_group",
        "distance", "diagonal", and None. If None, then no masking is applied. If
        "diagonal" is present, then zeros out the lower triangle. By default, None.

    Examples
    --------
    >>> xc = XtalConverter()

    >>> xc = XtalConverter(atom_range=(0, 83)) # assumes no radioactive elements in data
    """

    def __init__(
        self,
        atom_range: Union[Tuple[int, int], npt.ArrayLike] = (1, 118),
        frac_range: Tuple[float, float] = (0.0, 1.0),
        a_range: Tuple[float, float] = (2.0, 15.3),
        b_range: Tuple[float, float] = (2.0, 15.0),
        c_range: Tuple[float, float] = (2.0, 36.0),
        angles_range: Tuple[float, float] = (0.0, 180.0),
        num_sites_range: Tuple[float, float] = (0, 52),
        space_group_range: Tuple[int, int] = (1, 230),
        distance_range: Tuple[float, float] = (0.0, 18.0),
        max_sites: int = 52,
        save_dir: Union[str, "PathLike[str]"] = path.join("data", "preprocessed"),
        symprec: Union[float, Tuple[float, float]] = 0.1,
        angle_tolerance: Union[float, int, Tuple[float, float], Tuple[int, int]] = 5.0,
        encode_cell_type: Optional[str] = None,
        decode_cell_type: Optional[str] = None,
        relax_on_decode: bool = False,
        channels: int = 1,
        verbose: bool = True,
        element_encoding: str = "atomic",
        element_decoding_metric: Union[str, Callable] = "euclidean",
        mask_types: List[str] = [],
    ):
        """Instantiate an XtalConverter object with desired ranges and ``max_sites``."""
        self.atom_range = atom_range
        self.frac_range = frac_range
        self.a_range = a_range
        self.b_range = b_range
        self.c_range = c_range
        self.angles_range = angles_range
        self.num_sites_range = num_sites_range
        self.space_group_range = space_group_range
        self.distance_range = distance_range
        self.max_sites = max_sites
        self.save_dir = save_dir
        self.element_encoding = element_encoding
        self.element_decoding_metric = element_decoding_metric

        if isinstance(symprec, (float, int)):
            self.encode_symprec = symprec
            self.decode_symprec = symprec
        elif isinstance(symprec, tuple):
            self.encode_symprec = symprec[0]
            self.decode_symprec = symprec[1]

        if isinstance(angle_tolerance, (float, int)):
            self.encode_angle_tolerance = angle_tolerance
            self.decode_angle_tolerance = angle_tolerance
        elif isinstance(angle_tolerance, tuple):
            self.encode_angle_tolerance = angle_tolerance[0]
            self.decode_angle_tolerance = angle_tolerance[1]

        self.encode_cell_type = encode_cell_type
        self.decode_cell_type = decode_cell_type
        self.relax_on_decode = relax_on_decode

        self.channels = channels
        self.verbose = verbose

        if self.verbose:
            self.tqdm_if_verbose = tqdm
        else:
            self.tqdm_if_verbose = lambda x: x

        unsupported_mask_types = np.setdiff1d(mask_types, SUPPORTED_MASK_KEYS).tolist()
        if unsupported_mask_types != []:
            raise ValueError(
                f"{unsupported_mask_types} is/are not a valid mask type. Expected one of {SUPPORTED_MASK_KEYS}. Received {mask_types}"  # noqa: E501
            )

        self.mask_types = mask_types

        Path(save_dir).mkdir(exist_ok=True, parents=True)

    @property
    def _element_encoding_range(self):
        # We do *not* use cached_property as the result might change as
        # the users calls fit. However, we still want to cache, as we reuse
        # the result of this method for both encoding and decoding.
        return _element_encoding_range_cached(self.atom_range, self.element_encoding)

    def xtal2png(
        self,
        structures: List[Union[Structure, str, "PathLike[str]"]],
        show: bool = False,
        save: bool = True,
    ):
        """Encode crystal (via CIF filepath or Structure object) as PNG file.

        Parameters
        ----------
        structures : List[Union[Structure, str, PathLike[str]]]
            pymatgen Structure objects or path to CIF files.
        show : bool, optional
            Whether to display the PNG-encoded file, by default False
        save : bool, optional
            Whether to save the PNG-encoded file, by default True

        Returns
        -------
        imgs : List[Image.Image]
            PIL images that (approximately) encode the supplied crystal structures.

        Raises
        ------
        ValueError
            structures should be of same datatype
        ValueError
            structures should be of same datatype
        ValueError
            structures should be of type `str`, `os.PathLike` or
            `pymatgen.core.structure.Structure`

        Examples
        --------
        >>> coords = [[0, 0, 0], [0.75,0.5,0.75]]
        >>> lattice = Lattice.from_parameters(
        ...     a=3.84, b=3.84, c=3.84, alpha=120, beta=90, gamma=60
        ... )
        >>> structures = [Structure(lattice, ["Si", "Si"], coords),
        ... Structure(lattice, ["Ni", "Ni"], coords)]
        >>> xc = XtalConverter()
        >>> xc.xtal2png(structures, show=False, save=True)
        """
        self.savenames, S = self.process_filepaths_or_structures(structures)

        # convert structures to 3D NumPy Matrices
        self.data, self.id_data, self.id_mapper = self.structures_to_arrays(S)
        mn, mx = self.data.min(), self.data.max()
        if mn < 0:
            warn(
                f"lower RGB value(s) OOB ({mn} less than 0). thresholding to 0.. may throw off crystal structure parameters (e.g. if lattice parameters are thresholded)"  # noqa: E501
            )  # noqa
            self.data[self.data < 0] = 0

        if mx > 255:
            warn(
                f"upper RGB value(s) OOB ({mx} greater than 255). thresholding to 255.. may throw off crystal structure parameters (e.g. if lattice parameters are thresholded)"  # noqa: E501
            )  # noqa
            self.data[self.data > 255] = 255

        self.data = self.data.astype(np.uint8)

        # convert to PNG images. Save and/or show, if applicable
        imgs: List[Image.Image] = []
        for d, save_name in zip(self.data, self.savenames):
            mode = get_image_mode(d)
            d = np.squeeze(d)
            if mode == "RGB":
                d = d.transpose(1, 2, 0)
            img = Image.fromarray(d, mode=mode)
            imgs.append(img)
            if save:
                savepath = path.join(self.save_dir, save_name + ".png")
                img.save(savepath)
            if show:
                img.show()

        return imgs

    def fit(
        self,
        structures: List[Union[Structure, str, "PathLike[str]"]],
        y=None,
        fit_quantiles: Tuple[float, float] = (0.00, 0.99),
        verbose: Optional[bool] = None,
    ):
        """Find optimal range parameters for encoding crystal structures.

        Parameters
        ----------
        structures : List[Union[Structure, str, "PathLike[str]"]]
            List of pymatgen Structure objects.
        y : NoneType, optional
            No effect, for compatibility only, by default None
        fit_quantiles : Tuple[float,float], optional
            The lower and upper quantiles to use for fitting ranges to the data, by
            default (0.00, 0.99)
        verbose : Optional[bool], optional
            Whether to print information about the fitted ranges. If None, then defaults
            to ``self.verbose``. By default None

        Examples
        --------
        >>> fit(structures, , y=None, fit_quantiles=(0.00, 0.99), verbose=None, )
        OUTPUT
        """
        verbose = self.verbose if verbose is None else verbose

        _, S = self.process_filepaths_or_structures(structures)

        # TODO: deal with arbitrary site_properties
        atomic_numbers = [s.atomic_numbers for s in S]
        a = [s.lattice.a for s in S]
        b = [s.lattice.b for s in S]
        c = [s.lattice.c for s in S]
        space_group = [_get_space_group(s) for s in S]
        num_sites = [s.num_sites for s in S]
        distance = [s.distance_matrix for s in S]

        if verbose:
            print("range of atomic_numbers is: ", min(a), "-", max(a))
            print("range of a is: ", min(a), "-", max(a))
            print("range of b is: ", min(b), "-", max(b))
            print("range of c is: ", min(c), "-", max(c))
            print("range of space_group is: ", min(space_group), "-", max(space_group))
            print("range of num_sites is: ", min(num_sites), "-", max(num_sites))

        dis_min_tmp = []
        dis_max_tmp = []
        for d in tqdm(range(len(distance))):
            dis_min_tmp.append(min(distance[d][np.nonzero(distance[d])]))
            dis_max_tmp.append(max(distance[d][np.nonzero(distance[d])]))

        atoms = np.array(atomic_numbers, dtype="object")
        uniq_atoms = np.unique(list(chain(*atomic_numbers)))
        self._atom_range = [np.min(uniq_atoms), np.max(uniq_atoms)]
        self.atom_range = atoms
        self.space_group_range = (np.min(space_group), np.max(space_group))
        self.num_sites_range = (np.min(num_sites), np.max(num_sites))

        self.num_sites = np.max(num_sites)

        df = pd.DataFrame(
            dict(
                a=a,
                b=b,
                c=c,
                min_distance=dis_min_tmp,
                max_distance=dis_max_tmp,
            )
        )

        low_quantile, upp_quantile = fit_quantiles

        low_df = (
            df.apply(lambda a: np.quantile(a, low_quantile))
            .drop(["max_distance"])
            .rename(index={"min_distance": "distance"})
        )
        upp_df = (
            df.apply(lambda a: np.quantile(a, upp_quantile))
            .drop(["min_distance"])
            .rename(index={"max_distance": "distance"})
        )
        low_df.name = "low"
        upp_df.name = "upp"

        range_df = pd.concat((low_df, upp_df), axis=1)

        for name, bounds in range_df.iterrows():
            setattr(self, name + "_range", tuple(bounds))

    def process_filepaths_or_structures(
        self,
        structures: List[Union[Structure, str, "PathLike[str]"]],
    ) -> Tuple[List[str], List[Structure]]:
        """Extract (or create) save names and convert/passthrough the structures.

        Parameters
        ----------
        structures : Union[PathLike, Structure]
            List of filepaths or list of structures to be processed.

        Returns
        -------
        savenames : List[str]
            Save names of the files if filepaths are passed, otherwise some relatively
            unique names (due to 4 random characters being appended at the end) for each
            structure. See ``construct_save_name``.

        S : List[Structure]
            Processed structures.

        Raises
        ------
        ValueError
            "structures should be of same datatype, either strs or pymatgen Structures.
            structures[0] is {type(structures[0])}, but got type {type(s)} for entry
            {i}"
        ValueError
            "structures should be of same datatype, either strs or pymatgen Structures.
            structures[0] is {type(structures[0])}, but got type {type(s)} for entry
            {i}"
        ValueError
            "structures should be of type `str`, `os.PathLike` or
            `pymatgen.core.structure.Structure`, not {type(structures[i])} (entry {i})"

        Examples
        --------
        >>> savenames, structures = process_filepaths_or_structures(structures)
        """
        savenames: List[str] = []

        first_is_structure = isinstance(structures[0], Structure)
        for i, s in enumerate(structures):
            if isinstance(s, str) or isinstance(s, PathLike):
                if first_is_structure:
                    raise ValueError(
                        f"structures should be of same datatype, either strs or pymatgen Structures. structures[0] is {type(structures[0])}, but got type {type(s)} for entry {i}"  # noqa: E501
                    )

                structures[i] = Structure.from_file(s)
                savenames.append(Path(str(s)).stem)

            elif isinstance(s, Structure):
                if not first_is_structure:
                    raise ValueError(
                        f"structures should be of same datatype, either strs or pymatgen Structures. structures[0] is {type(structures[0])}, but got type {type(s)} for entry {i}"  # noqa
                    )

                structures[i] = s
                savenames.append(construct_save_name(s))
            else:
                raise ValueError(
                    f"structures should be of type `str`, `os.PathLike` or `pymatgen.core.structure.Structure`, not {type(structures[i])} (entry {i})"  # noqa
                )

        for i, s in enumerate(structures):
            assert isinstance(
                s, Structure
            ), f"structures[{i}]: {type(s)}, expected: Structure"
            assert not isinstance(s, str) and not isinstance(s, PathLike)
            if not s.is_ordered:
                raise ValueError(
                    "xtal2png does not support disordered structures. "
                    "Your input structure seems to contain partial occupancies. "
                    "Please resolve those and try again."
                )

        return savenames, structures  # type: ignore

    def png2xtal(
        self, images: List[Union[Image.Image, "PathLike"]], save: bool = False
    ) -> List[Structure]:
        """Decode PNG files as Structure objects.

        Parameters
        ----------
        images : List[Union[Image.Image, 'PathLike']]
            PIL images that (approximately) encode crystal structures.

        Examples
        --------
        >>> from xtal2png.utils.data import example_structures
        >>> xc = XtalConverter()
        >>> imgs = xc.xtal2png(example_structures)
        >>> xc.png2xtal(imgs)
        OUTPUT
        """
        if not isinstance(images, list):
            raise ValueError(
                f"images (or filepaths) should be of type list, received {type(images)}"
            )

        data_tmp = []
        if self.channels == 1:
            mode = "L"
        elif self.channels == 3:
            mode = "RGB"
        else:
            raise ValueError(
                f"expected grayscale (1-channel) or RGB (3-channels) image, but got {self.channels}-channels. Either set channels to 1 or 3 or use xc.structures_to_arrays and xc.arrays_to_structures directly instead of xc.xtal2png and xc.png2xtal"  # noqa: E501
            )
        for img in images:
            if isinstance(img, str):
                # load image from file
                with Image.open(img).convert(mode) as im:
                    arr = np.asarray(im)
            elif isinstance(img, Image.Image):
                arr = np.asarray(img.convert(mode))
            if mode == "RGB":
                arr = arr.transpose(2, 0, 1)
            data_tmp.append(arr)

        data = np.stack(data_tmp, axis=0)

        if mode == "L":
            data = np.expand_dims(data, 1)

        S = self.arrays_to_structures(data)

        if save:
            for s in self.tqdm_if_verbose(S):
                fpath = path.join(self.save_dir, construct_save_name(s) + ".cif")
                CifWriter(
                    s,
                    symprec=self.decode_symprec,
                    angle_tolerance=self.decode_angle_tolerance,
                ).write_file(fpath)

        return S

        # unscale values

    def structures_to_arrays(
        self,
        structures: Sequence[Structure],
        rgb_scaling=True,
    ) -> Tuple[NDArray, NDArray, Dict[str, int]]:
        """Convert pymatgen Structure to scaled 3D array of crystallographic info.

        ``atomic_numbers`` and ``distance_matrix`` get padded or cropped as appropriate,
        as these depend on the number of sites in the structure.

        Parameters
        ----------
        S : Sequence[Structure]
            Sequence (e.g. list) of pymatgen Structure object(s)

        Returns
        -------
        data : ArrayLike
            RGB-scaled arrays with first dimension corresponding to each crystal
            structure.

        id_data : ArrayLike
            Same shape as ``data``, except one-hot encoded to distinguish between the
            various types of information contained in ``data``. See ``id_mapper`` for
            the "legend" for this data.

        id_mapper : ArrayLike
            Dictionary containing the legend/key between the names of the blocks and the
            corresponding numbers in ``id_data``.

        Raises
        ------
        ValueError
            "`structures` should be a list of pymatgen Structure(s)"
        ValueError
            "crystal supplied with {n_sites} sites, which is more than {self.max_sites}
            sites. Remove crystal or increase `max_sites`."
        ValueError
            "len(atomic_numbers) {n_sites} and distance_matrix.shape[0]
            {s.distance_matrix.shape[0]} do not match"

        Examples
        --------
        >>> xc = XtalConverter()
        >>> data = xc.structures_to_arrays(structures)
        OUTPUT
        """

        for s in structures:
            if not isinstance(s, Structure):
                raise ValueError(
                    "`structures` should be a list of pymatgen Structure(s)"
                )
            if not s.is_ordered:
                raise ValueError(
                    "xtal2png does not support disordered structures. "
                    "Your input structure seems to contain partial occupancies. "
                    "Please resolve those and try again."
                )

        # extract crystallographic information
        element_encoding: List[List[int]] = []
        frac_coords_tmp: List[NDArray] = []
        latt_a: List[float] = []
        latt_b: List[float] = []
        latt_c: List[float] = []
        angles: List[List[float]] = []
        num_sites: List[float] = []
        space_group: List[int] = []
        distance_matrix_tmp: List[NDArray[np.float64]] = []

        for s in self.tqdm_if_verbose(structures):
            s = unit_cell_converter(
                s,
                self.encode_cell_type,
                symprec=self.encode_symprec,
                angle_tolerance=self.encode_angle_tolerance,
            )  # noqa: E501

            n_sites = len(s.atomic_numbers)
            if n_sites > self.max_sites:
                raise ValueError(
                    f"crystal supplied with {n_sites} sites, which is more than {self.max_sites} sites. Remove the offending crystal(s), increase `max_sites`, or use a more compact cell_type (see encode_cell_type and decode_cell_type kwargs)."  # noqa: E501
                )
            element_encoding.append(
                np.pad(
                    encode_many(s.atomic_numbers, self.element_encoding),
                    (0, self.max_sites - n_sites),
                ).tolist()
            )
            frac_coords_tmp.append(
                np.pad(s.frac_coords, ((0, self.max_sites - n_sites), (0, 0)))
            )
            latt_a.append(s._lattice.a)
            latt_b.append(s._lattice.b)
            latt_c.append(s._lattice.c)
            angles.append(list(s._lattice.angles))
            num_sites.append(s.num_sites)
            space_group.append(_get_space_group(s))

            dm = s.distance_matrix  # avoid repeat calculation
            if n_sites != dm.shape[0]:
                raise ValueError(
                    f"len(atomic_numbers) {n_sites} and distance_matrix.shape[0] {dm.shape[0]} do not match"  # noqa
                )  # noqa

            # assume that distance matrix is square
            padwidth = (0, self.max_sites - n_sites)
            distance_matrix_tmp.append(np.pad(dm, padwidth))
            # [0:max_sites, 0:max_sites]

        frac_coords = np.stack(frac_coords_tmp)
        distance_matrix = np.stack(distance_matrix_tmp)

        if rgb_scaling:
            # REVIEW: consider using modified pettifor scale instead of atomic numbers
            # REVIEW: consider using feature_range=atom_range or 2*atom_range
            # REVIEW: since it introduces a sort of non-linearity b.c. of rounding
            # ToDo: the range below is not optimal. For this the fit should return a
            # list of all the elements
            atom_scaled = rgb_scaler(
                element_encoding,
                data_range=self._element_encoding_range,
            )  # noqa
            frac_scaled = rgb_scaler(frac_coords, data_range=self.frac_range)
            a_scaled = rgb_scaler(latt_a, data_range=self.a_range)
            b_scaled = rgb_scaler(latt_b, data_range=self.b_range)
            c_scaled = rgb_scaler(latt_c, data_range=self.c_range)
            angles_scaled = rgb_scaler(angles, data_range=self.angles_range)
            num_sites_scaled = rgb_scaler(num_sites, data_range=self.num_sites_range)
            space_group_scaled = rgb_scaler(
                space_group, data_range=self.space_group_range
            )
            # NOTE: max_distance could be added as another (repeated) value/row to infer
            # NOTE: kind of like frac_distance_matrix, not sure if would be effective
            # NOTE: Or could normalize distance_matix by cell volume
            # NOTE: and possibly include cell volume as a (repeated) value/row to infer
            # NOTE: It's possible extra info like this isn't so bad, instilling the
            # physics
            # NOTE: but it could also just be extraneous work to predict/infer
            distance_scaled = rgb_scaler(
                distance_matrix, data_range=self.distance_range
            )

        else:
            feature_range = (0.0, 1.0)
            atom_scaled = element_wise_scaler(
                element_encoding,
                feature_range=feature_range,
                data_range=self._element_encoding_range,
            )
            frac_scaled = element_wise_scaler(
                frac_coords, feature_range=feature_range, data_range=self.frac_range
            )
            a_scaled = element_wise_scaler(
                latt_a, feature_range=feature_range, data_range=self.a_range
            )
            b_scaled = element_wise_scaler(
                latt_b, feature_range=feature_range, data_range=self.b_range
            )
            c_scaled = element_wise_scaler(
                latt_c, feature_range=feature_range, data_range=self.c_range
            )
            angles_scaled = element_wise_scaler(
                angles, feature_range=feature_range, data_range=self.angles_range
            )
            num_sites_scaled = element_wise_scaler(
                num_sites, feature_range=feature_range, data_range=self.num_sites_range
            )
            space_group_scaled = element_wise_scaler(
                space_group,
                feature_range=feature_range,
                data_range=self.space_group_range,
            )
            distance_scaled = element_wise_scaler(
                distance_matrix,
                feature_range=feature_range,
                data_range=self.distance_range,
            )

        atom_arr = np.expand_dims(atom_scaled, 2)
        frac_arr = frac_scaled
        a_arr = np.repeat(np.expand_dims(a_scaled, (1, 2)), self.max_sites, axis=1)
        b_arr = np.repeat(np.expand_dims(b_scaled, (1, 2)), self.max_sites, axis=1)
        c_arr = np.repeat(np.expand_dims(c_scaled, (1, 2)), self.max_sites, axis=1)
        angles_arr = np.repeat(np.expand_dims(angles_scaled, 1), self.max_sites, axis=1)
        num_sites_arr = np.repeat(
            np.expand_dims(num_sites_scaled, (1, 2)), self.max_sites, axis=1
        )
        space_group_arr = np.repeat(
            np.expand_dims(space_group_scaled, (1, 2)), self.max_sites, axis=1
        )
        distance_arr = distance_scaled

        data = self.assemble_blocks(
            atom_arr,
            frac_arr,
            a_arr,
            b_arr,
            c_arr,
            angles_arr,
            num_sites_arr,
            space_group_arr,
            distance_arr,
        )

        id_mapper = {
            ATOM_KEY: ATOM_ID,
            FRAC_KEY: FRAC_ID,
            A_KEY: A_ID,
            B_KEY: B_ID,
            C_KEY: C_ID,
            ANGLES_KEY: ANGLES_ID,
            NUM_SITES_KEY: NUM_SITES_ID,
            SPACE_GROUP_KEY: SPACE_GROUP_ID,
            DISTANCE_KEY: DISTANCE_ID,
        }

        id_blocks = [
            np.ones_like(atom_arr) * ATOM_ID,
            np.ones_like(frac_arr) * FRAC_ID,
            np.ones_like(a_arr) * A_ID,
            np.ones_like(b_arr) * B_ID,
            np.ones_like(c_arr) * C_ID,
            np.ones_like(angles_arr) * ANGLES_ID,
            np.ones_like(num_sites_arr) * NUM_SITES_ID,
            np.ones_like(space_group_arr) * SPACE_GROUP_ID,
            np.ones_like(distance_arr) * DISTANCE_ID,
        ]
        id_data = self.assemble_blocks(*id_blocks)

        # apply num_sites mask (zero out bottom and RHS blocks past num_sites)
        data = self.apply_num_sites_mask(data, num_sites)
        id_data = self.apply_num_sites_mask(id_data, num_sites)

        data = np.expand_dims(data, 1)
        id_data = np.expand_dims(id_data, 1)

        for mask_type in self.mask_types:
            if mask_type == LOWER_TRI_KEY:
                for d in data:
                    if d.shape[1] != d.shape[2]:
                        raise ValueError(
                            f"Expected square matrix in last two dimensions, received {d.shape}"  # noqa: E501
                        )
                    d[:, np.mask_indices(d.shape[1], np.tril)] = 0.0
            else:
                data[id_data == id_mapper[mask_type]] = 0.0

        data = np.repeat(data, self.channels, 1)
        id_data = np.repeat(id_data, self.channels, 1)

        return data, id_data, id_mapper

    def assemble_blocks(
        self,
        atom_arr,
        frac_arr,
        a_arr,
        b_arr,
        c_arr,
        angles_arr,
        num_sites,
        space_group_arr,
        distance_arr,
    ) -> NDArray:
        arrays = [
            atom_arr,
            frac_arr,
            a_arr,
            b_arr,
            c_arr,
            angles_arr,
            num_sites,
            space_group_arr,
        ]
        zero_pad = sum([arr.shape[2] for arr in arrays])
        n_structures = atom_arr.shape[0]
        zero = np.zeros((n_structures, zero_pad, zero_pad), dtype=int)

        vertical_arr = np.block(
            [
                [zero],
                [
                    atom_arr,
                    frac_arr,
                    a_arr,
                    b_arr,
                    c_arr,
                    angles_arr,
                    num_sites,
                    space_group_arr,
                ],
            ]
        )
        horizontal_arr = np.block(
            [
                atom_arr,
                frac_arr,
                a_arr,
                b_arr,
                c_arr,
                angles_arr,
                num_sites,
                space_group_arr,
            ]
        )
        horizontal_arr = np.moveaxis(horizontal_arr, 1, 2)
        left_arr = vertical_arr
        right_arr = np.block([[horizontal_arr], [distance_arr]])
        data = np.block([left_arr, right_arr])
        return data

    def disassemble_blocks(
        self,
        data,
        # id_data: Optional[NDArray] = None,
        # id_mapper: Optional[dict] = None,
    ):
        # TODO: implement a more robust solution using id_data and id_mapper

        # if (id_data is None) is not (id_mapper is None):
        #     raise ValueError(
        #         f"id_data (type: {type(id_data)}) and id_mapper (type: {type(id_mapper)}) should either both be assigned or both be None."  # noqa
        #     )
        # elif id_data is None and id_mapper is None:
        #     _, id_data, id_mapper = self.structures_to_arrays(dummy_structures)

        # assert (
        #     id_data is not None and id_mapper is not None
        # ), "id_data and id_mapper should not be None at this point"

        zero_pad = 12
        left_arr, right_arr = np.array_split(data, [zero_pad], axis=1)
        _, bottom_left = np.array_split(left_arr, [zero_pad], axis=2)

        lengths = [1, 3, 1, 1, 1, 3, 1]

        verts = np.array_split(bottom_left, np.cumsum(lengths), axis=1)

        top_right, bottom_right = np.array_split(right_arr, [zero_pad], axis=2)
        distance_arr = bottom_right

        horzs = np.array_split(top_right, np.cumsum(lengths), axis=2)

        def average_vert_horz(vert, horz):
            vert = np.float64(vert)
            horz = np.float64(horz)
            avg = (vert.swapaxes(1, 2) + horz) / 2
            return avg

        avgs = [average_vert_horz(v, h) for v, h in zip(verts, horzs)]

        (
            atom_arr,
            frac_arr,
            a_arr,
            b_arr,
            c_arr,
            angles_arr,
            num_sites_arr,
            space_group_arr,
        ) = avgs

        return (
            atom_arr,
            frac_arr,
            a_arr,
            b_arr,
            c_arr,
            angles_arr,
            num_sites_arr,
            space_group_arr,
            distance_arr,
        )

    def arrays_to_structures(
        self,
        data: np.ndarray,
        id_data: Optional[np.ndarray] = None,
        id_mapper: Optional[dict] = None,
        rgb_scaling: bool = True,
    ) -> List[Structure]:
        """Convert scaled crystal (xtal) arrays to pymatgen Structures.

        Parameters
        ----------
        data : np.ndarray
            3D array containing crystallographic information.

        id_data : ArrayLike
            Same shape as ``data``, except one-hot encoded to distinguish between the
            various types of information contained in ``data``. See ``id_mapper`` for
            the "legend" for this data.

        id_mapper : ArrayLike
            Dictionary containing the legend/key between the names of the blocks and the
            corresponding numbers in ``id_data``.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError(
                f"`data` should be of type `np.ndarray`.  Received type {type(data)}. Maybe you passed a tuple of (data, id_data, id_mapper) returned from `structures_to_arrays()` by accident?"  # noqa: E501
            )

        # convert to single channel and remove singleton dimension before disassembly
        data = np.mean(data, axis=1)

        # to extract num_sites for preprocess masking of data
        if id_data is None and id_mapper is None:
            _, id_data, id_mapper = self.structures_to_arrays(dummy_structures)

        if id_data is None or id_mapper is None:  # for mypy
            raise ValueError("id_data and id_mapper should not be None at this point")

        id_data = np.mean(id_data, axis=1)
        assert id_data is not None, "id_data should not be None at this point"

        num_sites = [d[id_data[0] == id_mapper[NUM_SITES_KEY]] for d in data]
        num_sites = [ns[np.where(ns > 0)] for ns in num_sites]
        num_sites = [np.mean(ns) for ns in num_sites]

        if rgb_scaling:
            num_sites = rgb_unscaler(num_sites, data_range=self.num_sites_range)
        else:
            num_sites = element_wise_unscaler(
                num_sites, feature_range=(0.0, 1.0), data_range=self.num_sites_range
            )
        assert isinstance(num_sites, np.ndarray)
        num_sites = np.round(num_sites).astype(int)

        data = self.apply_num_sites_mask(data, num_sites)

        # for decoding final crystal structure
        arrays = self.disassemble_blocks(
            data,
            #  id_data=id_data,
            #  id_mapper=id_mapper
        )

        (
            atom_scaled,
            frac_scaled,
            a_scaled_tmp,
            b_scaled_tmp,
            c_scaled_tmp,
            angles_scaled_tmp,
            _,
            space_group_scaled_tmp,
            distance_scaled,
        ) = [np.squeeze(arr, axis=2) if arr.shape[2] == 1 else arr for arr in arrays]

        a_scaled = np.mean(a_scaled_tmp, axis=1, where=a_scaled_tmp != 0)
        b_scaled = np.mean(b_scaled_tmp, axis=1, where=b_scaled_tmp != 0)
        c_scaled = np.mean(c_scaled_tmp, axis=1, where=c_scaled_tmp != 0)
        angles_scaled = np.mean(angles_scaled_tmp, axis=1, where=angles_scaled_tmp != 0)

        # num_sites_scaled = np.mean(num_sites_scaled_tmp, axis=1)
        space_group_scaled = np.round(np.mean(space_group_scaled_tmp, axis=1)).astype(
            int
        )

        if rgb_scaling:
            # ToDo: expose the distance options for the decoding
            unscaled_atom_encodings = [
                encoding
                for encoding in rgb_unscaler(
                    atom_scaled, data_range=self._element_encoding_range
                )
            ]

            atomic_symbols = [
                decode_many(
                    encoding, self.element_encoding, metric=self.element_decoding_metric
                )
                for encoding in unscaled_atom_encodings
            ]
            frac_coords = rgb_unscaler(frac_scaled, data_range=self.frac_range)
            latt_a = rgb_unscaler(a_scaled, data_range=self.a_range)
            latt_b = rgb_unscaler(b_scaled, data_range=self.b_range)
            latt_c = rgb_unscaler(c_scaled, data_range=self.c_range)
            angles = rgb_unscaler(angles_scaled, data_range=self.angles_range)

            # # num_sites, space_group, distance_matrix unecessary for making Structure
            # num_sites = rgb_unscaler(num_sites_scaled,
            # data_range=self.num_sites_range)
            space_group = rgb_unscaler(
                space_group_scaled, data_range=self.space_group_range
            )
            distance_matrix = rgb_unscaler(
                distance_scaled, data_range=self.distance_range
            )
        else:
            feature_range = (0.0, 1.0)
            unscaled_atom_encodings = [
                encoding
                for encoding in element_wise_unscaler(
                    atom_scaled,
                    feature_range=feature_range,
                    data_range=self._element_encoding_range,
                )
            ]
            atomic_symbols = [
                decode_many(
                    encoding, self.element_encoding, metric=self.element_decoding_metric
                )
                for encoding in unscaled_atom_encodings
            ]
            frac_coords = element_wise_unscaler(
                frac_scaled, feature_range=feature_range, data_range=self.frac_range
            )
            latt_a = element_wise_unscaler(
                a_scaled, feature_range=feature_range, data_range=self.a_range
            )
            latt_b = element_wise_unscaler(
                b_scaled, feature_range=feature_range, data_range=self.b_range
            )
            latt_c = element_wise_unscaler(
                c_scaled, feature_range=feature_range, data_range=self.c_range
            )
            angles = element_wise_unscaler(
                angles_scaled, feature_range=feature_range, data_range=self.angles_range
            )
            # num_sites = element_wise_unscaler(
            #     num_sites_scaled,
            #     feature_range=feature_range,
            #     data_range=self.num_sites_range,
            # )
            space_group = element_wise_unscaler(
                space_group_scaled,
                feature_range=feature_range,
                data_range=self.space_group_range,
            )
            distance_matrix = element_wise_unscaler(
                distance_scaled,
                feature_range=feature_range,
                data_range=self.distance_range,
            )

        # num_sites = np.round(num_sites).astype(int)

        for dm, ns in zip(distance_matrix, num_sites):
            np.fill_diagonal(dm, 0.0)
            # mask bottom and RHS via num_sites
            dm[ns:, :] = 0.0
            dm[:, ns:] = 0.0

        # technically unused, but to avoid issue with pre-commit for now:
        space_group, distance_matrix

        # TODO: tweak lattice parameters to match predicted space group rules

        if self.relax_on_decode:
            try:
                import tensorflow as tf
                from m3gnet.models import Relaxer
            except ImportError as e:
                print(e)
                print(
                    "For Windows users on Anaconda, you need to `pip install m3gnet` or set relax_on_decode=False."  # noqa: E501
                )
            if not self.verbose:
                tf.get_logger().setLevel(logging.ERROR)
            relaxer = Relaxer()  # This loads the default pre-trained model

        # build Structure-s
        S: List[Structure] = []

        num_structures = len(atomic_symbols)

        for i in self.tqdm_if_verbose(range(num_structures)):
            ns = num_sites[i]
            at = atomic_symbols[i][:ns]
            fr = frac_coords[i][:ns]

            a, b, c = latt_a[i], latt_b[i], latt_c[i]
            alpha, beta, gamma = angles[i]

            lattice = Lattice.from_parameters(
                a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma
            )
            s = Structure(lattice, at, fr)

            # REVIEW: round fractional coordinates to nearest multiple?
            if self.relax_on_decode:
                relaxed_results = relaxer.relax(s, verbose=self.verbose)
                s = relaxed_results["final_structure"]

            s = unit_cell_converter(
                s,
                self.decode_cell_type,
                symprec=self.decode_symprec,
                angle_tolerance=self.decode_angle_tolerance,
            )

            S.append(s)

        return S

    def apply_num_sites_mask(self, data, num_sites):
        tot = data.shape[-1]
        for d, ns in zip(data, num_sites):
            filler_dim = tot - self.max_sites  # i.e. 12
            # apply mask to bottom and RHS blocks
            d[:, filler_dim + ns :] = 0.0
            d[filler_dim + ns :, :] = 0.0

        return data


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )
