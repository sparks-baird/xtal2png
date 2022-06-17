"""Crystal to PNG conversion core functions and scripts."""
import argparse
import logging
import sys
from glob import glob

# from itertools import zip_longest
from os import PathLike, path
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
from uuid import uuid4
from warnings import warn

import numpy as np
import pandas as pd
import tensorflow as tf
from m3gnet.models import Relaxer
from numpy.typing import NDArray
from PIL import Image
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

from xtal2png import __version__
from xtal2png.utils.data import dummy_structures, rgb_scaler, rgb_unscaler

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
VOLUME_ID = 7
SPACE_GROUP_ID = 8
DISTANCE_ID = 9

ATOM_KEY = "atom"
FRAC_KEY = "frac"
A_KEY = "latt_a"
B_KEY = "latt_b"
C_KEY = "latt_c"
ANGLES_KEY = "angles"
VOLUME_KEY = "volume"
SPACE_GROUP_KEY = "space_group"
DISTANCE_KEY = "distance"


def construct_save_name(s: Structure) -> str:
    save_name = f"{s.formula.replace(' ', '')},volume={int(np.round(s.volume))},uid={str(uuid4())[0:4]}"  # noqa: E501
    return save_name


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
        Expected range for atomic number, by default (0, 117)
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
    volume_range : Tuple[float, float], optional
        Expected range for unit cell volumes, by default (0.0, 1000.0)
    space_group_range : Tuple[int, int], optional
        Expected range for space group numbers, by default (1, 230)
    distance_range : Tuple[float, float], optional
        Expected range for pairwise distances between sites, by default (0.0, 25.0)
    max_sites : int, optional
        Maximum number of sites to accomodate in encoding, by default 52
    save_dir : Union[str, 'PathLike[str]']
        Directory to save PNG files via :func:``xtal2png``,
        by default path.join("data", "interim")
    symprec : Union[float, Tuple[float, float]], optional
        The symmetry precision to use when decoding `pymatgen` structures via
        ``func:pymatgen.symmetry.analyzer.SpaceGroupAnalyzer.get_refined_structure``. If
        specified as a tuple, then ``symprec[0]`` applies to encoding and ``symprec[1]``
        applies to decoding. By default 0.1.
    angle_tolerance : Union[float, int, Tuple[float, float], Tuple[int, int]], optional
        The angle tolerance (degrees) to use when decoding `pymatgen` structures via
        ``func:pymatgen.symmetry.analyzer.SpaceGroupAnalyzer.get_refined_structure``. If
        specified as a tuple, then ``angle_tolerance[0]`` applies to encoding and
        ``angle_tolerance[1]`` applies to decoding. By default 5.0.
    encode_as_primitive : bool, optional
        Encode structures as symmetrized, primitive structures. Uses ``symprec`` if
        ``symprec`` is of type float, else uses ``symprec[0]`` if ``symprec`` is of type
        tuple. Same applies for ``angle_tolerance``. By default True
    decode_as_primitive : bool, optional
        Decode structures as symmetrized, primitive structures. Uses ``symprec`` if
        ``symprec`` is of type float, else uses ``symprec[1]`` if ``symprec`` is of type
        tuple. Same applies for ``angle_tolerance``. By default True
    relax_on_decode: bool, optional
        Use m3gnet to relax the decoded crystal structures.
    verbose: bool, optional
        Whether to print verbose debugging information or not.

    Examples
    --------
    >>> xc = XtalConverter()

    >>> xc = XtalConverter(atom_range=(0, 83)) # assumes no radioactive elements in data
    """

    def __init__(
        self,
        atom_range: Tuple[int, int] = (0, 117),
        frac_range: Tuple[float, float] = (0.0, 1.0),
        a_range: Tuple[float, float] = (2.0, 15.3),
        b_range: Tuple[float, float] = (2.0, 15.0),
        c_range: Tuple[float, float] = (2.0, 36.0),
        angles_range: Tuple[float, float] = (0.0, 180.0),
        volume_range: Tuple[float, float] = (0.0, 1500.0),
        space_group_range: Tuple[int, int] = (1, 230),
        distance_range: Tuple[float, float] = (0.0, 18.0),
        max_sites: int = 52,
        save_dir: Union[str, "PathLike[str]"] = path.join("data", "preprocessed"),
        symprec: Union[float, Tuple[float, float]] = 0.1,
        angle_tolerance: Union[float, int, Tuple[float, float], Tuple[int, int]] = 5.0,
        encode_as_primitive: bool = False,
        decode_as_primitive: bool = False,
        relax_on_decode: bool = True,
        verbose: bool = True,
    ):
        """Instantiate an XtalConverter object with desired ranges and ``max_sites``."""
        self.atom_range = atom_range
        self.frac_range = frac_range
        self.a_range = a_range
        self.b_range = b_range
        self.c_range = c_range
        self.angles_range = angles_range
        self.volume_range = volume_range
        self.space_group_range = space_group_range
        self.distance_range = distance_range
        self.max_sites = max_sites
        self.save_dir = save_dir

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

        self.encode_as_primitive = encode_as_primitive
        self.decode_as_primitive = decode_as_primitive
        self.relax_on_decode = relax_on_decode
        self.verbose = verbose

        Path(save_dir).mkdir(exist_ok=True, parents=True)

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

        save_names, S = self.process_filepaths_or_structures(structures)

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
        for d, save_name in zip(self.data, save_names):
            img = Image.fromarray(d, mode="L")
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
        fit_quantiles=(0.00, 0.99),
        verbose=None,
    ):
        verbose = self.verbose if verbose is None else verbose

        _, S = self.process_filepaths_or_structures(structures)

        # TODO: deal with arbitrary site_properties
        atomic_numbers = []
        a = []
        b = []
        c = []
        space_group = []
        volume = []
        distance = []
        num_sites = []

        for s in tqdm(S):
            atomic_numbers.append(s.atomic_numbers)
            lattice = s.lattice
            a.append(lattice.a)
            b.append(lattice.b)
            c.append(lattice.c)
            space_group.append(s.get_space_group_info()[1])
            volume.append(lattice.volume)
            distance.append(s.distance_matrix)
            num_sites.append(len(list(s.sites)))

        if verbose:
            print("range of atomic_numbers is: ", min(a), "-", max(a))
            print("range of a is: ", min(a), "-", max(a))
            print("range of b is: ", min(b), "-", max(b))
            print("range of c is: ", min(c), "-", max(c))
            print("range of space_group is: ", min(space_group), "-", max(space_group))
            print("range of volume is: ", min(volume), "-", max(volume))
            print("range of num_sites is: ", min(num_sites), "-", max(num_sites))

        dis_min_tmp = []
        dis_max_tmp = []
        for d in tqdm(range(len(distance))):
            dis_min_tmp.append(min(distance[d][np.nonzero(distance[d])]))
            dis_max_tmp.append(max(distance[d][np.nonzero(distance[d])]))

        atoms = np.array(atomic_numbers, dtype="object")
        self.atom_range = (min(np.min(atoms)), max(np.max(atoms)))
        self.space_group_range = (np.min(space_group), np.max(space_group))

        self.num_sites = np.max(num_sites)

        df = pd.DataFrame(
            dict(
                a=a,
                b=b,
                c=c,
                volume=volume,
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
        save_names : List[str]
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
        >>> save_names, structures = process_filepaths_or_structures(structures)
        """
        save_names: List[str] = []

        first_is_structure = isinstance(structures[0], Structure)
        for i, s in enumerate(structures):
            if isinstance(s, str) or isinstance(s, PathLike):
                if first_is_structure:
                    raise ValueError(
                        f"structures should be of same datatype, either strs or pymatgen Structures. structures[0] is {type(structures[0])}, but got type {type(s)} for entry {i}"  # noqa: E501
                    )

                structures[i] = Structure.from_file(s)
                save_names.append(Path(str(s)).stem)

            elif isinstance(s, Structure):
                if not first_is_structure:
                    raise ValueError(
                        f"structures should be of same datatype, either strs or pymatgen Structures. structures[0] is {type(structures[0])}, but got type {type(s)} for entry {i}"  # noqa
                    )

                structures[i] = s
                save_names.append(construct_save_name(s))
            else:
                raise ValueError(
                    f"structures should be of type `str`, `os.PathLike` or `pymatgen.core.structure.Structure`, not {type(structures[i])} (entry {i})"  # noqa
                )

        for i, s in enumerate(structures):
            assert isinstance(
                s, Structure
            ), f"structures[{i}]: {type(s)}, expected: Structure"
            assert not isinstance(s, str) and not isinstance(s, PathLike)

        return save_names, structures  # type: ignore

    def png2xtal(
        self, images: List[Union[Image.Image, "PathLike"]], save: bool = False
    ):
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
        data_tmp = []
        for img in images:
            if isinstance(img, str):
                # load image from file
                with Image.open(img).convert("L") as im:
                    data_tmp.append(np.asarray(im))
            elif isinstance(img, Image.Image):
                data_tmp.append(np.asarray(img.convert("L")))

        data = np.stack(data_tmp, axis=0)

        S = self.arrays_to_structures(data)

        if save:
            for s in S:
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
    ):
        """Convert pymatgen Structure to scaled 3D array of crystallographic info.

        ``atomic_numbers`` and ``distance_matrix` get padded or cropped as appropriate,
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
        if isinstance(structures, Structure):
            raise ValueError("`structures` should be a list of pymatgen Structure(s)")

        # extract crystallographic information
        atomic_numbers: List[List[int]] = []
        frac_coords_tmp: List[NDArray] = []
        latt_a: List[float] = []
        latt_b: List[float] = []
        latt_c: List[float] = []
        angles: List[List[float]] = []
        volume: List[float] = []
        space_group: List[int] = []
        distance_matrix_tmp: List[NDArray[np.float64]] = []

        sym_structures = []
        for s in structures:
            spa = SpacegroupAnalyzer(
                s,
                symprec=self.encode_symprec,
                angle_tolerance=self.encode_angle_tolerance,
            )
            if self.encode_as_primitive:
                s = spa.get_primitive_standard_structure()
            else:
                s = spa.get_refined_structure()
            sym_structures.append(s)

        structures = sym_structures

        for s in structures:
            n_sites = len(s.atomic_numbers)
            if n_sites > self.max_sites:
                raise ValueError(
                    f"crystal supplied with {n_sites} sites, which is more than {self.max_sites} sites. Remove crystal or increase `max_sites`."  # noqa
                )
            atomic_numbers.append(
                np.pad(
                    list(s.atomic_numbers),
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
            volume.append(s.volume)
            space_group.append(s.get_space_group_info()[1])

            if n_sites != s.distance_matrix.shape[0]:
                raise ValueError(
                    f"len(atomic_numbers) {n_sites} and distance_matrix.shape[0] {s.distance_matrix.shape[0]} do not match"  # noqa
                )  # noqa

            # assume that distance matrix is square
            padwidth = (0, self.max_sites - n_sites)
            distance_matrix_tmp.append(np.pad(s.distance_matrix, padwidth))
            # [0:max_sites, 0:max_sites]

        frac_coords = np.stack(frac_coords_tmp)
        distance_matrix = np.stack(distance_matrix_tmp)

        # REVIEW: consider using modified pettifor scale instead of atomic numbers
        # REVIEW: consider using feature_range=atom_range or 2*atom_range
        # REVIEW: since it introduces a sort of non-linearity b.c. of rounding
        atom_scaled = rgb_scaler(atomic_numbers, data_range=self.atom_range)
        frac_scaled = rgb_scaler(frac_coords, data_range=self.frac_range)
        a_scaled = rgb_scaler(latt_a, data_range=self.a_range)
        b_scaled = rgb_scaler(latt_b, data_range=self.b_range)
        c_scaled = rgb_scaler(latt_c, data_range=self.c_range)
        angles_scaled = rgb_scaler(angles, data_range=self.angles_range)
        volume_scaled = rgb_scaler(volume, data_range=self.volume_range)
        space_group_scaled = rgb_scaler(space_group, data_range=self.space_group_range)
        # NOTE: max_distance could be added as another (repeated) value/row to infer
        # NOTE: kind of like frac_distance_matrix, not sure if would be effective
        # NOTE: Or could normalize distance_matix by cell volume
        # NOTE: and possibly include cell volume as a (repeated) value/row to infer
        # NOTE: It's possible extra info like this isn't so bad, instilling the physics
        # NOTE: but it could also just be extraneous work to predict/infer
        distance_scaled = rgb_scaler(distance_matrix, data_range=self.distance_range)

        atom_arr = np.expand_dims(atom_scaled, 2)
        frac_arr = frac_scaled
        a_arr = np.repeat(np.expand_dims(a_scaled, (1, 2)), self.max_sites, axis=1)
        b_arr = np.repeat(np.expand_dims(b_scaled, (1, 2)), self.max_sites, axis=1)
        c_arr = np.repeat(np.expand_dims(c_scaled, (1, 2)), self.max_sites, axis=1)
        angles_arr = np.repeat(np.expand_dims(angles_scaled, 1), self.max_sites, axis=1)
        volume_arr = np.repeat(
            np.expand_dims(volume_scaled, (1, 2)), self.max_sites, axis=1
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
            volume_arr,
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
            VOLUME_KEY: VOLUME_ID,
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
            np.ones_like(volume_arr) * VOLUME_ID,
            np.ones_like(space_group_arr) * SPACE_GROUP_ID,
            np.ones_like(distance_arr) * DISTANCE_ID,
        ]
        id_data = self.assemble_blocks(*id_blocks)

        return data, id_data, id_mapper

    def assemble_blocks(
        self,
        atom_arr,
        frac_arr,
        a_arr,
        b_arr,
        c_arr,
        angles_arr,
        volume_arr,
        space_group_arr,
        distance_arr,
    ):
        arrays = [
            atom_arr,
            frac_arr,
            a_arr,
            b_arr,
            c_arr,
            angles_arr,
            volume_arr,
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
                    volume_arr,
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
                volume_arr,
                space_group_arr,
            ]
        )
        horizontal_arr = np.moveaxis(horizontal_arr, 1, 2)
        left_arr = vertical_arr
        right_arr = np.block([[horizontal_arr], [distance_arr]])
        data = np.block([left_arr, right_arr])
        return data

    def disassemble_blocks(
        self, data, id_data: Optional[NDArray] = None, id_mapper: Optional[dict] = None
    ):
        if (id_data is None) is not (id_mapper is None):
            raise ValueError(
                f"id_data (type: {type(id_data)}) and id_mapper (type: {type(id_mapper)}) should either both be assigned or both be None."  # noqa
            )
        elif id_data is None and id_mapper is None:
            _, id_data, id_mapper = self.structures_to_arrays(dummy_structures)

        assert (
            id_data is not None and id_mapper is not None
        ), "id_data and id_mapper should not be None at this point"

        [a.shape for a in np.array_split(data, [12], axis=1)]

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
            volume_arr,
            space_group_arr,
        ) = avgs

        return (
            atom_arr,
            frac_arr,
            a_arr,
            b_arr,
            c_arr,
            angles_arr,
            volume_arr,
            space_group_arr,
            distance_arr,
        )

    def arrays_to_structures(
        self,
        data: np.ndarray,
        id_data: Optional[np.ndarray] = None,
        id_mapper: Optional[dict] = None,
    ):
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
        arrays = self.disassemble_blocks(data, id_data=id_data, id_mapper=id_mapper)

        (
            atom_scaled,
            frac_scaled,
            a_scaled_tmp,
            b_scaled_tmp,
            c_scaled_tmp,
            angles_scaled_tmp,
            volume_scaled_tmp,
            space_group_scaled_tmp,
            distance_scaled,
        ) = [np.squeeze(arr, axis=2) if arr.shape[2] == 1 else arr for arr in arrays]

        a_scaled = np.mean(a_scaled_tmp, axis=1, where=a_scaled_tmp != 0)
        b_scaled = np.mean(b_scaled_tmp, axis=1, where=b_scaled_tmp != 0)
        c_scaled = np.mean(c_scaled_tmp, axis=1, where=c_scaled_tmp != 0)
        angles_scaled = np.mean(angles_scaled_tmp, axis=1, where=angles_scaled_tmp != 0)

        volume_scaled = np.mean(volume_scaled_tmp, axis=1)
        space_group_scaled = np.round(np.mean(space_group_scaled_tmp, axis=1)).astype(
            int
        )

        atomic_numbers = rgb_unscaler(atom_scaled, data_range=self.atom_range)
        frac_coords = rgb_unscaler(frac_scaled, data_range=self.frac_range)
        latt_a = rgb_unscaler(a_scaled, data_range=self.a_range)
        latt_b = rgb_unscaler(b_scaled, data_range=self.b_range)
        latt_c = rgb_unscaler(c_scaled, data_range=self.c_range)
        angles = rgb_unscaler(angles_scaled, data_range=self.angles_range)

        # # volume, space_group, distance_matrix unecessary for making Structure
        volume = rgb_unscaler(volume_scaled, data_range=self.volume_range)
        space_group = rgb_unscaler(
            space_group_scaled, data_range=self.space_group_range
        )
        distance_matrix = rgb_unscaler(distance_scaled, data_range=self.distance_range)
        for dm in distance_matrix:
            np.fill_diagonal(dm, 0.0)
        # technically unused, but to avoid issue with pre-commit for now:
        volume, space_group, distance_matrix

        atomic_numbers = np.round(atomic_numbers).astype(int)

        # TODO: tweak lattice parameters to match predicted space group rules

        if self.relax_on_decode:
            if not self.verbose:
                tf.get_logger().setLevel(logging.ERROR)
            relaxer = Relaxer()  # This loads the default pre-trained model

        # build Structure-s
        S: List[Structure] = []
        for i in range(len(atomic_numbers)):
            at = atomic_numbers[i]
            fr = frac_coords[i]
            # di = distance_matrix[i]
            site_ids = np.where(at > 0)

            at = at[site_ids]
            fr = fr[site_ids]
            # di_cropped = di[site_ids[0]][:, site_ids[0]]

            a, b, c = latt_a[i], latt_b[i], latt_c[i]
            alpha, beta, gamma = angles[i]

            lattice = Lattice.from_parameters(
                a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma
            )
            structure = Structure(lattice, at, fr)

            # REVIEW: round fractional coordinates to nearest multiple?
            if self.relax_on_decode:
                relaxed_results = relaxer.relax(structure, verbose=self.verbose)
                structure = relaxed_results["final_structure"]

                # relax_results = relaxer.relax()
                # final_structure = relax_results["final_structure"]
                # final_energy = relax_results["trajectory"].energies[-1] / 2

                # print(
                #     f"Relaxed lattice parameter is
                #     {final_structure.lattice.abc[0]:.3f} Å"
                # )
                # # TODO: print the initial energy as well (assuming it's available)
                # print(f"Final energy is {final_energy.item(): .3f} eV/atom")

            spa = SpacegroupAnalyzer(
                structure,
                symprec=self.decode_symprec,
                angle_tolerance=self.decode_angle_tolerance,
            )
            if self.decode_as_primitive:
                structure = spa.get_primitive_standard_structure()
            else:
                structure = spa.get_refined_structure()

            S.append(structure)

        if self.relax_on_decode:
            # restore default https://stackoverflow.com/a/51340381/13697228
            sys.stdout = sys.__stdout__

        return S


# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def parse_args(args):
    """Parse command line parameters.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Crystal to PNG encoder/decoder.")
    parser.add_argument(
        "--version",
        action="version",
        version="xtal2png {ver}".format(ver=__version__),
    )
    parser.add_argument(
        "-p",
        "--path",
        dest="fpath",
        help="Crystallographic information file (CIF) filepath (extension must be .cif or .CIF) or path to directory containing .cif files or processed PNG filepath or path to directory containing processed .png files (extension must be .png or .PNG). Assumes CIFs if --encode flag is used. Assumes PNGs if --decode flag is used.",  # noqa: E501
        type=str,
        metavar="STRING",
    )
    parser.add_argument(
        "-s",
        "--save-dir",
        dest="save_dir",
        default=".",
        help="Directory to save processed PNG files or decoded CIFs to.",
        type=str,
        metavar="STRING",
    )
    parser.add_argument(
        "--encode",
        action="store_true",
        help="Encode CIF files as PNG images.",
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Decode PNG images as CIF files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper allowing :func:`XtalConverter()` :func:`xtal2png()` and
    :func:`png2xtal()` methods to be called with string arguments in a CLI fashion.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "example.cif"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Beginning conversion to PNG format")

    if args.encode and args.decode:
        raise ValueError("Specify --encode or --decode, not both.")

    if args.encode:
        ext = ".cif"
    elif args.decode:
        ext = ".png"
    else:
        raise ValueError("Specify at least one of --encode or --decode")

    if Path(args.fpath).suffix in [ext, ext.upper()]:
        fpaths = [args.fpath]
    elif path.isdir(args.fpath):
        fpaths = glob(path.join(args.fpath, f"*{ext}"))
        if fpaths == []:
            raise ValueError(
                f"Assuming --path input is directory to files. No files of type {ext} present in {args.fpath}"  # noqa: E501
            )
    else:
        raise ValueError(
            f"Input should be a path to a single {ext} file or a path to a directory containing {ext} file(s). Received: {args.fpath}"  # noqa: E501
        )

    xc = XtalConverter(save_dir=args.save_dir)
    if args.encode:
        xc.xtal2png(fpaths, save=True)
    elif args.decode:
        xc.png2xtal(fpaths, save=True)

    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m xtal2png.core example.cif
    #
    run()
