"""Test customization via XtalConverter kwargs such as 3-channels or max_sites."""


import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from xtal2png.core import XtalConverter
from xtal2png.utils.data import (
    assert_structures_approximate_match,
    dummy_structures,
    example_structures,
    rgb_unscaler,
)


def test_png2xtal_three_channels():
    xc = XtalConverter(relax_on_decode=False, channels=3)
    imgs = xc.xtal2png(example_structures, show=False, save=False)
    img_shape = np.asarray(imgs[0]).shape
    if img_shape != (64, 64, 3):
        raise ValueError(f"Expected image shape: (3, 64, 64), received: {img_shape}")
    decoded_structures = xc.png2xtal(imgs)
    assert_structures_approximate_match(
        example_structures, decoded_structures, tol_multiplier=2.0
    )


def test_primitive_encoding():
    xc = XtalConverter(
        symprec=0.1,
        angle_tolerance=5.0,
        encode_cell_type="primitive_standard",
        decode_cell_type=None,
        relax_on_decode=False,
    )
    input_structures = [
        SpacegroupAnalyzer(
            s, symprec=0.1, angle_tolerance=5.0
        ).get_conventional_standard_structure()
        for s in example_structures
    ]
    data, id_data, id_mapper = xc.structures_to_arrays(input_structures)
    decoded_structures = xc.arrays_to_structures(data, id_data, id_mapper)
    assert_structures_approximate_match(
        example_structures, decoded_structures, tol_multiplier=2.0
    )
    return decoded_structures


def test_primitive_decoding():
    xc = XtalConverter(
        symprec=0.1,
        angle_tolerance=5.0,
        encode_cell_type=None,
        decode_cell_type="primitive_standard",
        relax_on_decode=False,
    )
    input_structures = [
        SpacegroupAnalyzer(
            s, symprec=0.1, angle_tolerance=5.0
        ).get_conventional_standard_structure()
        for s in example_structures
    ]
    data, id_data, id_mapper = xc.structures_to_arrays(input_structures)
    decoded_structures = xc.arrays_to_structures(data, id_data, id_mapper)
    # decoded has to be conventional too for compatibility with `get_s1_like_s2`
    decoded_structures = [
        SpacegroupAnalyzer(
            s, symprec=0.1, angle_tolerance=5.0
        ).get_conventional_standard_structure()
        for s in decoded_structures
    ]
    assert_structures_approximate_match(
        example_structures, decoded_structures, tol_multiplier=2.0
    )
    return decoded_structures


def test_relax_on_decode():
    xc = XtalConverter(relax_on_decode=True)
    imgs = xc.xtal2png(example_structures, show=False, save=False)
    decoded_structures = xc.png2xtal(imgs)
    assert_structures_approximate_match(
        example_structures, decoded_structures, tol_multiplier=4.0
    )
    return decoded_structures


def test_max_sites():
    max_sites = 10
    xc = XtalConverter(max_sites=max_sites)
    imgs = xc.xtal2png(dummy_structures)
    width, height = imgs[0].size
    if width != max_sites + 12:
        raise ValueError(
            f"Image width is not correct. Got {width}, expected {max_sites + 12}"
        )
    if height != max_sites + 12:
        raise ValueError(
            f"Image height is not correct. Got {height}, expected {max_sites + 12}"
        )
    decoded_structures = xc.png2xtal(imgs)
    assert_structures_approximate_match(dummy_structures, decoded_structures)
    return decoded_structures


def test_distance_mask():
    xc = XtalConverter(mask_types=["distance"])
    imgs = xc.xtal2png(example_structures)
    if not np.all(xc.data[xc.id_data == xc.id_mapper["distance"]] == 0):
        raise ValueError("Distance mask not applied correctly (id_mapper)")

    if not np.all(xc.data[:, :, 12:, 12:] == 0):
        raise ValueError("Distance mask not applied correctly (hardcoded)")

    return imgs


def test_lower_tri_mask():
    xc = XtalConverter(mask_types=["lower_tri"])
    imgs = xc.xtal2png(example_structures)
    if not np.all(xc.data[np.tril(xc.data[0, 0])] == 0):
        raise ValueError("Lower triangle mask not applied correctly")

    return imgs


def test_mask_error():
    xc = XtalConverter(mask_types=["num_sites"])
    imgs = xc.xtal2png(example_structures)

    decoded_structures = xc.png2xtal(imgs)

    for s in decoded_structures:
        if s.num_sites > 0:
            raise ValueError("Num sites mask should have wiped out atomic sites.")


def test_png2xtal_element_coder():
    xc = XtalConverter(element_encoding="mod_pettifor", relax_on_decode=False)
    imgs = xc.xtal2png(example_structures)
    decoded_structures = xc.png2xtal(imgs)
    assert_structures_approximate_match(
        example_structures, decoded_structures, tol_multiplier=2.0
    )


def test_num_sites_mask():
    xc = XtalConverter(relax_on_decode=False)
    data, id_data, id_mapper = xc.structures_to_arrays(example_structures)
    decoded_structures = xc.arrays_to_structures(data, id_data, id_mapper)

    num_sites = [d[idd == id_mapper["num_sites"]] for d, idd in zip(data, id_data)]
    num_sites = [ns[np.where(ns > 0)] for ns in num_sites]
    num_sites = rgb_unscaler(num_sites, xc.num_sites_range)
    num_sites = [np.round(np.mean(ns)).astype(int) for ns in num_sites]

    for i, (ns, s, ds) in enumerate(
        zip(num_sites, example_structures, decoded_structures)
    ):
        if ns != s.num_sites:
            raise ValueError(
                f"Num sites not encoded correctly for entry {i}. Received {ns}. Expected: {s.num_sites}"  # noqa: E501
            )
        if ds.num_sites != s.num_sites:
            raise ValueError(
                f"Num sites not decoded correctly for entry {i}. Received {ds.num_sites}. Expected: {s.num_sites}"  # noqa: E501
            )

    for ns, d in zip(num_sites, data):
        mask = np.zeros(d.shape, dtype=bool)
        mx = mask.shape[1]
        ms = xc.max_sites
        filler_dim = mx - ms  # i.e. 12
        # apply mask to bottom and RHS blocks
        mask[:, :, filler_dim + ns :] = True
        mask[:, filler_dim + ns :, :] = True
        if not np.all(d[mask] == 0):
            raise ValueError(
                f"Num sites mask not applied correctly. Received: {d}. Expected all zeros for bottom and RHS (last {ns} entries of each)."  # noqa: E501
            )
