"""Test encoding functionality using default kwargs."""


import numpy as np
import plotly.express as px
import pytest

from xtal2png.core import XtalConverter
from xtal2png.utils.data import example_structures
from xtal2png.utils.plotting import plot_and_save


def test_structures_to_arrays():
    xc = XtalConverter(relax_on_decode=False)
    data, _, _ = xc.structures_to_arrays(example_structures)
    return data


def test_structures_to_arrays_single():
    xc = XtalConverter(relax_on_decode=False)
    data, _, _ = xc.structures_to_arrays([example_structures[0]])
    return data


def test_xtal2png():
    xc = XtalConverter(relax_on_decode=False)
    imgs = xc.xtal2png(example_structures, show=False, save=True)
    return imgs


def test_xtal2png_single():
    xc = XtalConverter(relax_on_decode=False)
    imgs = xc.xtal2png([example_structures[0]], show=False, save=True)
    return imgs


def test_xtal2png_three_channels():
    xc = XtalConverter(relax_on_decode=False, channels=3)
    imgs = xc.xtal2png(example_structures, show=False, save=False)
    return imgs


def test_plot_and_save():
    df = px.data.tips()
    fig = px.histogram(df, x="day")
    plot_and_save("reports/figures/tmp", fig, mpl_kwargs={})


def test_structures_to_arrays_zero_one():
    xc = XtalConverter(relax_on_decode=False)
    data, _, _ = xc.structures_to_arrays(example_structures, rgb_scaling=False)

    if np.min(data) < 0.0:
        raise ValueError(
            f"minimum is less than 0 when rgb_output=False: {np.min(data)}"
        )
    if np.max(data) > 1.0:
        raise ValueError(
            f"maximum is greater than 1 when rgb_output=False: {np.max(data)}"
        )
    return data


def test_disordered_fail(get_disordered_structure):
    xc = XtalConverter(relax_on_decode=False)
    with pytest.raises(ValueError):
        xc.structures_to_arrays([get_disordered_structure])


if __name__ == "__main__":
    test_xtal2png_three_channels()
    imgs = test_xtal2png()
    imgs = test_xtal2png_single()
    test_xtal2png_single()
    test_structures_to_arrays_single()
    test_disordered_fail()
