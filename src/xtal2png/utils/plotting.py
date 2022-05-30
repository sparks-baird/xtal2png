from typing import Union

import plotly.graph_objs as go


def matplotlibify(
    fig: go.Figure,
    size: int = 24,
    width_inches: Union[float, int] = 3.5,
    height_inches: Union[float, int] = 3.5,
    dpi: int = 142,
    return_scale: bool = False,
) -> go.Figure:
    """Make plotly figures look more like matplotlib for academic publishing.

    modified from: https://medium.com/swlh/fa56ddd97539

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to be matplotlibified
    size : int, optional
        Font size for layout and axes, by default 24
    width_inches : Union[float, int], optional
        Width of matplotlib figure in inches, by default 3.5
    height_inches : Union[float, int], optional
        Height of matplotlib figure in Inches, by default 3.5
    dpi : int, optional
        Dots per inch (resolution) of matplotlib figure, by default 142. Leave as
        default unless you're willing to verify nothing strange happens with the output.
    return_scale : bool, optional
        If true, then return `scale` which is a quantity that helps with creating a
        high-resolution image at the specified absolute width and height in inches.
        More specifically:
        >>> width_default_px = fig.layout.width
        >>> targ_dpi = 300
        >>> scale = width_inches / (width_default_px / dpi) * (targ_dpi / dpi)
        Feel free to ignore this parameter.

    Returns
    -------
    fig : go.Figure
        The matplotlibified plotly figure.

    Examples
    --------
    >>> import plotly.express as px
    >>> df = px.data.tips()
    >>> fig = px.histogram(df, x="day")
    >>> fig.show()
    >>> fig = matplotlibify(fig, size=24, width_inches=3.5, height_inches=3.5, dpi=142)
    >>> fig.show()

    Note the difference between
    https://user-images.githubusercontent.com/45469701/171044741-35591a2c-dede-4df1-ae47-597bbfdb89cf.png
    and
    https://user-images.githubusercontent.com/45469701/171044746-84a0deb0-1e15-40bf-a459-a5a7d3425b20.png,
    which are both static exports of interactive plotly figures.
    """
    font_dict = dict(family="Arial", size=size, color="black")

    # app = QApplication(sys.argv)
    # screen = app.screens()[0]
    # dpi = screen.physicalDotsPerInch()
    # app.quit()

    fig.update_layout(
        font=font_dict,
        plot_bgcolor="white",
        width=width_inches * dpi,
        height=height_inches * dpi,
        margin=dict(r=40, t=20, b=10),
    )

    fig.update_yaxes(
        showline=True,  # add line at x=0
        linecolor="black",  # line color
        linewidth=2.4,  # line size
        ticks="inside",  # ticks outside axis
        tickfont=font_dict,  # tick label font
        mirror="allticks",  # add ticks to top/right axes
        tickwidth=2.4,  # tick width
        tickcolor="black",  # tick color
    )

    fig.update_xaxes(
        showline=True,
        showticklabels=True,
        linecolor="black",
        linewidth=2.4,
        ticks="inside",
        tickfont=font_dict,
        mirror="allticks",
        tickwidth=2.4,
        tickcolor="black",
    )
    fig.update(layout_coloraxis_showscale=False)

    width_default_px = fig.layout.width
    targ_dpi = 300
    scale = width_inches / (width_default_px / dpi) * (targ_dpi / dpi)

    if return_scale:
        return fig, scale
    else:
        return fig
