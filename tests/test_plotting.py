"""Test the plotting module"""

from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from pyscal import GasOil, GasWater, PyscalList, WaterOil, WaterOilGas, plotting


def test_get_satnum_from_tag():
    """Check that the SATNUM number can be retrieved from the model tag"""
    # Several PyscalLists of different model types to be checked
    pyscal_lists = [
        PyscalList(
            [
                WaterOil(tag="SATNUM 1"),
                WaterOil(tag="SATNUM 2"),
                WaterOil(tag="SATNUM 3"),
            ]
        ),
        PyscalList(
            [
                GasWater(tag="SATNUM 1"),
                GasWater(tag="SATNUM 2"),
                GasWater(tag="SATNUM 3"),
            ]
        ),
        PyscalList(
            [GasOil(tag="SATNUM 1"), GasOil(tag="SATNUM 2"), GasOil(tag="SATNUM 3")]
        ),
        PyscalList(
            [
                WaterOilGas(tag="SATNUM 1"),
                WaterOilGas(tag="SATNUM 2"),
                WaterOilGas(tag="SATNUM 3"),
            ]
        ),
    ]

    # Test that an integer is returned from this function
    for pyscal_list in pyscal_lists:
        for model in pyscal_list.pyscal_list:
            satnum = plotting.get_satnum_from_tag(model.tag)
            assert isinstance(satnum, int)

    # Test that this function doesn't work if SATNUM is represented as a float
    # in the model.tag string
    with pytest.raises(ValueError):
        satnum = plotting.get_satnum_from_tag(WaterOil(tag="SATNUM 3.0").tag)


def test_plotter():
    """Check if an Exception is raised if a model type is not included. This is done
    to check that all models have been implemented in the plotting module."""

    class DummyPyscalList:
        """
        Can't use the actual PyscalList, as this will raise its own exception
        (DummyModel is not a pyscal object), so a dummy PyscalList is used

        #If the PyscalList.pyscal_list instance variable name changes, this
        will still pass..."""

        def __init__(self, models: list) -> None:
            self.pyscal_list = models

    class DummyModel:
        """Dummy model"""

        def __init__(self, tag: str) -> None:
            self.tag = tag

    dummy_model_list = [
        DummyModel("SATNUM 1"),
        DummyModel("SATNUM 2"),
        DummyModel("SATNUM 3"),
    ]

    dummy_pyscal_list = DummyPyscalList(dummy_model_list)

    with pytest.raises(Exception):
        plotting.plotter(dummy_pyscal_list)


def test_pyscal_list_attr():
    """Check that the PyscalList class has an pyscal_list instance variable.
    This is accessed by the plotting module to loop through models to plot."""
    assert (
        hasattr(PyscalList(), "pyscal_list") is True
    ), "The PyscalList object should have a pyscal_list instance variable.\
        This is accessed by the plotting module."


def test_plot_relperm():
    """Test that a matplotlib.pyplot Figure instance is returned"""
    wateroil = WaterOil(swl=0.1, h=0.1)
    wateroil.add_corey_water()
    wateroil.add_corey_oil()

    kwargs = {"semilog": False, "pc": False}
    config = plotting.get_plot_config_options("WaterOil", **kwargs)
    fig = plotting.plot_relperm(wateroil.table, 1, config, **kwargs)

    assert isinstance(
        fig, plt.Figure
    ), "Type of returned object is not a matplotlib.pyplot Figure"


def test_plot_pc():
    """Test that a matplotlib.pyplot Figure instance is returned"""
    wateroil = WaterOil(swl=0.1, h=0.1)
    wateroil.add_corey_water()
    wateroil.add_corey_oil()
    wateroil.add_simple_J()

    kwargs = {"semilog": False, "pc": True}
    config = plotting.get_plot_config_options("WaterOil", **kwargs)
    fig = plotting.plot_pc(wateroil.table, 1, **config)

    assert isinstance(
        fig, plt.Figure
    ), "Type of returned object is not a matplotlib.pyplot Figure"


def test_wog_plotter(tmpdir):
    """Test that relative permeability figures are created by the plotter function"""
    wateroil = WaterOil(swl=0.1, h=0.1, tag="SATNUM 1")
    wateroil.add_corey_water()
    wateroil.add_corey_oil()
    wateroil.add_simple_J()

    gasoil = GasOil(swl=0.1, h=0.1, tag="SATNUM 1")
    gasoil.add_corey_gas()
    gasoil.add_corey_oil()

    wateroilgas = WaterOilGas()
    wateroilgas.wateroil = wateroil
    wateroilgas.gasoil = gasoil

    # Testing both rel perm and Pc plots are created
    kwargs = {"semilog": False, "pc": True, "outdir": tmpdir}
    kr_ow_plot_name = "krw_krow_SATNUM_1.png"
    kr_og_plot_name = "krw_krow_SATNUM_1.png"
    pc_ow_plot_name = "pcow_SATNUM_1.png"

    plotting.wog_plotter(wateroilgas, **kwargs)

    assert Path.exists(Path(tmpdir).joinpath(kr_ow_plot_name)), "Plot not created"
    assert Path.exists(Path(tmpdir).joinpath(kr_og_plot_name)), "Plot not created"
    assert Path.exists(Path(tmpdir).joinpath(pc_ow_plot_name)), "Plot not created"


def test_wo_plotter(tmpdir):
    """Test that relative permeability figures are created by the plotter function"""
    wateroil = WaterOil(swl=0.1, h=0.1, tag="SATNUM 1")
    wateroil.add_corey_water()
    wateroil.add_corey_oil()
    wateroil.add_simple_J()

    # Testing both rel perm and Pc plots are created
    kwargs = {"semilog": False, "pc": True, "outdir": tmpdir}
    kr_plot_name = "krw_krow_SATNUM_1.png"
    pc_plot_name = "pcow_SATNUM_1.png"

    plotting.wo_plotter(wateroil, **kwargs)

    assert Path.exists(Path(tmpdir).joinpath(kr_plot_name)), "Plot not created"
    assert Path.exists(Path(tmpdir).joinpath(pc_plot_name)), "Plot not created"


def test_wo_plotter_relperm_only(tmpdir):
    """Test that relative permeability figures are created by the plotter function"""
    wateroil = WaterOil(swl=0.1, h=0.1, tag="SATNUM 1")
    wateroil.add_corey_water()
    wateroil.add_corey_oil()
    wateroil.add_simple_J()

    # Testing only relperm plots are created
    kwargs = {"semilog": False, "pc": False, "outdir": tmpdir}
    kr_plot_name = "krw_krow_SATNUM_1.png"
    pc_plot_name = "pcow_SATNUM_1.png"

    plotting.wo_plotter(wateroil, **kwargs)

    assert Path.exists(Path(tmpdir).joinpath(kr_plot_name)), "Plot not created"
    assert not Path.exists(
        Path(tmpdir).joinpath(pc_plot_name)
    ), "Plot created when it shouldn't have been"


def test_go_plotter(tmpdir):
    """Test that relative permeability figures are created by the plotter function"""
    gasoil = GasOil(swl=0.1, h=0.1, tag="SATNUM 1")
    gasoil.add_corey_gas()
    gasoil.add_corey_oil()

    # Add Pc manually as there are currently no supporting functions for adding
    # Pc to GasOil instances
    gasoil.table["PC"] = 1 * len(gasoil.table)

    # Testing both rel perm and Pc plots are created
    kwargs = {"semilog": False, "pc": True, "outdir": tmpdir}
    kr_plot_name = "krg_krog_SATNUM_1.png"
    pc_plot_name = "pcog_SATNUM_1.png"

    plotting.go_plotter(gasoil, **kwargs)

    assert Path.exists(Path(tmpdir).joinpath(kr_plot_name)), "Plot not created"
    assert Path.exists(Path(tmpdir).joinpath(pc_plot_name)), "Plot not created"


def test_gw_plotter(tmpdir):
    """Test that relative permeability figures are created by the plotter function"""
    gaswater = GasWater(swl=0.1, h=0.1, tag="SATNUM 1")
    gaswater.add_corey_water()
    gaswater.add_corey_gas()
    gaswater.add_simple_J()

    # Testing both rel perm and Pc plots are created
    kwargs = {"semilog": False, "pc": True, "outdir": tmpdir}
    kr_plot_name = "krg_krw_SATNUM_1.png"
    pc_plot_name = "pcgw_SATNUM_1.png"

    plotting.gw_plotter(gaswater, **kwargs)

    assert Path.exists(Path(tmpdir).joinpath(kr_plot_name)), "Plot not created"
    assert Path.exists(Path(tmpdir).joinpath(pc_plot_name)), "Plot not created"


def test_save_figure(tmpdir):
    """Test that figure is saved"""
    fig = plt.Figure()

    config = {"curves": "dummy", "suffix": ""}
    fig_name = "dummy_SATNUM_1.png"
    plotting.save_figure(fig, 1, config, plot_type="relperm", outdir=tmpdir)

    assert Path.exists(Path(tmpdir).joinpath(fig_name)), "Figure not saved"
