import pytest

from pyscal import GasOil, GasWater, PyscalList, WaterOil, WaterOilGas, plotting


def test_get_satnum_from_tag():
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
    # Check if Exception is raised if a model type is not included. This is done
    # to check that all models have been implemented in the plotting module.

    class DummyPyscalList:
        # Can't use the actual PyscalList, as this will raise its own exception
        # (DummyModel is not a pyscal object), so a dummy PyscalList is used

        # If the PyscalList.pyscal_list instance variable name changes, this
        # will still pass...
        def __init__(self, models: list) -> None:
            self.pyscal_list = models

    class DummyModel:
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
    # Check that the PyscalList class has an pyscal_list instance variable.
    # This is access by the plotting module to loop through models to plot.
    assert (
        hasattr(PyscalList(), "pyscal_list") is True
    ), "The PyscalList object should have a pyscal_list instance variable.\
        This is accessed by the plotting module."
