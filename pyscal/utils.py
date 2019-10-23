"""Utility function for pyscal
"""
import pandas as pd

from pyscal.constants import SWINTEGERS


def interpolator(
    tableobject, curve1, curve2, parameter, sat="sw", kr1="krw", kr2="krow", pc="pc"
):
    """Interpolates between two curves using one parameter between 0 and
    1, does not care if it is water-oil or gas-oil.  First
    argument is the wateroil or gasoil object it is to populate.

    0 will return curve1
    1 will return curve2

    No return value, but modifies the object pointed to by first handle

    """

    curve1.table.rename(columns={kr1: kr1 + "_1"}, inplace=True)
    curve2.table.rename(columns={kr1: kr1 + "_2"}, inplace=True)
    curve1.table.rename(columns={kr2: kr2 + "_1"}, inplace=True)
    curve2.table.rename(columns={kr2: kr2 + "_2"}, inplace=True)
    curve1.table.rename(columns={pc: pc + "_1"}, inplace=True)
    curve2.table.rename(columns={pc: pc + "_2"}, inplace=True)

    # Result data container:
    satresult = pd.DataFrame(data=tableobject.table[sat], columns=[sat])

    # Merge swresult with curve1 and curve2, and interpolate all
    # columns in sw:
    intdf = (
        pd.concat([curve1.table, curve2.table, satresult], sort=True)
        .set_index(sat)
        .sort_index()
        .interpolate(method="slinear")
        .fillna(method="bfill")
        .fillna(method="ffill")
    )

    # Normalized saturations does not make sense for the
    # interpolant, remove:
    for col in ["swn", "son", "swnpc", "H", "J"]:
        if col in intdf.columns:
            del intdf[col]

    intdf[kr1] = intdf[kr1 + "_1"] * (1 - parameter) + intdf[kr1 + "_2"] * parameter
    intdf[kr2] = intdf[kr2 + "_1"] * (1 - parameter) + intdf[kr2 + "_2"] * parameter
    if pc + "_1" in curve1.table.columns and pc + "_2" in curve2.table.columns:
        intdf[pc] = intdf[pc + "_1"] * (1 - parameter) + intdf[pc + "_2"] * parameter
    else:
        intdf[pc] = 0

    # Slice out the resulting sw values and columns. Slicing on
    # floating point indices is not robust so we need to slice on an
    # integer version of the sw column
    tableobject.table["swint"] = list(
        map(int, list(map(round, tableobject.table[sat] * SWINTEGERS)))
    )
    intdf["swint"] = list(map(int, list(map(round, intdf.index.values * SWINTEGERS))))
    intdf = intdf.reset_index()
    intdf.drop_duplicates("swint", inplace=True)
    intdf.set_index("swint", inplace=True)
    intdf = intdf.loc[tableobject.table["swint"].values]
    intdf = intdf[[sat, kr1, kr2, pc]].reset_index()

    # intdf['swint'] = (intdf['sw'] * SWINTEGERS).astype(int)
    # intdf.drop_duplicates('swint', inplace=True)

    # Populate the WaterOil object
    tableobject.table[kr1] = intdf[kr1]
    tableobject.table[kr2] = intdf[kr2]
    tableobject.table[pc] = intdf[pc]
    tableobject.table.fillna(method="ffill", inplace=True)
    return
