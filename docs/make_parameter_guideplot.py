"""
This file is to be run manually on demand.

If updates to this file, run it and commit
the modified png files.
"""

import matplotlib.pyplot as plt

from pyscal import WaterOil, GasOil, GasWater


def main():
    """Run this to produce the PNG images for the RST docs"""
    make_gaswater_plot(show=False)
    plt.savefig("images/gaswater-endpoints.png", dpi=200)
    make_gasoil_plot(show=False)
    plt.savefig("images/gasoil-endpoints.png", dpi=200)
    make_wateroil_plot(show=False)
    plt.savefig("images/wateroil-endpoints.png", dpi=200)


def make_gasoil_plot(show=True, krgendanchor="sorg"):
    """Make a plot explaining the inputs to a GasOil object"""
    plt.xkcd()
    _, axes = plt.subplots()
    swl = 0.1
    sgcr = 0.2
    sorg = 0.2
    krgend = 0.7
    krgmax = 0.75
    kroend = 0.85
    gasoil = GasOil(sgcr=sgcr, sorg=sorg, swl=swl, krgendanchor=krgendanchor)
    gasoil2 = GasOil(sgcr=sgcr, sorg=sorg, swl=swl, krgendanchor=None)
    gasoil.add_corey_gas(ng=3, krgend=krgend, krgmax=krgmax)
    gasoil2.add_corey_gas(ng=3, krgend=krgend)
    gasoil.add_corey_oil(nog=3, kroend=kroend)
    gasoil2.add_corey_oil(nog=3, kroend=kroend)
    gasoil2.table.plot(
        ax=axes, x="sg", y="krg", c="pink", alpha=0.7, label="KRG*", linewidth=2
    )
    gasoil.table.plot(
        ax=axes, x="sg", y="krg", c="red", alpha=1, label="KRG", linewidth=2
    )
    gasoil.table.plot(
        ax=axes, x="sg", y="krog", c="green", alpha=1, label="KROG", linewidth=2
    )
    plt.ylim([-0.02, 1])
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    axes.annotate(
        "KROEND",
        xy=(0, kroend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(0.1, kroend - 0.2),
    )
    axes.annotate(
        "KRGEND",
        xy=(1 - sorg - swl, krgend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(1 - sorg - swl + 0.01, krgmax - 0.2),
    )
    # Overwrite with same text at identical spots for
    # two arrows:
    axes.annotate(
        "KRGEND",
        xy=(1 - swl, krgend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(1 - sorg - swl + 0.01, krgmax - 0.2),
    )
    axes.annotate(
        "KRGMAX",
        xy=(1 - swl, krgmax),
        arrowprops=dict(arrowstyle="->"),
        xytext=(1 - swl - 0.2, krgmax + 0.1),
    )
    axes.annotate(
        "SGCR",
        xy=(sgcr, 0),
        arrowprops=dict(arrowstyle="->"),
        xytext=(sgcr - 0.05, 0 + 0.14),
    )
    plt.xlabel("SG", labelpad=-10)
    axes.annotate(
        "",
        xy=(1 - sorg - swl, 0.02),
        xytext=(1 - swl, 0.02),
        arrowprops=dict(arrowstyle="<->"),
    )
    axes.text(1 - sorg - swl + 0.04, 0.04, "SORG")
    axes.annotate(
        "", xy=(1 - swl, 0.02), xytext=(1, 0.02), arrowprops=dict(arrowstyle="<->")
    )
    axes.text(1 - swl + 0.04, 0.04, "SWL")
    axes.legend(loc="upper center")
    if show:
        plt.show()


def make_wateroil_plot(show=True):
    """Make a plot explaining the inputs to a WaterOil object"""
    plt.xkcd()
    _, axes = plt.subplots()
    swirr = 0.05
    swl = 0.1
    swcr = 0.2
    sorw = 0.2
    krwend = 0.7
    krwmax = 0.75
    kroend = 0.85
    wateroil = WaterOil(swirr=swirr, swl=swl, swcr=swcr, sorw=sorw)
    wateroil.add_corey_water(nw=2, krwend=krwend, krwmax=krwmax)
    wateroil.add_corey_oil(now=2, kroend=kroend)
    wateroil.table.plot(
        ax=axes, x="sw", y="krw", c="blue", alpha=1, label="KRW", linewidth=2
    )
    wateroil.table.plot(
        ax=axes, x="sw", y="krow", c="green", alpha=1, label="KROW", linewidth=2
    )
    plt.ylim([-0.02, 1])
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    axes.annotate(
        "KROEND",
        xy=(swl, kroend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(swl - 0.1, kroend - 0.3),
    )
    axes.annotate(
        "KRWEND",
        xy=(1 - sorw, krwend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(1 - sorw - 0.09, krwmax - 0.2),
    )
    axes.annotate(
        "KRWMAX",
        xy=(1, krwmax),
        arrowprops=dict(arrowstyle="->"),
        xytext=(1 - 0.1, krwmax - 0.3),
    )
    axes.annotate(
        "SWIRR",
        xy=(swirr, 0),
        arrowprops=dict(arrowstyle="->"),
        xytext=(swirr - 0.12, 0 + 0.1),
    )
    axes.annotate(
        "SWL",
        xy=(swl, 0),
        arrowprops=dict(arrowstyle="->"),
        xytext=(swl - 0.05, 0 + 0.14),
    )
    axes.annotate(
        "SWCR",
        xy=(swcr, 0),
        arrowprops=dict(arrowstyle="->"),
        xytext=(swcr - 0.06, 0 + 0.21),
    )
    plt.xlabel("SW", labelpad=-10)
    axes.annotate(
        "", xy=(1 - sorw, 0.02), xytext=(1, 0.02), arrowprops=dict(arrowstyle="<->")
    )
    axes.text(1 - sorw + 0.04, 0.04, "SORW")
    if show:
        plt.show()


def make_gaswater_plot(show=True):
    """Make a plot explaining the inputs to a WaterOil object"""
    plt.xkcd()
    _, axes = plt.subplots()
    swirr = 0.05
    swl = 0.1
    swcr = 0.2
    sgcr = 0.4
    sgrw = 0.2
    krwend = 0.65
    krwmax = 0.75
    krgend = 0.85
    gaswater = GasWater(swirr=swirr, swl=swl, swcr=swcr, sgrw=sgrw, sgcr=sgcr, h=0.001)
    gaswater.add_corey_water(nw=2, krwend=krwend, krwmax=krwmax)
    gaswater.add_corey_gas(ng=2, krgend=krgend)
    gaswater.wateroil.table.plot(
        ax=axes, x="sw", y="krw", c="blue", alpha=1, label="KRW", linewidth=2
    )
    gaswater.gasoil.table.plot(
        ax=axes, x="sl", y="krg", c="red", alpha=1, label="KRG", linewidth=2
    )
    plt.ylim([-0.02, 1])
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    axes.annotate(
        "KRGEND",
        xy=(swl, krgend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(swl - 0.1, krgend - 0.3),
    )
    axes.annotate(
        "KRWEND",
        xy=(1 - sgrw, krwend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(1 - sgrw - 0.23, krwmax - 0.2),
    )
    axes.annotate(
        "KRWMAX",
        xy=(1, krwmax),
        arrowprops=dict(arrowstyle="->"),
        xytext=(1 - 0.1, krwmax - 0.3),
    )
    axes.annotate(
        "SWIRR",
        xy=(swirr, 0),
        arrowprops=dict(arrowstyle="->"),
        xytext=(swirr - 0.12, 0 + 0.1),
    )
    axes.annotate(
        "SWL",
        xy=(swl, 0),
        arrowprops=dict(arrowstyle="->"),
        xytext=(swl - 0.05, 0 + 0.14),
    )
    axes.annotate(
        "SWCR",
        xy=(swcr, 0),
        arrowprops=dict(arrowstyle="->"),
        xytext=(swcr - 0.06, 0 + 0.21),
    )
    plt.xlabel("SW", labelpad=-10)

    axes.text(1 - sgrw + 0.04, 0.04, "SGRW")
    axes.annotate(
        "", xy=(1 - sgrw, 0.02), xytext=(1, 0.02), arrowprops=dict(arrowstyle="<->")
    )

    axes.text(1 - sgcr + 0.04, 0.14, "SGCR")
    axes.annotate(
        "", xy=(1 - sgcr, 0.12), xytext=(1, 0.12), arrowprops=dict(arrowstyle="<->")
    )

    if show:
        plt.show()


if __name__ == "__main__":
    main()
