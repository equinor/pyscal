"""Generates plots for usage in documentation"""
from pathlib import Path

import matplotlib.pyplot as plt

from pyscal import GasOil, GasWater, WaterOil

IMG_DIR = Path(__file__).absolute().parent / "images"


def main():
    """Run this to produce the PNG images for the RST docs"""

    for plotname in [
        # Three first plots are there to show all possible
        # object parameterizations:
        "gasoil-endpoints",
        "gaswater-endpoints",
        "wateroil-endpoints",
        # Remaining plots are for dedicated flowing processes:
        "wateroil-idc2",
        "gasoil-cdi2",
        "gaswater-icd2",
        "gasoil-cid2",
        "gaswater-dci1",
        "gaswater-co2-icd2",
        "gaswater-paleogas-dci3",
        "wateroil-paleooil-idc1",
    ]:
        assert "_" not in plotname, "Keep names consistent please.."
        eval("make_" + plotname.replace("-", "_") + "(show=False)")
        plt.tight_layout()
        plt.savefig(IMG_DIR / (plotname + ".png"), dpi=200)


def make_gasoil_endpoints(show=True, krgendanchor="sorg"):
    """Make a plot explaining the inputs to a GasOil object"""
    plt.xkcd()
    _, axes = plt.subplots()
    swl = 0.1
    sgcr = 0.1
    sgro = 0.1
    sorg = 0.2
    krgend = 0.7
    krgmax = 0.75
    kroend = 0.8
    kromax = 0.85
    gasoil = GasOil(sgcr=sgcr, sorg=sorg, swl=swl, sgro=sgro, krgendanchor=krgendanchor)
    gasoil2 = GasOil(sgcr=sgcr, sorg=sorg, swl=swl, sgro=sgro, krgendanchor=None)
    gasoil.add_corey_gas(ng=2, krgend=krgend, krgmax=krgmax)
    gasoil2.add_corey_gas(ng=2, krgend=krgend)
    gasoil.add_corey_oil(nog=2, kroend=kroend, kromax=kromax)
    gasoil2.add_corey_oil(nog=2, kroend=kroend, kromax=kromax)
    gasoil2.table.plot(
        ax=axes, x="SG", y="KRG", c="pink", alpha=0.7, label="KRG*", linewidth=2
    )
    gasoil.table.plot(
        ax=axes, x="SG", y="KRG", c="red", alpha=1, label="KRG", linewidth=2
    )
    gasoil.table.plot(
        ax=axes, x="SG", y="KROG", c="green", alpha=1, label="KROG", linewidth=2
    )
    plt.ylim([-0.02, 1])
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    axes.annotate(
        "KROMAX",
        xy=(0, kromax),
        arrowprops=dict(arrowstyle="->"),
        xytext=(0.1, kromax + 0.07),
    )
    axes.annotate(
        "KROEND",
        xy=(sgro, kroend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(sgro + 0.07, kroend + 0.05),
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
    axes.text(0.0, 0.07, "SGCR")
    axes.annotate(
        "", xy=(-0.01, 0.04), xytext=(sgcr, 0.04), arrowprops=dict(arrowstyle="<->")
    )
    axes.text(0.0, 0.73, "SGRO")
    axes.annotate(
        "",
        xy=(-0.01, kroend - 0.02),
        xytext=(sgro, kroend - 0.02),
        arrowprops=dict(arrowstyle="<->"),
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


def make_wateroil_endpoints(show=True):
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
        ax=axes, x="SW", y="KRW", c="blue", alpha=1, label="KRW", linewidth=2
    )
    wateroil.table.plot(
        ax=axes, x="SW", y="KROW", c="green", alpha=1, label="KROW", linewidth=2
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


def make_gaswater_endpoints(show=True):
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
        ax=axes, x="SW", y="KRW", c="blue", alpha=1, label="KRW", linewidth=2
    )
    gaswater.gasoil.table.plot(
        ax=axes, x="SL", y="KRG", c="red", alpha=1, label="KRG", linewidth=2
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


def make_wateroil_idc2(show=True):
    plt.xkcd()
    _, axes = plt.subplots()
    swirr = 0.05
    swl = 0.1
    swcr = 0.1
    sorw = 0.25
    krwend = 0.8
    krwmax = 1
    kroend = 0.85
    wateroil = WaterOil(swirr=swirr, swl=swl, swcr=swcr, sorw=sorw)
    wateroil.add_corey_water(nw=2, krwend=krwend, krwmax=krwmax)
    wateroil.add_corey_oil(now=1.8, kroend=kroend)
    wateroil.table.plot(
        ax=axes, x="SW", y="KRW", c="blue", alpha=1, label="KRW", linewidth=2
    )
    wateroil.table.plot(
        ax=axes, x="SW", y="KROW", c="green", alpha=1, label="KROW", linewidth=2
    )
    plt.ylim([-0.02, 1.02])
    plt.xlim([-0.02, 1.02])
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    axes.annotate(
        "KROEND",
        xy=(swl, kroend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(swl + 0.1, kroend + 0.1),
    )
    axes.annotate(
        "KRWEND",
        xy=(1 - sorw, krwend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(1 - 0.15, krwend - 0.02),
    )
    axes.annotate(
        "KRWMAX",
        xy=(1, krwmax),
        arrowprops=dict(arrowstyle="->"),
        xytext=(1 - 0.3, krwmax - 0.02),
    )
    axes.annotate(
        "SWIRR",
        xy=(swirr, 0),
        arrowprops=dict(arrowstyle="->"),
        xytext=(swirr - 0.12, 0 + 0.1),
    )
    axes.annotate(
        "SWL≈SWCR",
        xy=(swl, 0),
        arrowprops=dict(arrowstyle="->"),
        xytext=(swl - 0.055, 0 + 0.14),
    )
    plt.xlabel("SW", labelpad=-10)
    axes.legend(loc="upper center")
    axes.annotate(
        "", xy=(1 - sorw, 0.02), xytext=(1, 0.02), arrowprops=dict(arrowstyle="<->")
    )
    axes.text(1 - sorw + 0.04, 0.04, "SORW")

    # Initial state and saturation direction:
    plt.vlines(swl, ymin=0, ymax=1, colors="darkorange", linestyles="dashed")
    axes.arrow(
        swl + 0.03,
        kroend - 0.01,
        0.05,
        -0.09,
        head_width=0.025,
        fill=True,
        facecolor="darkorange",
    )
    axes.arrow(
        swl + 0.03,
        0.03,
        0.05,
        0.01,
        head_width=0.025,
        fill=True,
        facecolor="darkorange",
    )

    if show:
        plt.show()


def make_gasoil_cdi2(show=True):
    plt.xkcd()
    _, axes = plt.subplots()
    swl = 0.1
    sgcr = 0.2
    sorg = 0.2
    krgend = 0.8
    krgmax = 1
    kroend = 0.85
    gasoil = GasOil(sgcr=sgcr, sorg=sorg, swl=swl, krgendanchor="sorg")
    gasoil.add_corey_gas(ng=3, krgend=krgend, krgmax=krgmax)
    gasoil.add_corey_oil(nog=3, kroend=kroend)
    gasoil.table.plot(
        ax=axes, x="SG", y="KRG", c="red", alpha=1, label="KRG", linewidth=2
    )
    gasoil.table.plot(
        ax=axes, x="SG", y="KROG", c="green", alpha=1, label="KROG", linewidth=2
    )
    plt.ylim([-0.02, 1])
    plt.xlim([-0.02, 1])
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    axes.annotate(
        "KROEND",
        xy=(0, kroend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(0.1, kroend + 0.02),
    )
    axes.annotate(
        "KRGEND",
        xy=(1 - sorg - swl, krgend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(1 - 0.2, krgend - 0.02),
    )
    axes.annotate(
        "KRGMAX",
        xy=(1 - swl, krgmax),
        arrowprops=dict(arrowstyle="->"),
        xytext=(1 - 0.35, krgmax - 0.04),
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

    # Initial state and saturation direction:
    plt.vlines(0.0, ymin=0, ymax=1, colors="darkorange", linestyles="dashed")
    axes.arrow(
        0.03,
        kroend - 0.03,
        0.08,
        -0.2,
        head_width=0.025,
        fill=True,
        facecolor="darkorange",
    )
    axes.arrow(
        0.03,
        0.03,
        0.1,
        0.001,
        head_width=0.025,
        fill=True,
        facecolor="darkorange",
    )
    if show:
        plt.show()


def make_gasoil_cid2(show=True):
    plt.xkcd()
    _, axes = plt.subplots()
    swl = 0.1
    sgcr = 0.15
    sgro = 0.15
    sorg = 0.15
    krgend = 1
    krgmax = 1
    kroend = 0.8
    kromax = 0.9
    gasoil = GasOil(sgcr=sgcr, sorg=sorg, sgro=sgro, swl=swl, krgendanchor="")
    gasoil.add_corey_gas(ng=3, krgend=krgend, krgmax=krgmax)
    gasoil.add_corey_oil(nog=3, kroend=kroend, kromax=kromax)
    gasoil.table.plot(
        ax=axes, x="SG", y="KRG", c="red", alpha=1, label="KRG", linewidth=2
    )
    gasoil.table.plot(
        ax=axes, x="SG", y="KROG", c="green", alpha=1, label="KROG", linewidth=2
    )
    plt.ylim([-0.02, 1])
    plt.xlim([-0.02, 1])
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    axes.annotate(
        "KROEND",
        xy=(sgro, kroend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(sgro + 0.1, kroend - 0.05),
    )
    axes.annotate(
        "KROMAX",
        xy=(0, kromax),
        arrowprops=dict(arrowstyle="->"),
        xytext=(0.1, kromax + 0.05),
    )
    axes.annotate(
        "KRGEND",
        xy=(1 - swl, krgend - 0.01),
        arrowprops=dict(arrowstyle="->"),
        xytext=(1 - 0.35, krgend - 0.04),
    )
    axes.annotate(
        "SGCR=SGRO",
        xy=(sgcr, 0),
        arrowprops=dict(arrowstyle="->"),
        xytext=(sgcr - 0.1, 0 + 0.14),
    )
    plt.xlabel("SG", labelpad=-10)
    axes.annotate(
        "",
        xy=(1 - sorg - swl, 0.08),
        xytext=(1 - swl, 0.08),
        arrowprops=dict(arrowstyle="<->"),
    )
    axes.text(1 - sorg - swl + 0.02, 0.1, "SORG")
    axes.annotate(
        "", xy=(1 - swl, 0.02), xytext=(1, 0.02), arrowprops=dict(arrowstyle="<->")
    )
    axes.text(1 - swl + 0.04, 0.04, "SWL")
    axes.legend(loc="upper center")

    # Initial state and saturation direction:
    plt.vlines(1 - swl, ymin=0, ymax=1, colors="darkorange", linestyles="dashed")
    axes.arrow(
        1 - swl - 0.03,
        krgend - 0.03,
        -0.08,
        -0.2,
        head_width=0.025,
        fill=True,
        facecolor="darkorange",
    )
    axes.arrow(
        1 - swl - 0.03,
        0.03,
        -0.13,
        0.001,
        head_width=0.025,
        fill=True,
        facecolor="darkorange",
    )
    if show:
        plt.show()


def make_gaswater_icd2(show=True):
    plt.xkcd()
    _, axes = plt.subplots()
    swl = 0.1
    swcr = 0.2
    sgcr = 0.2
    sgrw = 0.2
    krwend = 0.65
    krwmax = 0.75
    krgend = 1
    gaswater = GasWater(swl=swl, swcr=swcr, sgrw=sgrw, sgcr=sgcr, h=0.001)
    gaswater.add_corey_water(nw=2, krwend=krwend, krwmax=krwmax)
    gaswater.add_corey_gas(ng=2, krgend=krgend)
    gaswater.wateroil.table.plot(
        ax=axes, x="SW", y="KRW", c="blue", alpha=1, label="KRW", linewidth=2
    )
    gaswater.gasoil.table.plot(
        ax=axes, x="SL", y="KRG", c="red", alpha=1, label="KRG", linewidth=2
    )
    plt.ylim([-0.02, 1])
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    axes.annotate(
        "KRGEND",
        xy=(swl, krgend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(swl + 0.1, krgend - 0.1),
    )
    axes.annotate(
        "KRWEND",
        xy=(1 - sgrw, krwend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(1 - sgrw - 0.23, krwmax - 0.1),
    )
    axes.annotate(
        "KRWMAX",
        xy=(1 - 0.01, krwmax),
        arrowprops=dict(arrowstyle="->"),
        xytext=(1 - 0.23, krwmax),
    )
    axes.annotate(
        "SWL",
        xy=(swl, 0),
        arrowprops=dict(arrowstyle="->"),
        xytext=(swl - 0.07, 0 + 0.12),
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

    # Initial state and saturation direction:
    plt.vlines(swl, ymin=0, ymax=1, colors="darkorange", linestyles="dashed")
    axes.arrow(
        swl + 0.03,
        krgend - 0.03,
        0.08,
        -0.2,
        head_width=0.025,
        fill=True,
        facecolor="darkorange",
    )
    axes.arrow(
        swl + 0.03,
        0.03,
        0.1,
        0.005,
        head_width=0.025,
        fill=True,
        facecolor="darkorange",
    )

    if show:
        plt.show()


def make_gaswater_dci1(show=True):
    # CO2 storage
    plt.xkcd()
    _, axes = plt.subplots()
    swl = 0.1
    swcr = 0.1
    sgcr = 0.2
    sgrw = 0.0
    krwend = 1
    krgend = 0.8
    gaswater = GasWater(swl=swl, swcr=swcr, sgrw=sgrw, sgcr=sgcr, h=0.001)
    gaswater.add_corey_water(nw=2, krwend=krwend)
    gaswater.add_corey_gas(ng=2, krgend=krgend)
    gaswater.wateroil.table.plot(
        ax=axes, x="SW", y="KRW", c="blue", alpha=1, label="KRW", linewidth=2
    )
    gaswater.gasoil.table.plot(
        ax=axes, x="SL", y="KRG", c="red", alpha=1, label="KRG", linewidth=2
    )
    plt.ylim([-0.02, 1])
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.xlim([-0.02, 1.02])
    axes.annotate(
        "KRGEND",
        xy=(swl, krgend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(swl + 0.1, krgend - 0.1),
    )
    axes.annotate(
        "KRWEND",
        xy=(1 - sgrw, krwend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(1 - sgrw - 0.23, krwend - 0.1),
    )
    axes.annotate(
        "SWL=SWCR",
        xy=(swl, 0),
        arrowprops=dict(arrowstyle="->"),
        xytext=(swl - 0.08, 0 + 0.1),
    )
    plt.xlabel("SW", labelpad=-10)

    # axes.text(1 - sgrw + 0.04, 0.04, "SGRW")
    axes.annotate(
        "", xy=(1 - sgrw, 0.02), xytext=(1, 0.02), arrowprops=dict(arrowstyle="<->")
    )

    axes.text(1 - sgcr + 0.04, 0.14, "SGCR")
    axes.annotate(
        "", xy=(1 - sgcr, 0.12), xytext=(1, 0.12), arrowprops=dict(arrowstyle="<->")
    )

    # Initial state and saturation direction:
    plt.vlines(1, ymin=0, ymax=1, colors="darkorange", linestyles="dashed")
    axes.arrow(
        1 - 0.03,
        krwend - 0.03,
        -0.08,
        -0.14,
        head_width=0.025,
        fill=True,
        facecolor="darkorange",
    )
    axes.arrow(
        1 - 0.03,
        0.03,
        -0.1,
        0.005,
        head_width=0.025,
        fill=True,
        facecolor="darkorange",
    )
    axes.legend(loc="upper center")

    if show:
        plt.show()


def make_gaswater_co2_icd2(show=True):
    # CO2 trapping
    plt.xkcd()
    _, axes = plt.subplots()
    swl = 0.1
    swcr = 0.1
    sgcr = 0.3
    sgrw = 0.3
    krwend = 0.3
    krwmax = 1
    krgend = 0.8
    gaswater = GasWater(swl=swl, swcr=swcr, sgrw=sgrw, sgcr=sgcr, h=0.001)
    gaswater.add_corey_water(nw=3, krwend=krwend, krwmax=krwmax)
    gaswater.add_corey_gas(ng=2, krgend=krgend)
    gaswater.wateroil.table.plot(
        ax=axes, x="SW", y="KRW", c="blue", alpha=1, label="KRW", linewidth=2
    )
    gaswater.gasoil.table.plot(
        ax=axes, x="SL", y="KRG", c="red", alpha=1, label="KRG", linewidth=2
    )
    plt.ylim([-0.02, 1])
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.xlim([-0.02, 1.02])
    axes.annotate(
        "KRGEND",
        xy=(swl, krgend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(swl + 0.1, krgend - 0.1),
    )
    axes.annotate(
        "KRWEND",
        xy=(1 - sgrw, krwend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(1 - sgrw - 0.23, krwend + 0.1),
    )
    axes.annotate(
        "KRWMAX",
        xy=(1, krwmax),
        arrowprops=dict(arrowstyle="->"),
        xytext=(1 - 0.23, krwmax - 0.1),
    )
    axes.annotate(
        "SWL=SWCR",
        xy=(swl, 0),
        arrowprops=dict(arrowstyle="->"),
        xytext=(swl - 0.1, 0 + 0.1),
    )
    plt.xlabel("SW", labelpad=-10)

    axes.text(1 - sgrw + 0.04, 0.04, "SGRW=SGCR")
    axes.annotate(
        "", xy=(1 - sgrw, 0.02), xytext=(1, 0.02), arrowprops=dict(arrowstyle="<->")
    )

    # Initial state and saturation direction:
    sw_initial = 0.3
    plt.vlines(sw_initial, ymin=0, ymax=1, colors="darkorange", linestyles="dashed")
    axes.arrow(
        sw_initial + 0.01,
        0.37,
        0.08,
        -0.12,
        head_width=0.025,
        fill=True,
        facecolor="darkorange",
    )
    axes.arrow(
        sw_initial + 0.01,
        0.03,
        0.1,
        0.03,
        head_width=0.025,
        fill=True,
        facecolor="darkorange",
    )
    axes.legend(loc="upper center")

    if show:
        plt.show()


def make_wateroil_paleooil_idc1(show=True):
    plt.xkcd()
    _, axes = plt.subplots()
    swl = 0.1
    swcr = 0.2
    sorw = 0.15
    socr = 0.25
    krwend = 0.8
    krwmax = 1
    kroend = 0.85
    wateroil = WaterOil(swl=swl, swcr=swcr, sorw=sorw, socr=socr)
    wateroil.add_corey_water(nw=2, krwend=krwend, krwmax=krwmax)
    wateroil.add_corey_oil(now=2, kroend=kroend)
    wateroil.table.plot(
        ax=axes, x="SW", y="KRW", c="blue", alpha=1, label="KRW", linewidth=2
    )
    wateroil.table.plot(
        ax=axes, x="SW", y="KROW", c="green", alpha=1, label="KROW", linewidth=2
    )
    plt.ylim([-0.02, 1])
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    axes.annotate(
        "KROEND",
        xy=(swl, kroend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(swl + 0.07, kroend),
    )
    axes.annotate(
        "KRWEND",
        xy=(1 - sorw, krwend),
        arrowprops=dict(arrowstyle="->"),
        xytext=(1 - sorw - 0.15, krwend + 0.05),
    )
    axes.annotate(
        "KRWMAX",
        xy=(1, krwmax),
        arrowprops=dict(arrowstyle="->"),
        xytext=(1 - 0.25, krwmax - 0.05),
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
        "", xy=(1 - sorw, krwend), xytext=(1, krwend), arrowprops=dict(arrowstyle="<->")
    )
    axes.text(1 - sorw + 0.03, krwend - 0.05, "SORW")
    axes.annotate(
        "",
        xy=(1 - socr, 0.12),
        xytext=(1, 0.12),
        arrowprops=dict(arrowstyle="<->"),
    )
    axes.text(1 - socr + 0.05, 0.14, "SOCR")
    axes.legend(loc="upper center")

    # Initial state and saturation direction:
    plt.vlines(1 - sorw, ymin=0, ymax=1, colors="darkorange", linestyles="dashed")
    axes.arrow(
        1 - sorw - 0.03,
        krwend - 0.03,
        -0.05,
        -0.09,
        head_width=0.025,
        fill=True,
        facecolor="darkorange",
    )
    axes.arrow(
        1 - sorw - 0.03,
        0.03,
        -0.05,
        0.01,
        head_width=0.025,
        fill=True,
        facecolor="darkorange",
    )

    if show:
        plt.show()

def make_gaswater_paleogas_dci3(show=True):
    plt.xkcd()
    _, axes = plt.subplots()
    swl = 0.1
    sgl = 0.1
    swcr = 0.2
    sgcr = 0.3
    sgrw = sgl
    krwend = 0.85
    krgend = 0.9
    gaswater = GasWater(swl=swl, sgl=sgl, swcr=swcr, sgrw=sgrw, sgcr=sgcr, h=0.001)
    gaswater.add_corey_water(nw=3, krwend=krwend)
    gaswater.add_corey_gas(ng=2, krgend=krgend)
    gaswater.wateroil.table.plot(
        ax=axes, x="SW", y="KRW", c="blue", alpha=1, label="KRW", linewidth=2
    )
    gaswater.gasoil.table.plot(
        ax=axes, x="SL", y="KRG", c="red", alpha=1, label="KRG", linewidth=2
    )
    plt.ylim([-0.02, 1])
    plt.xlim([0, 1.02])
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
        xytext=(1 - sgrw - 0.23, krwend - 0.1),
    )
    axes.annotate(
        "SWL=SWCR",
        xy=(swl, 0),
        arrowprops=dict(arrowstyle="->"),
        xytext=(swl - 0.1, 0 + 0.14),
    )
    plt.xlabel("SW", labelpad=-10)
    axes.legend(loc="upper center")

    axes.text(1 - sgrw + 0.01, krwend - 0.05, "SGRW")
    axes.annotate(
        "", xy=(1 - sgrw, krwend), xytext=(1, krwend), arrowprops=dict(arrowstyle="<->")
    )

    axes.text(1 - sgcr + 0.04, 0.09, "SGCR")
    axes.annotate(
        "", xy=(1 - sgcr, 0.07), xytext=(1, 0.07), arrowprops=dict(arrowstyle="<->")
    )

    axes.text(1 - sgl + 0.02, 0.02, "SGL")
    axes.annotate(
        "", xy=(1 - sgl, 0.00), xytext=(1, 0.00), arrowprops=dict(arrowstyle="<->")
    )

    # Initial state and saturation direction:
    plt.vlines(1 - sgl, ymin=0, ymax=1, colors="darkorange", linestyles="dashed")
    # Upper arrow, along water curve:
    axes.arrow(
        1 - sgl - 0.03,
        krwend - 0.06,
        -0.05,
        -0.13,
        head_width=0.025,
        fill=True,
        facecolor="darkorange",
    )
    # Lower arrow, along gas curve:
    axes.arrow(
        1 - sgl - 0.03,
        0.03,
        -0.08,
        0.00,
        head_width=0.025,
        fill=True,
        facecolor="darkorange",
    )
    if show:
        plt.show()


if __name__ == "__main__":
    main()
