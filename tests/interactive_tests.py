"""
Run this module from command line to run a few
tests intended for human inspection

 $ python interactive_tests.py

If you want to run individual tests, import this module in
a Python session and run the functions manually.
"""

from __future__ import print_function

import random

import numpy as np

from pyscal import WaterOil, WaterOilGas, GasOil, GasWater, utils, PyscalFactory

from test_scalrecommendation import LOW_SAMPLE_LET, BASE_SAMPLE_LET, HIGH_SAMPLE_LET


def interpolation_art(repeats=50, interpolants=30, curvetype="corey"):
    """This code was used to create the Pyscal logo"""
    from matplotlib import pyplot as plt

    cmap = plt.get_cmap("viridis")
    _, mpl_ax = plt.subplots()
    for _ in range(repeats):
        swl = random.uniform(0, 0.1)
        swcr = swl + random.uniform(0, 0.1)
        sorw = random.uniform(0, 0.2)
        wo_low = WaterOil(swl=swl, swcr=swcr, sorw=sorw)
        wo_high = WaterOil(swl=swl + 0.1, swcr=swcr + 0.1, sorw=sorw + 0.1)
        if curvetype == "corey":
            wo_low.add_corey_water(
                nw=random.uniform(1, 3), krwend=random.uniform(0.5, 1)
            )
            wo_high.add_corey_water(
                nw=random.uniform(1, 3), krwend=random.uniform(0.5, 1)
            )
            wo_low.add_corey_oil(
                now=random.uniform(1, 3), kroend=random.uniform(0.5, 1)
            )
            wo_high.add_corey_oil(
                now=random.uniform(1, 3), kroend=random.uniform(0.5, 1)
            )
        elif curvetype == "let":
            wo_low.add_LET_water(
                l=random.uniform(1, 3),
                e=random.uniform(1, 3),
                t=random.uniform(1, 3),
                krwend=random.uniform(0.5, 1),
            )
            wo_high.add_LET_water(
                l=random.uniform(1, 3),
                e=random.uniform(1, 3),
                t=random.uniform(1, 3),
                krwend=random.uniform(0.5, 1),
            )
            wo_low.add_LET_oil(
                l=random.uniform(1, 3),
                e=random.uniform(1, 3),
                t=random.uniform(1, 3),
                kroend=random.uniform(0.5, 1),
            )
            wo_high.add_LET_oil(
                l=random.uniform(1, 3),
                e=random.uniform(1, 3),
                t=random.uniform(1, 3),
                kroend=random.uniform(0.5, 1),
            )
        else:
            print("ERROR, wrong curvetype")
        color = cmap(random.random())
        for tparam in np.arange(0, 1, 1.0 / interpolants):
            wo_ip = utils.interpolate_wo(wo_low, wo_high, tparam)
            wo_ip.plotkrwkrow(mpl_ax, color=color, alpha=0.3)
    plt.show()


def test_interpolate_wo():
    """Discrete test scenarios for wateroil interpolation"""
    swl_l = random.uniform(0, 0.1)
    swcr_l = swl_l + random.uniform(0, 0.1)
    sorw_l = random.uniform(0, 0.2)
    swl_h = random.uniform(0, 0.1)
    swcr_h = swl_h + random.uniform(0, 0.1)
    sorw_h = random.uniform(0, 0.2)
    wo_low = WaterOil(swl=swl_l, swcr=swcr_l, sorw=sorw_l, h=0.001)
    wo_high = WaterOil(swl=swl_h, swcr=swcr_h, sorw=sorw_h, h=0.001)
    wo_low.add_corey_water(nw=random.uniform(1, 3), krwend=random.uniform(0.5, 1))
    wo_high.add_corey_water(nw=random.uniform(1, 3), krwend=random.uniform(0.5, 1))
    wo_low.add_corey_oil(now=random.uniform(1, 3), kroend=random.uniform(0.5, 1))
    wo_high.add_corey_oil(now=random.uniform(1, 3), kroend=random.uniform(0.5, 1))
    wo_low.add_simple_J(a=random.uniform(0.1, 2), b=random.uniform(-2, -1))
    wo_high.add_simple_J(a=random.uniform(0.1, 2), b=random.uniform(-2, 1))
    print(
        " ** Low curve WaterOil (red):\n"
        + wo_low.swcomment
        + wo_low.krwcomment
        + wo_low.krowcomment
        + wo_low.pccomment
    )
    print(
        " ** High curve WaterOil (blue):\n"
        + wo_high.swcomment
        + wo_high.krwcomment
        + wo_high.krowcomment
        + wo_high.pccomment
    )

    from matplotlib import pyplot as plt

    _, mpl_ax = plt.subplots()
    wo_low.plotkrwkrow(mpl_ax, color="red")
    wo_high.plotkrwkrow(mpl_ax, color="blue")
    for tparam in np.arange(0, 1, 0.1):
        wo_ip = utils.interpolate_wo(wo_low, wo_high, tparam, h=0.001)
        wo_ip.plotkrwkrow(mpl_ax, color="green")
    mpl_ax.set_title("WaterOil, random Corey, linear y-scale")
    plt.show()

    # Plot again with log yscale:
    _, mpl_ax = plt.subplots()
    wo_low.plotkrwkrow(mpl_ax, color="red")
    wo_high.plotkrwkrow(mpl_ax, color="blue")
    for tparam in np.arange(0, 1, 0.1):
        wo_ip = utils.interpolate_wo(wo_low, wo_high, tparam, h=0.001)
        wo_ip.plotkrwkrow(mpl_ax, color="green", logyscale=True)
    mpl_ax.set_title("WaterOil, random Corey, log y-scale")
    plt.show()

    # Capillary pressure
    _, mpl_ax = plt.subplots()
    wo_low.plotpc(mpl_ax, color="red", logyscale=True)
    wo_high.plotpc(mpl_ax, color="blue", logyscale=True)
    for tparam in np.arange(0, 1, 0.1):
        wo_ip = utils.interpolate_wo(wo_low, wo_high, tparam, h=0.001)
        wo_ip.plotpc(mpl_ax, color="green", logyscale=True)
    mpl_ax.set_title("WaterOil, capillary pressure")
    plt.show()


def test_interpolate_go():
    """Interactive tests for gasoil"""
    swl_l = random.uniform(0, 0.1)
    sgcr_l = random.uniform(0, 0.1)
    swl_h = random.uniform(0, 0.1)
    sgcr_h = random.uniform(0, 0.1)
    sorg_l = random.uniform(0, 0.2)
    sorg_h = random.uniform(0, 0.2)
    if random.uniform(0, 1) > 0.5:
        krgendanchor_l = "sorg"
    else:
        krgendanchor_l = ""
    if random.uniform(0, 1) > 0.5:
        krgendanchor_h = "sorg"
    else:
        krgendanchor_h = ""
    go_low = GasOil(
        swl=swl_l, sgcr=sgcr_l, sorg=sorg_l, krgendanchor=krgendanchor_l, h=0.001
    )
    go_high = GasOil(
        swl=swl_h, sgcr=sgcr_h, sorg=sorg_h, krgendanchor=krgendanchor_h, h=0.001
    )
    go_low.add_corey_gas(ng=random.uniform(1, 3), krgend=random.uniform(0.5, 1))
    go_high.add_corey_gas(ng=random.uniform(1, 3), krgend=random.uniform(0.5, 1))
    go_low.add_corey_oil(nog=random.uniform(1, 3), kroend=random.uniform(0.5, 1))
    go_high.add_corey_oil(nog=random.uniform(1, 3), kroend=random.uniform(0.5, 1))
    print(
        " ** Low curve GasOil (red):\n"
        + go_low.sgcomment
        + go_low.krgcomment
        + go_low.krogcomment
    )
    print(
        " ** High curve GasOil (blue):\n"
        + go_high.sgcomment
        + go_high.krgcomment
        + go_high.krogcomment
    )

    from matplotlib import pyplot as plt

    _, mpl_ax = plt.subplots()
    go_low.plotkrgkrog(mpl_ax, color="red")
    go_high.plotkrgkrog(mpl_ax, color="blue")

    for tparam in np.arange(0, 1, 0.1):
        go_ip = utils.interpolate_go(go_low, go_high, tparam)
        go_ip.plotkrgkrog(mpl_ax, color="green")
    mpl_ax.set_title("GasOil, random Corey, linear y-scale")
    plt.show()

    _, mpl_ax = plt.subplots()
    go_low.plotkrgkrog(mpl_ax, color="red")
    go_high.plotkrgkrog(mpl_ax, color="blue")
    # Plot again with log yscale:
    for tparam in np.arange(0, 1, 0.1):
        go_ip = utils.interpolate_go(go_low, go_high, tparam)
        go_ip.plotkrgkrog(mpl_ax, color="green", logyscale=True)
    mpl_ax.set_title("GasOil, random Corey, log y-scale")
    plt.show()

    # Capillary pressure - This is barely supported for gasoil
    # so the plotpc() function is missing. Include calculations
    # here so we ensure we don't crash on the all zeros.
    # _, mpl_ax = plt.subplots()
    # go_low.plotpc(mpl_ax, color="red", logyscale=True)
    # go_high.plotpc(mpl_ax, color="blue", logyscale=True)
    for tparam in np.arange(0, 1, 0.1):
        go_ip = utils.interpolate_go(go_low, go_high, tparam, h=0.001)
        # go_ip.plotpc(mpl_ax, color="green", logyscale=True)
    # mpl_ax.set_title("GasOil, capillary pressure")
    # plt.show()


def test_interpolate_gw():
    """Discrete test scenarios for gaswater interpolation"""
    swl_l = random.uniform(0, 0.1)
    swcr_l = swl_l + random.uniform(0, 0.1)
    sgrw_l = random.uniform(0, 0.2)
    sgcr_l = random.uniform(0, 0.3)
    swl_h = random.uniform(0, 0.1)
    swcr_h = swl_h + random.uniform(0, 0.1)
    sgrw_h = random.uniform(0, 0.2)
    sgcr_h = random.uniform(0, 0.3)
    gw_low = GasWater(swl=swl_l, swcr=swcr_l, sgrw=sgrw_l, sgcr=sgcr_l, h=0.001)
    gw_high = GasWater(swl=swl_h, swcr=swcr_h, sgrw=sgrw_h, sgcr=sgcr_h, h=0.001)
    gw_low.add_corey_water(nw=random.uniform(1, 3), krwend=random.uniform(0.5, 1))
    gw_high.add_corey_water(nw=random.uniform(1, 3), krwend=random.uniform(0.5, 1))
    gw_low.add_corey_gas(ng=random.uniform(1, 3), krgend=random.uniform(0.5, 1))
    gw_high.add_corey_gas(ng=random.uniform(1, 3), krgend=random.uniform(0.5, 1))
    print(
        " ** Low curve GasWater (red):\n"
        + gw_low.swcomment
        + gw_low.krwcomment
        + gw_low.krgcomment
    )
    print(
        " ** High curve GasWater (blue):\n"
        + gw_high.swcomment
        + gw_high.krwcomment
        + gw_high.krgcomment
    )

    from matplotlib import pyplot as plt

    _, mpl_ax = plt.subplots()
    gw_low.plotkrwkrg(mpl_ax, color="red")
    gw_high.plotkrwkrg(mpl_ax, color="blue")
    for tparam in np.arange(0, 1, 0.1):
        gw_wo_ip = utils.interpolate_wo(
            gw_low.wateroil, gw_high.wateroil, tparam, h=0.001
        )
        gw_go_ip = utils.interpolate_go(gw_low.gasoil, gw_high.gasoil, tparam, h=0.001)
        gw_ip = GasWater()
        gw_ip.gasoil = gw_go_ip
        gw_ip.wateroil = gw_wo_ip
        gw_ip.plotkrwkrg(mpl_ax, color="green")
    mpl_ax.set_title("GasWater, random Corey, linear y-scale")
    plt.show()

    # Plot again with log yscale:
    _, mpl_ax = plt.subplots()
    gw_low.plotkrwkrg(mpl_ax, color="red")
    gw_high.plotkrwkrg(mpl_ax, color="blue")
    for tparam in np.arange(0, 1, 0.1):
        gw_wo_ip = utils.interpolate_wo(
            gw_low.wateroil, gw_high.wateroil, tparam, h=0.001
        )
        gw_go_ip = utils.interpolate_go(gw_low.gasoil, gw_high.gasoil, tparam, h=0.001)
        gw_ip = GasWater()
        gw_ip.gasoil = gw_go_ip
        gw_ip.wateroil = gw_wo_ip
        gw_ip.plotkrwkrg(mpl_ax, color="green", logyscale=True)
    mpl_ax.set_title("GasWater, random Corey, log y-scale")
    plt.show()


def interpolateplottest():
    """Demonstration of interpolation pointwise between LET curves"""
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.style.use("ggplot")

    rec = PyscalFactory.create_scal_recommendation(
        {"low": LOW_SAMPLE_LET, "base": BASE_SAMPLE_LET, "high": HIGH_SAMPLE_LET},
        "FOO",
        h=0.001,
    )
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # Choosing logarithmic spaced interpolation parameters
    # is not the same as interpolating in log(kr)-space
    # check the effect by setting
    #  for t in -2 + np.logspace(1e-5,1e-1,15):
    # and
    #  for t in -1 + np.logspace(1e-5,1e-1,15)
    # in the loops below. Curves get clustered to the bottom
    # in both linear and log(kr) spaces, but there
    # still might be some other distribution for the interpolants
    # that yields something that spans nicely both the linear and the
    # logarithmic kr space (?)

    for tparam in np.arange(-1, 0, 0.2):
        interp = rec.interpolate(tparam, h=0.001)
        interp.wateroil.plotkrwkrow(ax1, "r")
        interp.wateroil.plotkrwkrow(ax2, "r")

    for tparam in np.arange(0, 1, 0.2):
        interp = rec.interpolate(tparam, h=0.001)
        interp.wateroil.plotkrwkrow(ax1, "g")
        interp.wateroil.plotkrwkrow(ax2, "g")

    rec.low.wateroil.plotkrwkrow(ax1, linewidth=2, linestyle=":")
    rec.base.wateroil.plotkrwkrow(ax1, linewidth=2)
    rec.high.wateroil.plotkrwkrow(ax1, linewidth=2, linestyle="--")
    rec.low.wateroil.plotkrwkrow(ax2, linewidth=2, linestyle=":")
    rec.base.wateroil.plotkrwkrow(ax2, linewidth=2)
    rec.high.wateroil.plotkrwkrow(ax2, linewidth=2, linestyle="--")
    ax2.set_yscale("log")
    ax2.set_ylim([1e-10, 1])
    ax1.set_title("Water-oil, low, base, high and interpolants")
    ax2.set_title("Water-oil, low, base, high and interpolants")

    for tparam in np.arange(-1, 0, 0.2):
        interp = rec.interpolate(tparam, h=0.001)
        interp.gasoil.plotkrgkrog(ax3, "r")
        interp.gasoil.plotkrgkrog(ax4, "r")

    for tparam in np.arange(0, 1, 0.2):
        interp = rec.interpolate(tparam, h=0.001)
        interp.gasoil.plotkrgkrog(ax3, "g")
        interp.gasoil.plotkrgkrog(ax4, "g")

    rec.low.gasoil.plotkrgkrog(ax3, linewidth=2, linestyle=":")
    rec.base.gasoil.plotkrgkrog(ax3, linewidth=2)
    rec.high.gasoil.plotkrgkrog(ax3, linewidth=2, linestyle="--")
    rec.low.gasoil.plotkrgkrog(ax4, linewidth=2, linestyle=":")
    rec.base.gasoil.plotkrgkrog(ax4, linewidth=2)
    rec.high.gasoil.plotkrgkrog(ax4, linewidth=2, linestyle="--")
    ax3.set_title("Gas-oil, low, base, high and interpolants")
    ax4.set_title("Gas-oil, low, base, high and interpolants")
    ax4.set_yscale("log")
    ax4.set_ylim([1e-05, 1])
    plt.subplots_adjust(hspace=0.3)
    plt.show()


def interpolatetest():
    """Test interpolation (sample test) in a scal recommentation"""
    rec = PyscalFactory.create_scal_recommendation(
        {"low": LOW_SAMPLE_LET, "base": BASE_SAMPLE_LET, "high": HIGH_SAMPLE_LET},
        "foo",
        h=0.001,
    )
    rec.add_simple_J()  # Add default pc curve
    #    print rec.low.wateroil.table
    interpolant = rec.interpolate(0.3, parameter2=-0.9, h=0.05)
    print(interpolant.wateroil.SWOF())
    print(interpolant.gasoil.SGOF())

    print("Consistency check: ", end=" ")
    print(interpolant.threephaseconsistency())


def letspan():
    """Demonstration of how random LET (random individually and
    uncorrelated in L, eand T) curves span out the relperm-space
    between low and high LET curves

    If the low and high LET curves do not cross, the random
    curves are all between the low and high curves.

    """

    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.style.use("ggplot")

    let_w = {
        "l": [2, 4],
        "e": [1, 2],
        "t": [2, 1],  # first value should be larger than first to avoid crossing
        "krwend": [0.9, 0.5],
        "sorw": [0.05, 0.1],
    }

    # Parameter test set from SCAL group, first is pessimistic, second
    # is optimistic
    let_w = {
        "l": [2.323, 4.436],
        "e": [2, 8],
        "t": [1.329, 0.766],  # first value should be larger
        # than first to avoid crossing
        "krwend": [0.9, 0.6],
        "sorw": [0.02, 0.137],
    }

    # LET oil:
    let_o = {
        "l": [4.944, 2.537],
        "e": [5, 2],
        "t": [0.68, 1.549],  # first value should be larger
        # than first to avoid crossing
        "kroend": [1, 1],
        "sorw": [0.02, 0.137],
    }

    # We need sorted versions for the random function
    slimw = {x: sorted(let_w[x]) for x in let_w}
    slimo = {x: sorted(let_o[x]) for x in let_o}

    _, mpl_ax = plt.subplots()
    for _ in range(100):
        swof = WaterOil(
            h=0.01, swl=0.16, sorw=random.uniform(slimw["sorw"][0], slimw["sorw"][1])
        )
        swof.add_LET_water(
            l=random.uniform(slimw["l"][0], slimw["l"][1]),
            e=random.uniform(slimw["e"][0], slimw["e"][1]),
            t=random.uniform(slimw["t"][0], slimw["t"][1]),
            krwend=random.uniform(slimw["krwend"][0], slimw["krwend"][1]),
        )
        swof.add_LET_oil(
            l=random.uniform(slimo["l"][0], slimo["l"][1]),
            e=random.uniform(slimo["e"][0], slimo["e"][1]),
            t=random.uniform(slimo["t"][0], slimo["t"][1]),
        )
        swof.plotkrwkrow(mpl_ax=mpl_ax, alpha=0.1)
    # Boundary lines
    swof = WaterOil(h=0.01, sorw=let_w["sorw"][0], swl=0.16)
    swof.add_LET_water(
        l=let_w["l"][0], e=let_w["e"][0], t=let_w["t"][0], krwend=let_w["krwend"][0]
    )
    swof.add_LET_oil(l=let_o["l"][0], e=let_o["e"][0], t=let_o["t"][0])
    swof.plotkrwkrow(mpl_ax=mpl_ax, color="red", label="Low")
    swof = WaterOil(h=0.01, sorw=let_w["sorw"][1], swl=0.16)
    swof.add_LET_water(
        l=let_w["l"][1], e=let_w["e"][1], t=let_w["t"][1], krwend=let_w["krwend"][1]
    )
    swof.add_LET_oil(l=let_o["l"][1], e=let_o["e"][1], t=let_o["t"][1])
    swof.plotkrwkrow(mpl_ax=mpl_ax, color="red", label="High")
    # mpl_ax.set_yscale('log')
    plt.show()


def testplot():
    """Generate and plot relperm curves

    Use this as a template function.
    """
    import matplotlib.pyplot as plt

    swof = WaterOil(tag="Testcurve", h=0.01, swirr=0.2, swl=0.2, sorw=0.1)
    swof.add_corey_water(nw=5, krwend=0.7, krwmax=0.9)
    swof.add_corey_oil(now=2, kroend=0.95)
    swof.add_LET_water(l=2, e=1, t=1.4, krwend=0.7, krwmax=0.9)
    swof.add_LET_oil(l=2, e=1, t=1.4, kroend=0.9)

    # Print the first 7 lines of SWOF:
    print("\n".join(swof.SWOF().split("\n")[0:8]))
    _, mpl_ax = plt.subplots()
    swof.plotkrwkrow(mpl_ax)
    plt.show()


def multiplesatnums():
    """Test of a case where there are multiple satnums"""
    satnums = []
    satnums.append(
        WaterOilGas(swirr=0.01, swl=0.1, sgcr=0.05, sorg=0.1, tag="Good sand")
    )
    satnums[0].wateroil.add_corey_oil(3)
    satnums[0].wateroil.add_corey_water(2.5)
    satnums[0].wateroil.add_simple_J()
    satnums[0].gasoil.add_corey_oil(4)
    satnums[0].gasoil.add_corey_gas(1)

    satnums.append(
        WaterOilGas(swirr=0.35, swl=0.4, sgcr=0.05, sorg=0.1, tag="Bad sand")
    )
    satnums[1].wateroil.add_corey_oil(4)
    satnums[1].wateroil.add_corey_water(5)
    satnums[1].wateroil.add_simple_J()
    satnums[1].gasoil.add_corey_oil(4)
    satnums[1].gasoil.add_corey_gas(1)

    if satnums[0].selfcheck() and satnums[1].selfcheck():
        print(satnums[0].wateroil.SWOF())
        print(satnums[1].wateroil.SWOF(header=False))
        print(satnums[0].gasoil.SGOF())
        print(satnums[1].gasoil.SGOF(header=False))


def testgascurves():
    """test of gas-oil curves"""
    import matplotlib.pyplot as plt

    sgof = GasOil(tag="Testcurve", h=0.02, swirr=0.18, swl=0.31, sorg=0.09, sgcr=0.04)
    sgof.add_corey_gas(ng=1.5, krgend=0.7)
    sgof.add_corey_oil(nog=2, kroend=0.4)
    sgof.add_LET_gas(l=2, e=1, t=1.4, krgend=0.9)
    sgof.add_LET_oil(l=2, e=3, t=1.4, kroend=0.7)

    print(sgof.table)
    _, mpl_ax = plt.subplots()
    sgof.plotkrgkrog(mpl_ax)
    # mpl_ax.set_yscale('log')
    print(sgof.SGOF())
    plt.show()


def main():
    """Entry point for interactive tests, will run
    some tests where the intention is that the user will
    look at what is displayed, and potentially see if
    something looks really bad"""
    print("-- **********************************")
    print("-- Manual check of output")
    swof = WaterOil(tag="Good sand, SATNUM 1", h=0.1, swl=0.1)
    swof.add_corey_water()
    swof.add_LET_water()
    swof.add_corey_oil()
    swof.add_simple_J()

    print(swof.SWOF())

    sgof = GasOil(tag="Good sand, SATNUM 1", h=0.1)
    sgof.add_corey_gas()
    sgof.add_corey_oil()
    print(sgof.SGOF())

    print("")
    print("-- ******************************************")
    print("-- Manual visual check of interpolation in LET-space")
    print("--  Check:")
    print("--   * green curves are between red and blue blue line")
    print("-- (close plot window to continue)")
    for _ in range(0, 5):
        test_interpolate_wo()
        test_interpolate_go()
        test_interpolate_gw()
    print("")
    print("-- ******************************************")
    print("-- Manual visual check of interpolation in LET-space")
    print("--  Check:")
    print("--   * Red curves are between dotted and solid blue line")
    print("--   * Green curves are between solid blue and dashed")
    print("-- (close plot window to continue)")
    interpolateplottest()

    print("")
    print("-- ***********************************************")
    print("-- Span of LET curves when LET parameters are varied")
    print("-- within the bounds of the parameters of the red curves")
    print("-- Blue dim curves are allowed to go outside the red boundary curves")
    print("-- (close plot window to continue)")
    letspan()


if __name__ == "__main__":
    main()
