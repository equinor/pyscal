"""Example code for benchmarking of the "fast" feature"""
import timeit

from pyscal import WaterOilGas


def benchme(fast=False, doprint=False):
    """Test function for benchmarking the "fast"-feature

    Pipe the following code snip into "ipython" on the shell:
    > echo "import benchme
      %timeit benchme.benchme()
      %timeit benchme.benchme(fast=True)
      " | ipython

    or run this module directly
    """
    wog = WaterOilGas(swl=0.1, h=0.1, fast=fast)
    wog.wateroil.add_corey_oil(now=3, kroend=0.9)
    wog.wateroil.add_corey_water(nw=2, krwend=0.2)
    wog.gasoil.add_corey_oil()
    wog.gasoil.add_corey_gas()
    if not doprint:
        len(wog.SWOF())
        len(wog.SGOF())
    else:
        print(wog.SWOF())
        print(wog.SGOF())


if __name__ == "__main__":
    print("Running in robust and slow mode:")
    print(
        timeit.timeit(
            stmt="benchme(fast=False)", setup="from benchme import benchme", number=100
        )
    )
    print("Running in fast mode:")
    print(
        timeit.timeit(
            stmt="benchme(fast=True)", setup="from benchme import benchme", number=100
        )
    )
