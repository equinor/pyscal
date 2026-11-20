"""Compare specified and tabulated water critical saturation in one SWOF table."""

import io
import json

import pandas as pd

from pyscal import WaterOil


def critical_saturation_example() -> dict[str, float]:
    """Return diagnostics for a synthetic Corey curve; no simulator is run."""
    wateroil = WaterOil(swl=0.044, swcr=0.057, sorw=0.239814, h=0.01)
    wateroil.add_corey_water(nw=4)
    wateroil.add_corey_oil(now=2)

    # Audit the actual exported numbers, including formatting/monotonicity
    # handling, rather than assuming the in-memory table is what gets read.
    rows = [
        line
        for line in wateroil.SWOF().splitlines()
        if line.strip() and not line.startswith(("--", "SWOF", "/"))
    ]
    printed = pd.read_csv(
        io.StringIO("\n".join(rows)),
        sep=r"\s+",
        names=["SW", "KRW", "KROW", "PC"],
    )
    cutoff = 1e-6  # Illustrative, explicitly chosen; not read from a simulator.
    return {
        "specified_swcr": wateroil.swcr,
        "last_in_memory_zero_sw": float(
            wateroil.table.loc[wateroil.table["KRW"] == 0.0, "SW"].max()
        ),
        "last_printed_zero_sw": float(printed.loc[printed["KRW"] == 0.0, "SW"].max()),
        "cutoff": cutoff,
        "last_printed_sw_at_or_below_cutoff": float(
            printed.loc[printed["KRW"] <= cutoff, "SW"].max()
        ),
    }


if __name__ == "__main__":
    print(json.dumps(critical_saturation_example(), indent=2, allow_nan=False))
