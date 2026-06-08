"""
Validation against NOAA's official published harmonic constants.

Solves one year of real hourly data at a few NOAA stations and checks that the
major constituents match NOAA's accepted constants (amplitude and Greenwich
phase). Uses the committed example data, so it runs offline.
"""

import json
import os

import numpy as np
import pytest

from utide import solve

DATA = os.path.join(os.path.dirname(__file__), os.pardir, "examples", "data")
NPZ = os.path.join(DATA, "noaa_hourly_2023.npz")
HARCON = os.path.join(DATA, "noaa_harcon.json")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(NPZ) and os.path.exists(HARCON)),
    reason="NOAA example data not present",
)


def test_matches_noaa_harmonic_constants():
    d = np.load(NPZ, allow_pickle=True)
    harcon = json.load(open(HARCON))
    t = d["t_days"].astype(float)
    X = d["levels"].astype(float)
    ids = [str(x) for x in d["ids"]]
    lats = d["lats"]
    # clearly semidiurnal, large-tide stations where the majors are well determined
    for sid in ["8410140", "8443970", "9414290"]:  # Eastport, Boston, San Francisco
        s = ids.index(sid)
        coef = solve(
            t,
            X[:, s],
            lat=float(lats[s]),
            method="ols",
            conf_int="none",
            epoch="2023-01-01",
            verbose=False,
        )
        i_of = {n: i for i, n in enumerate(coef["name"])}
        hc = harcon[sid]
        for c in ["M2", "S2", "N2", "K1", "O1"]:
            assert c in i_of and c in hc
            i = i_of[c]
            amp_noaa, phase_noaa = hc[c]
            assert abs(coef["A"][i] - amp_noaa) / amp_noaa < 0.07, (sid, c, "amplitude")
            dphi = abs((coef["g"][i] - phase_noaa + 180) % 360 - 180)
            assert dphi < 7.0, (sid, c, "phase")
