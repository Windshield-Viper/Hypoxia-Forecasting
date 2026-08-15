#!/usr/bin/env python3
"""Generate the synthetic stand-in dataset this repository ships with.

The real Narragansett Bay Fixed-Site Monitoring Network data is owned by RI DEM,
the Narragansett Bay Commission and URI GSO, and is not ours to redistribute.
Every data file in this repo is produced by this script instead: same filenames,
same columns, same dtypes, same value ranges, plausible seasonal structure — and
not one real measurement.

The numbers are physically flavoured (dissolved-oxygen saturation follows a
temperature/salinity curve, bottom water goes hypoxic in late summer at the
upper-bay stations) so that the notebooks produce output that *looks* like the
real analysis. They are still fabricated. Nothing derived from them is a finding
about Narragansett Bay.

Usage:
    python make_sample_data.py            # writes every file listed in OUTPUTS
    python make_sample_data.py --no-model # skip the slow Prophet fit
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

SEED = 20250314

START = "1995-06-20"
END = "2024-12-31"

OUTPUTS = [
    "Daily_Means_Through_2022.csv",
    "Daily_Means_Through_2022.xlsx",
    "Daily_Means_Through_2022_cleaned_NEW.csv",
    "Daily_Means_Through_2022_cleaned.csv",
    "fixed_site_data.csv",
    "serialized_model.json",
    "GD_forecast.csv",
]

# ---------------------------------------------------------------------------
# Station configuration
#
# `upper` is position along the bay's estuarine gradient (1.0 = head of the
# Providence/Seekonk river system, 0.0 = mouth of the East Passage). It drives
# salinity, chlorophyll and the severity of summer bottom-water hypoxia.
#
# `season` is the months the sonde is deployed; year-round stations use None.
# ---------------------------------------------------------------------------
SUMMER = (5, 6, 7, 8, 9, 10)
WINTER = (11, 12, 1, 2, 3, 4)

STATIONS = {
    # code: (RI DEM id, depths,                 upper, bed_m, season, first_year, presence)
    # GD and TW are the year-round continuous records the forecasting notebooks
    # rely on; keep their presence high or Prophet/AutoARIMA have nothing to fit.
    # GD (the URI GSO dock sonde) is SURFACE ONLY — it has no bottom probe at
    # all. cleaning.ipynb drops UB and HW for exactly that reason but keeps GD
    # because it runs year round, and prophet.ipynb then models its surface DO.
    # Giving it a bottom series breaks that notebook's column bookkeeping.
    "GD": ("F7", ("surface",), 0.15, 6.0, None, 1995, 0.97),
    "TW": ("F3", ("surface", "bottom"), 0.30, 12.0, None, 2001, 0.90),
    "PD": ("F4", ("surface", "bottom"), 1.00, 3.0, None, 2001, 0.72),
    "BR": ("B4", ("surface", "mid", "bottom"), 0.90, 8.0, None, 2001, 0.42),
    "NP": ("B2", ("surface", "bottom"), 0.55, 8.0, SUMMER, 2001, 0.63),
    "CP": ("B3", ("surface", "bottom"), 0.70, 6.0, SUMMER, 2001, 0.63),
    "MV": ("B6", ("surface", "bottom"), 0.50, 4.0, SUMMER, 2001, 0.61),
    "QP": ("B7", ("surface", "bottom"), 0.25, 10.0, SUMMER, 2003, 0.60),
    "MH": ("B12", ("surface", "bottom"), 0.40, 7.0, SUMMER, 2003, 0.59),
    "PP": ("B13", ("surface", "bottom"), 0.35, 9.0, SUMMER, 2003, 0.62),
    "SR": ("B14", ("surface", "bottom"), 0.55, 3.0, SUMMER, 2003, 0.60),
    "UB": ("B3W", ("surface",), 0.80, 5.0, WINTER, 2005, 0.50),
    "GB": ("F5", ("surface",), 0.60, 2.5, SUMMER, 2003, 0.48),
    "HW": ("", ("surface",), 0.45, 4.0, SUMMER, 2005, 0.40),
    "CR": ("B10", ("surface", "bottom"), 0.45, 4.0, SUMMER, 2005, 0.30),
    "TR": ("B11", ("surface", "bottom"), 0.60, 5.0, SUMMER, 2005, 0.29),
}

PARAMETERS = [
    "Temp_C",
    "Salinity_ppt",
    "DO_pct",
    "DO_mg/L",
    "pH",
    "Depth_m",
    "Density_g/cm3",
    "Chl_ug/L",
    "Turb_NTU",
]

# Chlorophyll was not logged at every station. Note this is missingness by
# STATION, not by year: a station that logs chlorophyll logs it on every day it
# collected anything. Gating it by year instead produces rows that carry DO but
# no Chl, and the regressors in init_corr_models.ipynb reject partial rows.
CHL_STATIONS = {"GD", "TW", "PD", "BR", "NP"}
TURB_STATIONS = {"BR", "PD"}
TURB_YEARS = (2018, 2019, 2020)

# The older fixed-site extract covers fewer deployment days than the current one.
FIXED_SITE_KEEP = 0.55

# prophet.ipynb cell 10 hardcodes how many observations follow its 2024-01-01
# training cutoff -- `make_future_dataframe(periods=336)` for GD -- and then
# fills the regressor columns with `do_series[col].values`, a raw positional
# assignment. If GD's 2024 observation count is anything other than 336 that
# line raises a length mismatch, so pin the count rather than leave it to the
# presence draw. (The notebook's comment records 365 for TW and 351 for PD,
# used by the commented-out variants of that cell.)
PINNED_YEAR_COUNT = {"GD": (2024, 336)}


def ar1(rng, n, phi, sigma):
    """AR(1) noise, so the series wanders instead of jittering."""
    eps = rng.normal(0.0, sigma, n)
    out = np.empty(n)
    out[0] = eps[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def do_saturation(temp_c, sal_ppt):
    """Dissolved-oxygen solubility (mg/L), Benson-Krause shape with a salinity term."""
    fresh = 14.62 - 0.3898 * temp_c + 0.006969 * temp_c**2 - 0.00005897 * temp_c**3
    return fresh * (1.0 - 0.0054 * sal_ppt)


def water_density(temp_c, sal_ppt):
    """Seawater density in g/cm3, tuned to the ~1.016-1.022 range of the real file."""
    return 1.0 + 0.00077 * sal_ppt - 0.0000072 * (temp_c - 4.0) ** 2


def station_frame(code, cfg, dates, rng):
    """One station: a daily record per deployed depth, masked to days it collected."""
    _dem, depths, upper, bed_m, season, first_year, presence = cfg

    n = len(dates)
    doy = dates.dayofyear.to_numpy()
    year = dates.year.to_numpy()
    month = dates.month.to_numpy()

    # Annual cycle: water is warmest in early August, coldest in early February.
    warm = np.sin(2 * np.pi * (doy - 115) / 365.25)
    # Stratification/respiration season, peaking late summer.
    strat = np.clip(np.sin(2 * np.pi * (doy - 150) / 365.25), 0, None) ** 1.5

    # Was the sonde in the water that day? Decided ONCE per station-day and
    # shared by every probe on the station, because a deployment carries its
    # surface and bottom sensors together — in the real record the two depths
    # co-occur 94-98% of the time. Masking each depth independently instead
    # drops that to ~70% and silently destroys the stratification pairs the
    # cleaning notebook is built around.
    day_keep = year >= first_year
    if season is not None:
        day_keep = day_keep & np.isin(month, season)
    day_keep = day_keep & (rng.random(n) < presence)
    outage = rng.random(n) < 0.0016
    outage = pd.Series(outage).rolling(21, min_periods=1).max().to_numpy().astype(bool)
    day_keep = day_keep & ~outage

    pin = PINNED_YEAR_COUNT.get(code)
    if pin is not None:
        pin_year, target = pin
        in_year = np.flatnonzero(year == pin_year)
        kept = in_year[day_keep[in_year]]
        if len(kept) > target:
            day_keep[rng.choice(kept, len(kept) - target, replace=False)] = False
        elif len(kept) < target:
            spare = in_year[~day_keep[in_year]]
            day_keep[rng.choice(spare, target - len(kept), replace=False)] = True

    frames = []
    for depth in depths:
        is_surface = depth == "surface"
        frac = {"surface": 0.08, "mid": 0.5, "bottom": 0.92}[depth]

        # --- temperature ------------------------------------------------
        temp = 12.4 + 9.6 * warm + ar1(rng, n, 0.93, 0.55)
        temp += 0.6 * upper                      # shallow upper bay runs warmer
        temp += 0.012 * (year - 2010)            # slight warming trend
        if not is_surface:
            temp -= strat * (1.4 + 2.6 * (1 - upper)) * frac
        temp = np.clip(temp, -2.05, 28.98)

        # --- salinity ---------------------------------------------------
        sal = 31.6 - 9.0 * upper - 2.2 * np.clip(-warm, 0, None) + ar1(rng, n, 0.9, 0.9)
        sal += (1.0 - frac) * -0.4 + frac * 1.5 * upper   # saltier at depth
        sal = np.clip(sal, 0.11, 47.88)

        # --- dissolved oxygen -------------------------------------------
        sat = do_saturation(temp, sal)
        if is_surface:
            ratio = 0.94 + 0.10 * strat + ar1(rng, n, 0.85, 0.05)
        else:
            drawdown = strat * upper * rng.uniform(0.35, 0.95) * frac
            ratio = 0.90 - drawdown + ar1(rng, n, 0.88, 0.06)
        ratio = np.clip(ratio, 0.0002, 2.6)
        do_mg = np.clip(sat * ratio, 0.002, 22.94)
        do_pct = np.clip(do_mg / np.maximum(sat, 0.1) * 100.0, 0.03, 307.8)

        # --- the rest ---------------------------------------------------
        ph = np.clip(7.86 + 0.22 * warm - 0.30 * upper + ar1(rng, n, 0.9, 0.09), 6.4, 8.9)
        density = water_density(temp, sal)
        depth_m = np.clip(
            bed_m * frac + rng.normal(0, 0.35, n) + 0.55 * np.sin(2 * np.pi * doy / 14.7),
            -0.14, 15.45,
        )
        chl = np.clip(
            np.exp(rng.normal(1.0 + 1.1 * upper + 0.9 * strat, 0.72, n)), -0.18, 279.6
        )
        turb = np.clip(np.exp(rng.normal(0.5, 0.9, n)), 0.0, 37.58)

        # --- sensor artefacts, exactly the kind the cleaning notebooks meet
        n_bad = max(1, int(0.0004 * n))
        for arr, bad in ((ph, 33.94), (ph, 0.63), (density, 1024.0), (do_pct, 307.8)):
            arr[rng.integers(0, n, n_bad)] = bad

        # A little independent per-probe dropout on top of the shared day mask,
        # so the two depths agree ~96% of the time rather than always. Stations
        # with a single probe skip it: there is no second sensor to disagree
        # with, and it would perturb the pinned counts above.
        probe_keep = day_keep if len(depths) == 1 else day_keep & (rng.random(n) < 0.965)

        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "site": code,
                    "depth": depth,
                    "month": month,
                    "year": year,
                    "Temp_C": temp.round(6),
                    "Salinity_ppt": sal.round(6),
                    "DO_pct": do_pct.round(6),
                    "DO_mg/L": do_mg.round(6),
                    "pH": ph.round(6),
                    "Depth_m": depth_m.round(6),
                    "Density_g/cm3": density.round(9),
                    "Chl_ug/L": chl.round(6) if code in CHL_STATIONS else np.nan,
                    "Turb_NTU": np.where(
                        np.isin(year, TURB_YEARS) if code in TURB_STATIONS else False,
                        turb.round(6),
                        np.nan,
                    ),
                }
            )[probe_keep]
        )

    return pd.concat(frames, ignore_index=True)


def build_long(rng):
    """The wide-per-station frames, melted into the long shape of the real file."""
    dates = pd.date_range(START, END, freq="D")
    parts = [station_frame(code, cfg, dates, rng) for code, cfg in STATIONS.items()]
    wide = pd.concat(parts, ignore_index=True)

    long = wide.melt(
        id_vars=["date", "site", "depth", "month", "year"],
        value_vars=PARAMETERS,
        var_name="parameter",
        value_name="measure",
    ).dropna(subset=["measure"])

    long = long[["date", "parameter", "measure", "site", "depth", "month", "year"]]
    long = long.sort_values(["date", "site", "depth", "parameter"], kind="stable")
    return long.reset_index(drop=True)


def run_cleaning(long):
    """cleaning.ipynb, verbatim in logic: pivot, drop stations, pivot again."""
    df = long.copy()
    df["date"] = pd.to_datetime(df["date"])
    piv = df.pivot_table(
        index=["date", "site", "depth"], columns="parameter", values="measure", aggfunc="mean"
    ).reset_index()

    piv = piv[~piv["site"].str.contains("GB")]
    piv = piv[~piv["site"].str.contains("CR|TR")]
    piv = piv[~piv["site"].str.contains("UB")]
    piv = piv[~piv["site"].str.contains("HW")]
    piv = piv[~(piv["site"] == "BR") | (piv["depth"] != "mid")]
    piv = piv.drop(columns=["Turb_NTU", "DO_pct"])

    new = piv.pivot_table(
        index=["date", "site"],
        columns="depth",
        values=["DO_mg/L", "Density_g/cm3", "Depth_m", "Salinity_ppt", "Temp_C", "pH", "Chl_ug/L"],
    )
    new.columns = [f"{pos}_{col}" for col, pos in new.columns]
    new = new.reset_index().sort_values(by=["site"])
    return new


def build_fixed_site(long, rng):
    """The older extract: different column names, different parameter vocabulary."""
    rename = {
        "Temp_C": "Temp",
        "Salinity_ppt": "Salinity",
        "DO_pct": "DO.",
        "DO_mg/L": "DO.Conc",
        "Depth_m": "Depth",
        "pH": "pH",
        "Chl_ug/L": "Chl",
        "Turb_NTU": "Turb",
        "Density_g/cm3": "Density",
    }
    sites = ["BR", "GD", "NP", "CP", "MV", "PD", "PP", "TW", "MH", "QP", "SR", "GB"]

    df = long[
        (long["site"].isin(sites))
        & (long["date"] >= "2001-05-12")
        & (long["date"] <= "2019-12-31")
    ].copy()
    df["param"] = df["parameter"].map(rename)
    df = df.dropna(subset=["param"])

    # This extract predates the current one and covers fewer deployment days.
    # Sample whole station-days so every parameter stays present together.
    days = df[["site", "depth", "date"]].drop_duplicates()
    days = days[rng.random(len(days)) < FIXED_SITE_KEEP]
    df = df.merge(days, on=["site", "depth", "date"], how="inner")

    # Specific conductance only exists in this older extract.
    sal = df[df["param"] == "Salinity"]
    cond = sal.copy()
    cond["param"] = "SpCond"
    cond["measure"] = (sal["measure"].to_numpy() * 1.72
                       + rng.normal(0, 0.4, len(sal))).round(6)
    df = pd.concat([df, cond], ignore_index=True)

    df = df.rename(columns={"date": "Date"})
    df = df[["Date", "param", "measure", "site", "depth", "month", "year"]]
    df = df.sort_values(["site", "depth", "param", "Date"], kind="stable")
    return df.reset_index(drop=True)


def fit_prophet(cleaned):
    """prophet.ipynb's GD pipeline, so the shipped model file is fitted on fake data."""
    from prophet import Prophet
    from prophet.diagnostics import cross_validation
    from prophet.serialize import model_to_json

    df_new = cleaned.copy()
    df_new["date"] = pd.to_datetime(df_new["date"])
    df_new = df_new.sort_values(by=["site", "date"])

    df = df_new[df_new["site"] == "GD"].dropna(subset=["surface_DO_mg/L"])
    df = df.drop(columns=["bottom_Density_g/cm3", "surface_Density_g/cm3", "surface_Depth_m"])
    df = df.set_index(["date", "site"]).interpolate(
        method="linear", limit_direction="both"
    ).reset_index()

    shift_cols = [c for c in df.columns if c not in ("date", "bottom_DO_mg/L", "site")]
    shifted = df.copy()
    shifted[shift_cols] = shifted[shift_cols].shift(14)
    shifted = shifted.iloc[14:].reset_index(drop=True)

    series = shifted.set_index("date").drop(columns=["site"])
    series = series.drop(columns=[c for c in series.columns if "bottom" in c]).dropna()

    train = series[series.index < "2024-01-01"]
    new_df = train.reset_index(drop=True)
    new_df["ds"] = train.index
    new_df = new_df.rename(columns={"surface_DO_mg/L": "y"})

    m = Prophet(changepoint_prior_scale=0.001, seasonality_prior_scale=0.01)
    for column in new_df.columns:
        if column not in ("ds", "y"):
            m.add_regressor(column)
    m.fit(new_df)

    with open("serialized_model.json", "w") as fout:
        fout.write(model_to_json(m))

    df_cv = cross_validation(
        m, initial="730 days", period="28 days", horizon="14 days", parallel="processes"
    )
    df_cv.to_csv("GD_forecast.csv", index=False)
    return len(df_cv)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-model", action="store_true",
                    help="skip the Prophet fit (serialized_model.json / GD_forecast.csv)")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)

    print("building long-format daily means ...")
    long = build_long(rng)
    long.to_csv("Daily_Means_Through_2022.csv", index=False)
    print(f"  Daily_Means_Through_2022.csv          {len(long):>9,} rows")

    print("writing the Excel original ...")
    with pd.ExcelWriter(
        "Daily_Means_Through_2022.xlsx", engine="xlsxwriter",
        engine_kwargs={"options": {"constant_memory": True}},
    ) as xl:
        long.to_excel(xl, sheet_name="Daily_Means", index=False)
        # No document properties: the real workbook carried its authors' names.
        xl.book.set_properties({"title": "Synthetic sample data", "author": "",
                                "company": "", "comments": "Generated by make_sample_data.py"})

    print("running the cleaning pipeline ...")
    cleaned = run_cleaning(long)
    cleaned.to_csv("Daily_Means_Through_2022_cleaned_NEW.csv", index=False)
    # sktimestuff.ipynb reads the older filename; ship both so it runs.
    cleaned.to_csv("Daily_Means_Through_2022_cleaned.csv", index=False)
    print(f"  Daily_Means_Through_2022_cleaned_NEW.csv {len(cleaned):>6,} rows")

    print("building the older fixed-site extract ...")
    fixed = build_fixed_site(long, rng)
    fixed.to_csv("fixed_site_data.csv", index=False)
    print(f"  fixed_site_data.csv                   {len(fixed):>9,} rows")

    if args.no_model:
        print("skipping Prophet (--no-model)")
        return

    print("fitting Prophet on the synthetic GD series ...")
    n_cv = fit_prophet(cleaned)
    print(f"  GD_forecast.csv                       {n_cv:>9,} rows")
    print("done.")


if __name__ == "__main__":
    main()
