# Narragansett Bay Hypoxia

Forecasting dissolved oxygen in Narragansett Bay, Rhode Island, from co-measured water-quality
indicators — with the goal of predicting **hypoxia**, the low-oxygen condition (below 5 mg/L) that
stresses marine life and drives fish kills.

By Siddharth Gupta.

---

## ⚠️ The data in this repository is synthetic

The real measurements come from the **Narragansett Bay Fixed-Site Monitoring Network** and are owned
by the [Rhode Island Department of Environmental Management](https://dem.ri.gov/), the Narragansett
Bay Commission, and the URI Graduate School of Oceanography. They are not mine to redistribute, so
they are not here.

Every data file in this repository was fabricated by [`make_sample_data.py`](make_sample_data.py):
same filenames, same columns, same dtypes, same value ranges, plausible seasonal structure — and not
one real observation. It exists so the notebooks run end to end for anyone who clones the repo.

**Nothing computed from these files is a finding about Narragansett Bay.** Any correlation, forecast
error, or hypoxia rate you see after re-running a notebook is a property of the random number
generator.

One exception worth being explicit about: **the PNG figures saved in the notebooks were rendered
during the original analysis, on the real data.** They are aggregate time-series and diagnostic plots
of the kind that appear in a published paper. Everything that carried actual records — interactive
Plotly figures (which serialize the underlying arrays), printed dataframes, `.describe()` tables —
was stripped. So the *pictures* are from the real study; the *numbers you can load* are not.

To work with the real data, contact RI DEM, or reach me at emailsiddha@gmail.com.

## Results

After a broad model sweep, **Prophet with extra regressors** worked best on the continuous stations
(the seasonal stations proved hard to model at all). On the real record it achieved an **average
5.02% error on two-week dissolved-oxygen forecasts**, using past DO together with past temperature,
salinity, chlorophyll, and pH.

That number comes from the original study. It is not reproducible from this repository.

## The data

Buoy sondes log every 15 minutes; these files are **daily means**. Records run 1995–2024, with heavy
missingness — most stations are seasonal deployments, and gaps are common even mid-season.

### Stations

| RI DEM ID | Location | Code |
|---|---|---|
| B2 | North of Prudence Island (Upper Bay) | NP |
| B3 | South of Conimicut Point | CP |
| B3W | Upper Bay Winter Station | UB |
| B4 | Bullock's Neck / Lower Providence River (below Fields Point WWTF) | BR |
| B6 | Mount View (mouth of Greenwich Bay) | MV |
| B7 | Quonset Point | QP |
| B10 | Cole River (Mt Hope Bay) | CR |
| B11 | Taunton River (Mt Hope Bay) | TR |
| B12 | Mount Hope Bay | MH |
| B13 | Poppasquash Point (Upper East Passage) | PP |
| B14 | Sally Rock (Mid-Greenwich Bay) | SR |
| F3 | T-Wharf (South of Prudence Island, East Passage) | TW |
| F4 | Phillipsdale (Seekonk River, below Bucklin WWTF) | PD |
| F5 | Greenwich Bay (near the mouth of Apponaug Cove) | GB |
| F7 | URI GSO Dock | GD |

Plus `HW`. `GD`, `TW`, and `PD` are year-round; the rest are seasonal. **`BR` is the only station
with a mid-depth probe** — everywhere else it is surface and bottom.

### Parameters

`Temp_C`, `Salinity_ppt`, `DO_mg/L`, `DO_pct`, `pH`, `Depth_m`, `Density_g/cm3`, `Chl_ug/L`,
`Turb_NTU`. Units are in the column names.

### Files

| File | Shape |
|---|---|
| `Daily_Means_Through_2022.xlsx` | the original workbook, one row per measurement |
| `Daily_Means_Through_2022.csv` | same, as CSV: `date, parameter, measure, site, depth, month, year` |
| `Daily_Means_Through_2022_cleaned_NEW.csv` | wide modeling table: `date, site, {bottom,surface}_{Chl_ug/L, DO_mg/L, Density_g/cm3, Depth_m, Salinity_ppt, Temp_C, pH}` |
| `Daily_Means_Through_2022_cleaned.csv` | an earlier copy of the above, read by `sktimestuff.ipynb` |
| `fixed_site_data.csv` | older extract with different naming: `Date, param, measure, site, depth, month, year`, where `param` is one of `Temp, Salinity, DO., DO.Conc, Depth, pH, Chl, Turb, SpCond, Density` |
| `serialized_model.json` | the fitted Prophet model. Note that Prophet embeds its full training frame in the `history` field — on the real data this file *is* a copy of the dataset, which is why the version here is fitted on synthetic input |
| `GD_forecast.csv` | Prophet cross-validation output: `ds, yhat, yhat_lower, yhat_upper, y, cutoff` |

### Cleaning rules

`cleaning.ipynb` pivots long → wide and drops:

- **`GB`** — too shallow
- **`CR`, `TR`** — too many missing values
- **`UB`, `HW`** — surface only, so no stratification pair
- **`BR` mid-depth** — not enough data
- **`Turb_NTU`** — barely collected
- **`DO_pct`** — target leakage; it's a deterministic function of the variable being predicted

leaving 11 stations, surface/bottom pairs, and columns named `{depth}_{parameter}`.

## Layout

Run order:

1. [`convert_to_csv.ipynb`](convert_to_csv.ipynb) — Excel → CSV
2. [`cleaning.ipynb`](cleaning.ipynb) — pivot, drop stations and leaky columns → `..._cleaned_NEW.csv`
3. [`quickvizofmissing.ipynb`](quickvizofmissing.ipynb), [`bad_imputation.ipynb`](bad_imputation.ipynb) — missingness maps and imputation attempts
4. Correlation models — [`init_corr_models.ipynb`](init_corr_models.ipynb),
   [`init_corr_models_new_data.ipynb`](init_corr_models_new_data.ipynb),
   [`sklearn correlation models.ipynb`](sklearn%20correlation%20models.ipynb),
   [`xgb_correlation_models.ipynb`](xgb_correlation_models.ipynb) (with SHAP attribution)
5. Time series — [`statsmodels_stuff.ipynb`](statsmodels_stuff.ipynb),
   [`autoarima.ipynb`](autoarima.ipynb), [`sktimestuff.ipynb`](sktimestuff.ipynb),
   [`darts_forecasting.ipynb`](darts_forecasting.ipynb), [`new_darts.ipynb`](new_darts.ipynb),
   [`xgb_timeseries.ipynb`](xgb_timeseries.ipynb), and the winner,
   [`prophet.ipynb`](prophet.ipynb)
6. Frontend — [`streamlit_wireframes.ipynb`](streamlit_wireframes.ipynb) →
   [`streamlit_stuff.py`](streamlit_stuff.py)

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python make_sample_data.py               # regenerate the synthetic data
python make_sample_data.py --no-model    # ... skipping the slow Prophet refit
streamlit run streamlit_stuff.py
```

The generator takes a few minutes; most of that is the Prophet cross-validation
that produces `serialized_model.json` and `GD_forecast.csv`. `--no-model` writes just the CSVs
and the workbook in well under a minute.

Python 3.11 or 3.12. The notebooks target the pandas 2.x API and break on pandas 3.

Notebooks read their inputs as bare relative paths, so run Jupyter from the repository root.

The app takes an optional password gate — set `APP_PASSWORD` in the environment or in
`.streamlit/secrets.toml`. Unset, the app is open, which is the sensible default when it is serving
fabricated data.

`requirements.txt` covers every library the notebooks import, including some heavy ones (`darts`,
`sktime`, `pmdarima`). The Streamlit app and the data generator need only the core set: `numpy`,
`pandas`, `prophet`, `plotly`, `openpyxl`, `xlsxwriter`, `streamlit`.

## License

Code is MIT — see [LICENSE](LICENSE). This covers the code only, and conveys no rights to the
underlying Fixed-Site Monitoring Network data, which is not distributed here.
