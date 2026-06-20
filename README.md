# Investment Decisions – Figures

Two Python scripts generate the report figures for case study.
Both run without arguments straight from the IDE
and export each figure as `.png`, `.pdf` and `.svg`.

- **`without_co2_cap.py`** – figures for the no-CO₂-limit run. 
- **`carbon_comparison.py`** – emissions comparison of a 3 Gt budget vs. no limit (without policies)

## Required data

CSV exports of the model run, one set per run folder:

- **`run_summary.csv`** – one row per run
- **`*_profitability_*.csv`** – one csv file per run

## Folder structure (default run)

```
data/
├── until2050_no_carbon_cap/          # input for without_co2_cap.py
│   ├── run_summary.csv
│   ├── *_profitability_*.csv
│   └── figures/                      # output
└── carbon-emission-comparison/       # input for carbon_comparison.py
    ├── with_cap/    (run_summary.csv + *_profitability_*.csv)
    ├── without_cap/ (run_summary.csv + *_profitability_*.csv)
    └── figures/                      # output
```
