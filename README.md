ï»¿# Bitcoin Debt Solution Simulation (bdssim)

This repository implements a nation-state Bitcoin reserve accumulation simulator with game-theoretic contagion between sovereign actors. The engine models market impact, adoption diffusion (Bass and Granovetter threshold), deterministic supply, treasury execution policy, accounting, reporting, and a Typer CLI.

## Quickstart

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[tests]"
pytest -q
bdssim run --config configs/base.yaml --out results/base
```

Outputs include CSV and Parquet timeseries, summary tables, configuration snapshots, and labelled PNG plots.

## Modules

- `bdssim.market`: square-root impact, venue mix, execution pacing, exogenous flows.
- `bdssim.adoption`: Bass diffusion and threshold-network contagion on a country graph.
- `bdssim.supply`: Bitcoin halving schedule, miner selling, effective float.
- `bdssim.treasury`: U.S. and sovereign policy schedulers and take-profit logic.
- `bdssim.accounting`: FIFO ledger, cost basis, realized/unrealized P&L, debt coverage.
- `bdssim.engine`: simulation loop, Monte Carlo, scenarios, state tracking.
- `bdssim.reporting`: Matplotlib plots, summary exports.
- `bdssim.cli`: Typer CLI commands (`run`, `mc`, `compare`, `plot`, `validate`).

## Disclaimer

This project is a research simulator. It is not financial advice, trading guidance, or legal policy recommendation. Replace illustrative data with calibrated datasets before decision-making and comply with applicable regulations.

## Interactive Dashboard
Run an interactive dashboard with Streamlit:

`ash
streamlit run bdssim/dashboard.py
`

Adjust parameters in the sidebar to explore market, policy, and adoption sensitivities. Outputs update live with charts and downloadable CSV results.


## Batch Scenario Report
Generate a Monte Carlo markdown report for multiple scenarios:

`ash
bdssim report configs/base.yaml configs/optimistic.yaml --runs 500 --out results/reports --as-pdf
`

The command runs the requested number of simulations per config and writes a timestamped Markdown summary covering p5/p50/p95 outcomes for price, coverage, holdings, drawdown, and adopters.


## Advanced Analytics
Generate network-aware diagnostics and recommended seed countries for a scenario:

`ash
bdssim analyze --config configs/base.yaml --out results/analysis --top-k 5
`

The analysis command runs the scenario, computes network centrality metrics, adoption cascade statistics, produces recommended sovereign seed sets, and writes nalysis.md alongside CSV artifacts (centrality ranking, adoption log).

## Live Price Integration
Fetch today's BTC/USD spot price when running simulations:

```bash
bdssim run --config configs/base.yaml --use-live-price
bdssim mc --config configs/base.yaml --runs 500 --use-live-price
```

The engine pulls data from CoinGecko and injects the price into the scenario before simulation. The Streamlit dashboard also exposes a "Use live BTC price" button for rapid calibration.

## Model Note
bdssim is a stylised simulation model. Replace the illustrative CSVs with calibrated data, validate assumptions (impact coefficients, adoption thresholds, supply haircuts), and treat outputs as scenario analysis rather than deterministic predictions.
## Mission Control Dashboard
For a streamlined, gamified dashboard where you pick allies and tune parameters with plain-English tooltips:

```bash
streamlit run bdssim/dashboard_arcade.py
```

You can fetch the live BTC price, set mission length, daily budget, and adoption mood ("innovation" and "peer contagion"), then immediately see the mission score, adoption curve, and recommended network targets.
## Adoption Parameters
See `docs/adoption_parameters.md` for a plain-English crib sheet covering the Bass diffusion knobs (`p`, `q`, `m`) and the threshold-network sliders (`default_theta`, `alpha_peer`, etc.). The arcade dashboard (`streamlit run bdssim/dashboard_arcade.py`) exposes the same controls with tooltips if you prefer to experiment visually.

### Playbooks & Cascades
Use staged playbooks to model regional waves:

```bash
bdssim run --config configs/base.yaml --playbook configs/playbooks/transatlantic.yaml --use-live-price
```

Or refer to the curated library with named keys:

```bash
bdssim playbooks
bdssim run --config configs/base.yaml --playbook-name transatlantic
```

Monte Carlo with stages + PDF export:

```bash
bdssim report configs/base.yaml configs/optimistic.yaml --runs 500 --playbook-name eastward --as-pdf
```

The arcade dashboard (`streamlit run bdssim/dashboard_arcade.py`) pulls the same playbook library—pick a preset or build custom waves with tooltips explaining each knob.


