# Analysis Workflow

1. **Calibration** – Update `configs/` and `data/` with the latest market/liquidity priors. Use the notebooks for exploratory calibration and parameter validation.
2. **Interactive exploration** – Launch `streamlit run bdssim/dashboard.py` to stress-test execution, adoption, and policy settings interactively.
3. **Focused analytics** – Run `bdssim analyze` for any scenario to obtain network centrality rankings, cascade diagnostics, and recommended sovereign seed sets.
4. **Monte Carlo research runs** – Execute `bdssim report` with multiple scenarios and hundreds/thousands of runs to compile percentile summaries suitable for publication.
5. **Document findings** – Combine the generated Markdown reports, CSV tables, and plots into the final research narrative, citing sources listed in `README.md` and notebooks.
6. **Publish** – Use `bdssim report ... --as-pdf` to generate archive-quality PDFs alongside Markdown summaries for distribution.
7. **Mission Control** – For a gamified exploration, run `streamlit run bdssim/dashboard_arcade.py` to recruit allies, tweak parameters in plain English, and instantly view mission scores.

7. **Stage Cascades** – Use the curated playbook index (`bdssim playbooks`) or your own YAML files in `configs/playbooks/`, then run them via `--playbook-name` / `--playbook` or experiment in the arcade dashboard.


