# Adoption Parameters Cheat Sheet

bdssim offers two adoption models. This guide explains each setting in plain language.

## Bass Diffusion (`adoption.model: "bass"`)
- **`p` – Innovation spark**: how often a country adopts on its own. Higher values give earlier first movers.
- **`q` – Peer contagion**: the strength of copycat behaviour. Larger values speed up contagion once adoption starts.
- **`m` – Market potential**: maximum number of sovereign adopters in the scenario. Lower `m` saturates faster; higher `m` allows bigger cascades.

### Tuning suggestions
- If nothing ever adopts, try raising `p` to ~0.02 and `q` to ~0.4.
- Choose `m` close to the number of realistic candidates (e.g. 20 for G20).

## Threshold Network (`adoption.model: "threshold"`)
- **`default_theta` – Base reluctance**: average appetite to adopt. Lower is more eager.
- **`alpha_peer` – Peer pressure**: larger values mean neighbours matter more.
- **`alpha_momentum` – Price momentum boost**: positive values reward rising BTC prices.
- **`alpha_liquidity` – Liquidity boost**: higher values make deep markets more persuasive.
- **`policy_penalty` – Policy friction**: captures IMF/BIS doctrine headwinds. Reduce (or add relief steps) to ease constraints.
- **`bloc_relief_steps`**: list of `{bloc_share, penalty_reduction}` pairs that lower the penalty once a share of the bloc has adopted.
- **`min_days_between_adoptions`**: cool-down to avoid instant domino effects.

### Practical tips
- Lower `default_theta` and `policy_penalty`, or raise `alpha_peer`, to trigger cascades sooner.
- Use the `bdssim analyze` command to inspect centrality rankings and identify high-influence seed countries.
- Experiment interactively with `streamlit run bdssim/dashboard_arcade.py`; the sidebar sliders map directly to these parameters.

