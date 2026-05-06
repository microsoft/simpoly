# Tg analysis: cooling-scan + 21-step protocols

Fit polymer glass-transition temperatures (Tg) from aggregated MD outputs
using the self-consistent Patrone equation, with a parametric bootstrap
for the 21-step protocol.

## Running

From the repo root, with `simpoly` importable (pip-installed or via
`PYTHONPATH=src`):

```bash
# Cooling protocol (constant cooling-rate scans)
python src/simpoly/poly_arena/analysis/cli/cooling_protocol.py \
  --inputs <stages_csv> [<stages_csv> ...] \
  --out <out_dir>

# 21-step protocol (parametric bootstrap of the SC fit)
python src/simpoly/poly_arena/analysis/cli/21step_protocol.py \
  --stages <stages_csv> \
  --n-samples 500 --n-workers 16 \
  --out <out_dir>
```

Each `<out_dir>` contains:

- `tg_per_seed.csv`, per-seed fit results (cooling).
- `tg_results.csv`, per-polymer Tg with experimental delta / MAE.
- 21-step runs additionally emit bootstrap mean / std / Student-t SEM.
