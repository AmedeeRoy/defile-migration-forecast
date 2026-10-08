"""Explore: the data and statistics behind defileViz's Explore page (#55).

Contained here on purpose, so the forecast does not grow into a tool for the Explore page:

- `export`: a raw aggregation of the defile-dataset release tables (daily totals, effort, taxa,
  reports), written as JSON by `scripts/build_explore.py`. No model processing.
- `profile`: the effort adjustment (time-of-day profile, coverage, annual index).
- `trend`: a taxon's smooth trend, season and phenology shift, and gap-filled annual totals; a
  GAM and a Gaussian-process variant, benchmarked by `scripts/benchmark_trend.py`.

The boundary runs one way. This package may import the forecast's shared pieces
(`src.data.counts` to read the release, `src.phenology.fit_ratio_surface`, `src.metrics`), but
nothing outside it imports from it, except the scripts that build or analyse the export
(`tests/test_explore.py` checks this). If something here becomes useful to the forecast (a
population trend, say), it reaches the forecast through a named file it reads, as
`species_doy_statistics.json` does today, with an entry in DECISIONS.md, never as an import.
"""
