# PR #47: fresh checkpoint review

All 11 species were retrained on the current prepared release using seed 0, two CPU threads per process, the existing species hyperparameters, the existing `period` split, a 50-epoch limit and patience 10. Training years remain 1966–2025; the rebuilt pooled phenology uses available data through partial 2026. This keeps the phase 0 bug-fix scope of issue #18; it does not implement the phase 1 next-year split or training-only phenology.

The current release was built on 2026-10-14 at 19:14:42 UTC and records uncommitted source changes. Exact input and checkpoint hashes, training code revision and diagnostics are in [the manifest](pr47-retraining.json). Checkpoints, phenology and normalization are consistent: each checkpoint loads strictly, all per-run normalization parameters and year/day transforms equal the existing production normalization, and that file's committed serialization is retained. An offline prediction smoke test exercised the prediction dataloader and NetCDF export with cached 2025 weather relabelled to the forecast dates; no live forecast API was called.

## Review outcome

**Keep the PR in draft.** The artifacts are technically compatible, but this checkpoint set is not an accuracy improvement. Mean species skill versus phenology falls from 0.229 to 0.170; Hobby changes from positive skill to worse than phenology. Common Buzzard and Merlin also regress substantially. Do not merge or deploy these candidates automatically.

The comparison evaluates existing production artifacts and the fresh candidates on the same current held-out rows with the same rebuilt scoring baseline. Production predictions retain their original prior; candidates use the new prior. The artifacts have different training inputs, so this is a deployment comparison, not a controlled estimate of the layer-order change. These are single-seed results, with a pooled baseline that includes held-out years. The historical saturation example in the PR's original experiment is not reproduced by the current production checkpoints on these rows.

| Species                | Production skill | Fresh skill | Best epoch (zero-based) | Fresh max birds/hour | Fresh saturated daily outputs |
| ---------------------- | ---------------: | ----------: | ----------------------: | -------------------: | ----------------------------: |
| Black Kite             |            0.173 |       0.145 |                      27 |              280.708 |                             0 |
| Common Buzzard         |            0.462 |       0.308 |                       9 |              124.638 |                             0 |
| Common Kestrel         |            0.037 |       0.047 |                       5 |                4.997 |                             0 |
| Eurasian Hobby         |            0.255 |      -0.060 |                      16 |                0.918 |                             0 |
| Eurasian Sparrowhawk   |            0.287 |       0.285 |                      30 |               13.475 |                             0 |
| European Honey Buzzard |            0.439 |       0.394 |                      21 |              169.282 |                             0 |
| Hen Harrier            |            0.340 |       0.274 |                      11 |                0.267 |                             0 |
| Merlin                 |            0.220 |       0.094 |                      12 |                0.040 |                             0 |
| Osprey                 |            0.010 |       0.094 |                      28 |                1.933 |                             0 |
| Red Kite               |            0.159 |       0.156 |                      24 |              216.518 |                             0 |
| Western Marsh Harrier  |            0.141 |       0.138 |                      38 |                4.138 |                             0 |

Saturation means `out_d >= 1 - 1e-6` on held-out rows. All weights and predictions are finite, night predictions are zero where the prior is zero, hourly masks preserve survey durations, and all 11 offline prediction exports pass. Some fresh channels still have small running variances or appreciable effective BatchNorm gains; the absence of saturation here is not a guarantee for unseen weather.

## Required loading repair

The first attempted run exposed NaNs from full-day weather stops: the old hourly mask treated equal start/end clock hours as zero duration. The datamodule now wraps exposure across UTC midnight, retaining full-day and partial-hour intervals. Regression tests cover full-day weather zeros, a midnight crossing, and partial hours. The interrupted runs were discarded; all promoted candidates were trained from scratch after this repair.

## Reproduction

Build counts with `python scripts/build_counts.py --dataset ../defile-dataset/output`, then run `python scripts/build_phenology_stats.py`. Use the input hashes to verify the snapshot; copying a later output changes the experiment. For each species, run the existing training entry point with its experiment config, keeping separate run and normalization paths:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python src/train.py experiment=common_buzzard trainer=cpu seed=0 hydra.run.dir=logs/pr47/common_buzzard +data.transform_data_path=logs/pr47/common_buzzard/transform_data.pickle
```

The recorded sweep only disabled console progress/config/model-summary printing; training, saliency, testing and reporting settings were retained. Logs, per-run configs, metrics, normalization copies, test NetCDF files and PDF reports are under `logs/qa/pr47/retraining/current/` in the PR worktree. Validation: 56 offline tests pass; network tests are excluded.
