# BatchNorm ordering: scope of #47

Data processing is separate: #58 is merged, and #59 fixes full-day exposure and unresolved species-day targets. This PR is stacked on #59 and changes only the daily network's layer order, its regression test and documentation. It does not change datasets, priors, normalization or production checkpoints.

Conv → ReLU → BatchNorm can give a channel with varying negative pre-activations a running variance of zero because ReLU removes that variation before BatchNorm sees it. Conv → BatchNorm → ReLU preserves that variance. The regression test exercises this mechanism directly without training a model.

The previous comparison changed the data, prior and architecture together. Mean skill of 0.229 versus 0.170 is a combined deployment result, not evidence that either the data refresh or BatchNorm ordering caused a regression. Existing experiment artifacts are preserved in prior commits and local reports; they are not candidates in this PR's final diff.

Data correctness does not require retraining. To estimate the BatchNorm effect later, compare fresh old-order and new-order runs using identical prepared counts, prior, normalization, split, seed, hyperparameters and CPU thread count. That missing control has not been run. Production compatibility still requires matching reviewed checkpoints before merging this architecture change; keep #47 in draft meanwhile.
