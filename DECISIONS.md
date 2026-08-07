# Défilé migration forecast — decisions log

A running log of settled calls, kept so later work knows what's already been tried,
what worked, and what was rejected and why. Not a design spec, and deliberately not
tied to specific files or lines — those drift; read the code for that. See
`DEVELOPMENT.md` for what's still open.

## Model architecture

**The hourly shape sub-network could collapse to a flat, constant prediction** after an
earlier fix removed a hardcoded dawn/dusk zero-mask — confirmed directly on some seeds
after retraining. Root cause: the training loss only ever checked the *combined*
prediction's masked average, never the hourly shape on its own, so a flat hourly output
paired with a correctly-scaled overall magnitude could satisfy the loss without
learning any real diurnal shape.

Two loss-side penalties were tried and rejected in favour of an architectural fix:
penalizing raw night-hour output directly, and directly supervising shape against real
hour-by-hour survey data on the subset of dates that have it. Both raised the cost of
collapsing without removing the collapse basin itself, and the second only ever covered
a small fraction of dates.

**Landed instead: anchor the hourly network's default output to a smooth climatological
shape**, built from the existing day-of-year phenology baseline (with deep night forced
to zero from sun position rather than fitted, since there's essentially no real data to
fit against there), with the network free to override that default wherever real
weather evidence justifies it. At initialization the network reproduces the
climatological shape almost exactly, removing the flat/constant state that training
could fall into when it has little else to go on.

**First attempt at anchoring was wrong and rejected**: normalizing the hourly output
into a probability distribution over the day forced all magnitude information into the
daily sub-network alone, cutting the model's ability to predict large counts by roughly
7x in testing. Fixed by anchoring each hour's *default level* independently instead,
without forcing the 24 hours to compete for a fixed total — this keeps both
sub-networks contributing to magnitude, matching the intended split between daily-scale
and hourly-scale information.

**Confirmed with a 3-seed comparison against the previous architecture, on Common
Buzzard**: the collapse did not recur on any seed, and shape metrics stayed
consistently tight across seeds instead of varying widely as they did when a collapse
occurred. Overall magnitude was comparable to before; shape accuracy was slightly,
consistently a bit lower — a small tradeoff, accepted for now, worth revisiting (see
`DEVELOPMENT.md`).

## Training reproducibility

**Training is now fully deterministic.** The same seed and config used to sometimes
produce different outcomes across runs, which made it hard to tell whether a change
actually mattered or was just noise. Verified directly: repeated runs with the same
seed now produce identical results.
