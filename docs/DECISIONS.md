# Decisions

Append as calls are made. The point is that a decision made for a reason in
week one is not quietly reversed in week three because the reason was
forgotten.

## 2026-10-10 — Effect defined on the probability scale

`tau` is set directly as `P(Y=1|treated) - P(Y=1|untreated)` rather than as a
shift in the logit. A logit shift produces an effect that varies with the
baseline rate even when no heterogeneity was intended, which would make the
`heterogeneity=0` scenario dishonest.

## 2026-10-10 — Response and effect features are disjoint

Baseline conversion and treatment effect are driven by different features. If
they shared features, a response model would score well on uplift metrics for
the wrong reason and the central comparison in the README would be worthless.

## 2026-10-10 — Qini definition pinned and documented

`Q(k) = Y_t(k) - Y_c(k) * N_t(k)/N_c(k)`. Several incompatible definitions
circulate. The docstring in `uplift/metrics.py` states which is in use so
numbers here are not compared with numbers computed under another.

## 2026-10-10 — `qini_coefficient` requires an explicit reference

There is no agreed way to construct an optimal Qini curve from observed
outcomes alone, so the function refuses to guess and takes the reference
ranking as an argument. On real data this means no normalised coefficient is
reported, which is the honest position.

## 2026-10-10 — Summary ratios pooled before division

Averaging per-seed ratios across seeds is wrong when the denominator is near
zero: the mean is dominated by whichever seed had the smallest oracle area.
Raw metrics are averaged first, then the ratio is formed. The small-effect
scenario is where this matters and it changed the reported numbers materially.

## 2026-10-10 — Small-effect scenario reported as a null result

Ratios above 1.0 there are division by noise, not learners beating the
oracle. Reported as "nothing is measurable at this effect size" rather than
dropped, because the failure is the finding.

## 2026-10-10 — Hillstrom not committed

Downloaded on first run. Redistribution terms are the author's to set.

## 2026-10-10 — README numbers checked against results, not trusted

`scripts/check_readme_claims.py` parses the README table and compares every
cell with the summary recomputed from `results/synthetic_results.csv`. Two
cells were wrong when the check was first run. Prose drifting from the data
it claims to report is the most common way a repo of this kind embarrasses
its author, and it is cheap to make impossible.

## 2026-10-10 — Python floor is 3.10, development on 3.14

A portfolio repo is cloned and run by people on whatever interpreter they
already have, so the floor stays low. Nothing here needs syntax newer than
3.10. Development happens on 3.14 and CI tests 3.10, 3.12, 3.13 and 3.14, so
both ends of the claim are checked rather than assumed.

`tests/test_packaging.py` enforces it: every source file must parse under the
declared floor, and the dependency floors are checked against the APIs the
code actually calls. That caught a real bug — `numpy>=1.24` was declared while
`metrics.py` calls `np.trapezoid`, which only exists from numpy 2.0. The
declared floor would have installed a numpy that cannot run the code.

## 2026-10-10 — Tests importable without installing first

`pythonpath = ["."]` in the pytest config, so a bare `pytest` on a fresh
clone finds the package. Without it the first thing a reader sees is
`ModuleNotFoundError: No module named 'uplift'`, which reads as broken code
rather than an uninstalled package.

A root `conftest.py` was tried first and does not work: pytest inserts the
test file's own directory, not the rootdir, so `tests/` landed on `sys.path`
and the repo root did not. The `pythonpath` setting is the documented
mechanism.

## 2026-10-11 — Continuous outcomes rejected, not supported

`load_hillstrom(outcome="spend")` raises, and `BaseLearner._check` raises on
any outcome that is not 0/1. Previously the cast to `int` truncated spend
into dozens of classes and the classifiers fitted them without complaint.
Guarding was chosen over adding regression variants of the learners: spend
is zero for over 99% of customers, so it carries less signal than
conversion, which is already underpowered, and a second family of learners
would need its own synthetic validation to be worth trusting.

## 2026-10-11 — Profit charged on the treated arm's footing

`Q(k)` counts incremental conversions among the treated customers in the top
k, not among everyone in the top k. The Hillstrom profit figure therefore
charges the contact cost for the treated customers only
(`metrics.treated_at_k`). As first written it charged for the whole top k,
about twice as many contacts as the conversions were measured over, which
made every row lose money. The cost and value themselves are invented;
Hillstrom publishes neither.

## 2026-10-11 — Downloaded file validated before caching

A response without the expected columns is rejected rather than written to
`data/raw`. A moved or parked page still returns 200, and caching it left
every later run failing on a `KeyError` with no hint to fetch the file by
hand.

## 2026-10-11 — Hillstrom evaluated over 20 splits, not one

On a single 70/30 split the S-, T- and X-learners came out in a different
order depending on the seed, so any ordering quoted from one split would
have been a claim about the seed. The script now repeats the split over 20
seeds, writes one row per split and learner, and reports the mean, the
spread, and the paired per-split difference between each pair of learners.

The spread is a standard deviation across splits and is not presented as a
standard error or a confidence interval. The splits reuse the same 42,693
customers, so they are not independent and no significance test is run on
them. The paired differences carry the comparison instead: both learners are
scored on the same test customers, and the count of splits on which one
leads is reported as it is.

The average treatment effect is computed on the full sample rather than a
training split. It involves no model, so there is nothing to hold out for.

## Open

- Uplift trees not implemented. Deliberate scope cut for the first version.
- Validation against `causalml` reference implementations not yet written.
- Tick Size Pilot (randomised US equity tick-size experiment, 2016-2018) as a
  second real dataset. Assignment data is a single free file from FINRA;
  outcome data is the open question, and delisted tickers are the risk.
- Power analysis via subsampling: shrink the sample and show where the
  measurable effect size floor sits.
