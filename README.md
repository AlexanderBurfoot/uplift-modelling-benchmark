# Uplift modelling benchmark

Estimating who a treatment actually changes, and measuring when the estimate
can be trusted.

A standard A/B test returns one number: the average effect across everyone.
That number hides the thing a targeting decision needs, which is that most
populations contain people the treatment persuades, people who would have
converted regardless, people who never convert, and people the treatment
puts off. Uplift modelling estimates the effect per individual so those
groups can be separated.

The difficulty is that the per-individual effect is never observable. Each
person is either treated or not, so there is no ground truth to score
against, and uplift models are normally evaluated only on ranking quality.
**This repo adds a synthetic generator where the true per-person effect is
known exactly**, which makes estimation error measurable and turns "which
learner ranks best" into "which learner fails, when, and why".

## What is here

| Module | Contents |
|---|---|
| `uplift/dgp.py` | Synthetic randomised experiment with known individual effects |
| `uplift/learners.py` | S-, T- and X-learners, plus a response model as the comparison |
| `uplift/metrics.py` | Qini curve and area; PEHE and effect correlation as oracle metrics |
| `uplift/data.py` | Hillstrom email experiment loader |

The learners are written from their definitions rather than imported. The
response model is not an uplift model at all: it predicts who converts, which
is what gets built when nobody has thought about the difference, and showing
where it misleads is the most useful single result below.

## Results on synthetic data

20,000 individuals, 70/30 train-test split, averaged over three seeds.
`qini_vs_oracle` is Qini area as a fraction of the area achieved by ranking
on the true effect, so 1.0 is perfect ranking and 0 is no better than random.

| Scenario | Response | S-learner | T-learner | X-learner |
|---|---|---|---|---|
| Baseline (5pp effect, balanced arms) | 0.27 | 0.70 | 0.64 | **0.79** |
| Constant effect for everyone | −0.12 | 0.84 | 0.24 | 0.38 |
| Unbalanced arms (15% treated) | −0.04 | 0.49 | 0.46 | **0.52** |
| 25% sleeping dogs | 0.00 | **0.43** | 0.28 | 0.23 |
| Small effect (1pp) | 0.29 | 0.01 | 1.07 | 1.35 |

Reading these honestly:

**The response model is worse than useless for targeting.** In three of five
scenarios its Qini area is at or below zero, meaning targeting its top-ranked
customers captures no more incremental conversions than targeting at random,
and sometimes fewer. It ranks people by baseline conversion probability,
which is exactly the group that would have converted anyway.

**The X-learner leads where theory says it should**, on the baseline and with
unbalanced arms, but the margin over the S-learner on unbalanced arms is 0.03,
well inside seed-to-seed variation. Three seeds cannot support a claim that
one beats the other there.

**The S-learner is not the weak baseline it is usually described as.** It wins
on three of the five scenarios and has the lowest PEHE throughout. Its
reputation comes from regularised linear models dropping the treatment term;
with gradient boosting over a 5-percentage-point effect, that failure mode
does not appear.

**The small-effect row is a null result, not a win.** Ratios above 1.0 are not
a learner beating the oracle. With a 1-percentage-point effect the oracle's
own Qini area is 4.7 incremental conversions against a sampling standard
deviation of roughly 7, so the denominator is smaller than the noise and every
ratio in that row is division by noise. The correct conclusion is that at this
effect size and sample size nothing is measurable, and the normalised
coefficient should not be reported at all. Raw areas and a power calculation
are the honest output.

Full per-seed numbers are in `results/synthetic_results.csv`.

## Running it

Python 3.10 or later. Developed on 3.14; CI runs 3.10, 3.12, 3.13 and 3.14,
so the declared floor is tested rather than assumed.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                          # 98 tests
python scripts/run_synthetic.py # the table above
python scripts/run_hillstrom.py # real randomised data
```

The editable install matters: the tests import `uplift`, and bare `pytest`
without it will fail at collection.

Hillstrom downloads on first run to `data/raw` and is not committed.

## Scope

Everything here assumes randomised assignment. None of these learners
corrects for confounding, and applying them to observational data (ad
exposure logs, self-selected enrolments) produces confident nonsense. The
Qini definition used is stated in `uplift/metrics.py`, because several
incompatible ones circulate and comparing numbers across implementations
without checking which is in use is a common error.

Not included: propensity-score methods, difference-in-differences,
instrumental variables, or anything else aimed at observational data.
