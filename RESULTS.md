Experimental Results

This file is the numerical record for the experiments summarized in the
main README. The goal here is completeness rather than another project
introduction: model selection, held-out policy values, paired
comparisons, seed sensitivity, optimization results, robustness checks,
and numerical controls are collected below.

1. Experimental protocol

The base contract is a one-year Bermudan arithmetic-average Asian put
simulated under risk-neutral geometric Brownian motion:

spot $$S_0=100$$;

strike $$K=100$$;

risk-free rate $$r=5%$$;

volatility $$\sigma=20%$$;

50 future exercise dates;

running average includes time zero.

Each model receives 20,000 training paths. Model configurations are
selected using 50,000 validation paths before final policy evaluation.
An independent 20,000-path quadratic teacher policy supplies targets for
the fixed-target prediction experiment. Separate 50,000-path samples are
used for prediction testing and full-policy evaluation.

The experiment imports the project's original simulation and
option-policy code. The original Asian and vanilla baselines, along with
pathwise adapter equivalence, are checked before the extended ML
experiments run.

The hold-to-expiration Asian value on the final policy-test sample is
$3.3636, with an approximate 95% interval of [$3.3174,
$3.4098].

2. Full Asian exercise policies

The first comparison uses training seed 42. Every policy is evaluated on
the same 50,000 unseen paths, which allows pathwise paired
comparisons against the quadratic baseline.

| Model | Test value | Value 95% interval | Gain vs. quadratic | Paired gain 95% interval | Early exercise | Train time |
|---|---:|---:|---:|---:|---:|---:|
| Quadratic | $3.9531 | [$3.9073, $3.9989] | $0.0000 | [$0.0000, $0.0000] | 40.7% | 0.10 s |
| Ridge | $3.9530 | [$3.9073, $3.9987] | -$0.0000 | [-$0.0033, $0.0032] | 40.8% | 0.27 s |
| Random forest | $3.9164 | [$3.8715, $3.9614] | -$0.0366 | [-$0.0437, -$0.0296] | 42.1% | 12.60 s |
| Gradient boosting | $3.9281 | [$3.8828, $3.9734] | -$0.0250 | [-$0.0318, -$0.0182] | 40.5% | 4.17 s |

For this seed, ridge is effectively indistinguishable from quadratic.
The neural network is also close: its paired interval includes zero, so
this experiment does not show a reliable difference from the quadratic
policy. The selected forest and boosting policies are lower on these
paths.

This should not be read as a universal ranking of model families.
Configuration selection used a finite validation search, and the models
were selected using a controlled prediction criterion rather than
independently tuned to maximize final policy value.



3. Training-seed sensitivity

To check whether the result was driven by one training sample, each
model family was retrained using seeds 42, 43, and 44 and evaluated on
the shared test sample.

| Model | Seed 42 | Seed 43 | Seed 44 | Mean across policies |
|---|---:|---:|---:|---:|
| Quadratic | $3.9531 | $3.9587 | $3.9565 | $3.9561 |
| Ridge | $3.9530 | $3.9583 | $3.9566 | $3.9560 |
| Random forest | $3.9164 | $3.9167 | $3.9159 | $3.9163 |
| Gradient boosting | $3.9281 | $3.9300 | $3.9290 | $3.9290 |
| Neural network | $3.9481 | $3.9521 | $3.9442 | $3.9482 |

The mean across the three trained policies is descriptive. These are
not three independent test sets, so the uncertainty should not be
divided by $$\sqrt{3}$$. Seed-specific paired intervals are stored
in results_ml/policy_test.csv.

The broad pattern is stable across these three training samples:
quadratic and ridge remain extremely close, the neural policy is nearby,
and the selected tree ensembles are lower.

4. Feature ablation: what information does the policy need?

This was the largest effect in the project.

The immediate payoff of an Asian put is determined by the running
average, but the current spot price contains information about how that
average may evolve. I therefore compared policies using only the running
average with policies using both spot and running average.

The same selected model-family settings and seed 42 are used on both
sides of the comparison.

| Model | Average only | Spot + average | Gain | Paired 95% interval | Average-only early exercise |
|---|---:|---:|---:|---:|---:|
| Quadratic | $3.3632 | $3.9531 | **$0.5899** | **[$0.5739, $0.6058]** | **6.6%** |


For both model families, adding spot price produces a much larger change
than switching among the full-state model classes in the base
experiment.

This is the main empirical takeaway: for this problem and training
setup, representing the state correctly mattered more than increasing
model complexity.



The figure shows policy outputs at sampled observed states. It should
not be interpreted as comparison with a known optimal exercise boundary,
because the optimal boundary is unavailable.

5. Prediction accuracy on a common target

Policy value is the final quantity of interest, but comparing regressors
inside a backward exercise algorithm is messy: changing today's exercise
decision changes what happens later.

To create a cleaner supervised-learning comparison, all candidate models
also predict the same target at date 25. Inputs are the current spot
and running average, conditional on the option being in the money. The
target is the discounted payoff obtained by following an independently
trained quadratic teacher policy from that point onward.

| Model | Selected configuration | Validation MSE | Test MSE |
|---|---|---:|---:|
| Quadratic | `quadratic` | 4.9821 | 4.9609 |
| Ridge | `ridge_d3_a1.0` | 4.9804 | 4.9605 |
| Random forest | `forest_d8` | 5.1073 | 5.1258 |
| Gradient boosting | `boost_l15` | 5.0747 | 5.0756 |
| Neural network | `neural_w32_adam_wd0.0001` | 4.9765 | 4.9705 |




These targets still contain future-path noise and are generated by a
teacher policy. They are not exact optimal continuation values.

The selected configurations were chosen by single-date validation MSE
and then reused across exercise dates. They were not separately tuned to
maximize full-policy validation value. The comparison therefore
represents the implemented, finite search rather than the best
attainable version of each model family.

A useful distinction emerged here: lower continuation-value MSE does not
automatically imply higher policy value. Errors close to the exercise
boundary can change a stopping decision, while even larger errors far
from the boundary may leave the decision unchanged.

6. Neural-network training

The neural model uses spot and running average as inputs, two hidden
layers with ReLU activations, and one continuation-value output.
Training uses mini-batches, mean-squared error, train-only feature
scaling, and a validation split for early stopping.

The selected base configuration is neural_w32_adam_wd0.0001.



Neural policy models reserve 15% of their supplied training observations
for early stopping, while the non-neural families use all supplied
observations. This difference should be kept in mind when comparing
training procedures.

7. Manual optimization experiment

Batch GD, mini-batch SGD, momentum, and Nesterov were implemented from
scratch for the same regularized quadratic continuation-regression
objective.

All four methods use:

standardized quadratic features;

ridge penalty $$\lambda=0.001$$;

learning rate $$0.03$$;

batch size 256 for SGD, momentum, and Nesterov.

The analytic gradient was checked numerically, and the closed-form ridge
solution provides a reference objective.


| Method | Final objective | Gap to optimum | Updates | Time |
|---|---:|---:|---:|---:|
| GD | 2.647239 | 0.108052 | 400 | 0.125 s |
| SGD | 2.550227 | 0.011040 | 14,800 | 0.445 s |
| Momentum | 2.542132 | 0.002945 | 14,800 | 0.432 s |
| Nesterov | 2.543284 | 0.004097 | 14,800 | 0.474 s |



Equal epoch counts do not imply equal update counts because GD uses the
full dataset while the other methods update once per mini-batch.

Additional scaling and learning-rate runs are stored in
optimization.csv. Each run should be compared with its own closed-form
objective because standardizing polynomial features changes the
coordinate meaning of ridge regularization.

The optimizer implementations pass the analytic-gradient check and a
well-conditioned convergence test. The continuation dataset is more
ill-conditioned, so finite-budget objective gaps need not vanish.

8. Retrained robustness checks

To see whether the base result was peculiar to one contract setting, I
froze the selected model settings and retrained the models under
three parameter changes.

These are retraining experiments, not transfer tests of one fixed
pretrained model. One training seed is used per condition.

| Scenario | Model | Policy value | Gain vs. quadratic | Paired gain 95% interval |
|---|---|---:|---:|---:|
| Low volatility | Quadratic | $1.5617 | $0.0000 | [$0.0000, $0.0000] |
| Low volatility | Ridge | $1.5593 | -$0.0024 | [-$0.0046, -$0.0001] |
| Low volatility | Neural | $1.5624 | $0.0007 | [-$0.0023, $0.0038] |
| High volatility | Quadratic | $7.4636 | $0.0000 | [$0.0000, $0.0000] |
| High volatility | Ridge | $7.4682 | $0.0046 | [-$0.0004, $0.0096] |
| High volatility | Neural | $7.4556 | -$0.0080 | [-$0.0167, $0.0007] |
| Higher strike | Quadratic | $12.0170 | $0.0000 | [$0.0000, $0.0000] |
| Higher strike | Ridge | $12.0191 | $0.0021 | [-$0.0044, $0.0086] |

The settings are frozen from the base scenario, so these checks test how
the chosen workflows behave after retraining under changed contract
parameters. They do not establish general superiority of one model
family.

9. Ordinary-put numerical control

The repository retains a simpler ordinary-put experiment because it
provides numerical references that are unavailable for the Asian
early-exercise problem.

Learned policies

| Model | Policy value | Approximate 95% interval |
|---|---:|---:|
| Quadratic | $6.0228 | [$5.9601, $6.0854] |
| Random forest | $5.9198 | [$5.8562, $5.9833] |
| Neural network | $6.0372 | [$5.9760, $6.0984] |

### Pricing references

- Matched-schedule tree, 500 steps: **$6.0773**
- Matched-schedule tree, 1,000 steps: **$6.0781**
- European Black--Scholes value: **$5.5735**
- European Monte Carlo value: **$5.5416**
- European Monte Carlo 95% interval: **[$5.4659, $5.6173]**

The tree values are numerical approximations. A Monte Carlo policy
estimate can occasionally lie above a numerical tree reference because
both Monte Carlo error and grid error are present.

10. Statistical and experimental qualifications

The reported 95% intervals quantify evaluation-path uncertainty
conditional on a trained policy. They do not capture all training
uncertainty.

Other qualifications:

model comparisons are exploratory and use no multiple-comparison
adjustment;

the three training seeds share one test sample;

the neural models reserve part of their supplied training
observations for early stopping;

learning curves use one nested sample order;

runtime includes Python and library overhead and depends on
hardware;

the first PyTorch fit includes startup overhead;

the hyperparameter search is finite rather than exhaustive;

the exact optimal Asian price and exercise regions remain unknown.

The results therefore describe this simulator, these contract settings,
and the implemented training budgets. They are not evidence of market
profitability.

11. Reproducibility

The full experiment suite completed in 5.85 minutes on the recorded
environment.

Run the numerical and behavioral checks with:

python run.py check

Run the complete experiment with:

python run.py full

A full run writes numerical outputs and figures to a new timestamped
directory under runs/.

Six numerical/behavioral checks passed, along with reproduction of the
original baselines and pathwise parity checks. The integration results
are stored in:

results_ml/integration_checks.json

Detailed per-seed policy comparisons are in:

results_ml/policy_test.csv

Additional optimizer runs are in:

optimization.csv

The project-root README.md contains the conceptual
overview, implementation map, and setup instructions.
