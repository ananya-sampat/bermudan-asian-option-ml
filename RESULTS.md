# ML for option exercise: measured results

This report describes an executed, reproducible simulation study. All model configurations were chosen using validation data before the final test samples were opened. Source code, numerical checks, raw results, payoff arrays and trained policies are included in the accompanying project folder.

## Main findings

Read the measured tables below as comparisons of the implemented training workflows and finite hyperparameter search. The original project remains the foundation. The neural architecture and polynomial ridge strength were reselected using the user’s simulator and validation paths; results from the earlier standalone package are not mixed into these tables.

## What was implemented

Manual GD, mini-batch SGD, momentum and Nesterov; polynomial and ridge regression; train-only scaling; random forests; gradient-boosted trees; a PyTorch neural network with explicit mini-batch training, early stopping and optimizer/regularization comparisons; learning curves; backward exercise-policy training; held-out paired evaluation; feature ablation; three-seed base results; and retrained parameter robustness. Classification and reinforcement learning were optional and were not added.

## Experimental setup

Risk-neutral GBM, spot/strike $100, rate 5%, volatility 20%, one year, 50 future exercise dates. Asian average includes time zero. Each model has 20,000 training paths; validation uses 50,000 paths. An independent 20,000-path quadratic teacher supplies the fixed-target task. Separate 50,000-path samples assess prediction and full-policy value. This run imports the uploaded simulation.py, asians_payoff.py, asian_lsm.py and longstaff_schwartz.py. Polynomial ridge fitting reuses ridge_lsm.fit_ridge. All 16 original files are preserved byte-for-byte. Original Asian and vanilla baselines and pathwise adapter equivalence are checked before running new experiments.

Hold-to-expiration Asian value on the policy test sample: **$3.3636**, approximate 95% interval **[3.3174, 3.4098]**.

## Full Asian exercise policies

The table below uses training seed 42 and the same 50,000 unseen evaluation paths. Dollar values are discounted payoffs per option unit. Paired intervals compare each policy with the quadratic on the same paths.

| Model | Test value | Value 95% interval | Gain vs quadratic | Paired gain 95% interval | Early exercise | Train seconds |
|---|---|---|---|---|---|---|
| quadratic | $3.9531 | [3.9073, 3.9989] | $0.0000 | [0.0000, 0.0000] | 40.7% | 0.10 |
| ridge | $3.9530 | [3.9073, 3.9987] | $-0.0000 | [-0.0033, 0.0032] | 40.8% | 0.27 |
| forest | $3.9164 | [3.8715, 3.9614] | $-0.0366 | [-0.0437, -0.0296] | 42.1% | 12.60 |
| boost | $3.9281 | [3.8828, 3.9734] | $-0.0250 | [-0.0318, -0.0182] | 40.5% | 4.17 |
| neural | $3.9481 | [3.9023, 3.9939] | $-0.0050 | [-0.0108, 0.0009] | 39.5% | 29.13 |

### Training-seed sensitivity

| Model | Seed 42 | Seed 43 | Seed 44 | Mean across three policies |
|---|---|---|---|---|
| quadratic | $3.9531 | $3.9587 | $3.9565 | $3.9561 |
| ridge | $3.9530 | $3.9583 | $3.9566 | $3.9560 |
| forest | $3.9164 | $3.9167 | $3.9159 | $3.9163 |
| boost | $3.9281 | $3.9300 | $3.9290 | $3.9290 |
| neural | $3.9481 | $3.9521 | $3.9442 | $3.9482 |

All three policies per family share a test sample. The average across policies is descriptive: it is not three independent test sets and does not justify dividing uncertainty by sqrt(3). Full per-seed paired intervals are in `results_ml/policy_test.csv`.

## Prediction accuracy on common targets

Every candidate predicts the same realized future-policy payoff at date 25, conditional on being in the money. These targets contain future-path noise and come from an independent quadratic teacher. Lower test MSE is better, but it does not measure error against exact optimal continuation.

| Model | Selected configuration | Validation MSE | Test MSE |
|---|---|---|---|
| quadratic | quadratic | 4.9821 | 4.9609 |
| ridge | ridge_d3_a1.0 | 4.9804 | 4.9605 |
| forest | forest_d8 | 5.1073 | 5.1258 |
| boost | boost_l15 | 5.0747 | 5.0756 |
| neural | neural_w32_adam_wd0.0001 | 4.9765 | 4.9705 |

Configurations were selected by single-date validation MSE and reused across exercise dates. They were not independently tuned to maximize full-policy validation value. This is a controlled, limited search, not each model family’s best attainable performance.

## Feature ablation

Average-only and stock-plus-average use the same selected family settings (seed 42). The paired interval measures the full-feature policy minus the average-only policy.

| Model | Average only | Stock + average | Gain | Paired 95% interval | Average-only early exercise |
|---|---|---|---|---|---|
| quadratic | $3.3632 | $3.9531 | $0.5899 | [0.5739, 0.6058] | 6.6% |
| neural | $3.3647 | $3.9481 | $0.5834 | [0.5678, 0.5990] | 13.1% |

## Manual optimization

All four methods use the same standardized quadratic design, lambda 0.001 and learning rate 0.03. The closed-form optimum provides an implementation/convergence reference. SGD, momentum and Nesterov use batches of 256; GD uses the full dataset. Equal epoch counts do not imply equal update counts.

| Method | Final objective | Gap to optimum | Updates | Seconds |
|---|---|---|---|---|
| gd | 2.647239 | 0.108052 | 400 | 0.125 |
| sgd | 2.550227 | 0.011040 | 14800 | 0.445 |
| momentum | 2.542132 | 0.002945 | 14800 | 0.432 |
| nesterov | 2.543284 | 0.004097 | 14800 | 0.474 |

The additional scaling/learning-rate runs are in `optimization.csv`. Compare each run with its own closed-form objective: standardizing polynomial features changes the coordinate meaning of ridge regularization. The optimizer implementations passed an analytic-gradient check and a well-conditioned convergence test; this dataset is more ill-conditioned and finite-budget gaps need not vanish.

## Retrained robustness

One training seed per condition; settings frozen from the base scenario. Each model is retrained under the new parameters. This does not test transfer of a fixed pretrained network. Exercise still starts at the first future date, including strike 110.

| Scenario | Model | Value | Gain vs quadratic | Paired gain 95% interval |
|---|---|---|---|---|
| low_vol | quadratic | $1.5617 | $0.0000 | [0.0000, 0.0000] |
| low_vol | ridge | $1.5593 | $-0.0024 | [-0.0046, -0.0001] |
| low_vol | neural | $1.5624 | $0.0007 | [-0.0023, 0.0038] |
| high_vol | quadratic | $7.4636 | $0.0000 | [0.0000, 0.0000] |
| high_vol | ridge | $7.4682 | $0.0046 | [-0.0004, 0.0096] |
| high_vol | neural | $7.4556 | $-0.0080 | [-0.0167, 0.0007] |
| higher_strike | quadratic | $12.0170 | $0.0000 | [0.0000, 0.0000] |
| higher_strike | ridge | $12.0191 | $0.0021 | [-0.0044, 0.0086] |
| higher_strike | neural | $12.0104 | $-0.0066 | [-0.0172, 0.0040] |

## Ordinary-put control

| Model | Policy value | Approximate 95% interval |
|---|---|---|
| quadratic | $6.0228 | [5.9601, 6.0854] |
| forest | $5.9198 | [5.8562, 5.9833] |
| neural | $6.0372 | [5.9760, 6.0984] |

Matched-schedule tree values: $6.0773 (500 steps), $6.0781 (1,000 steps). European Black–Scholes value: $5.5735; European Monte Carlo: $5.5416, 95% interval [5.4659, 5.6173]. Trees are numerical approximations. Policy means above a tree price can result from Monte Carlo and grid error.

## Qualifications

The 95% intervals quantify evaluation-path uncertainty conditional on each trained policy, not training uncertainty. Comparisons are exploratory with no multiple-comparison adjustment. Neural policy models reserve 15% of supplied training observations for early stopping while the other families use all observations. Learning curves use one nested sample order. Runtime includes Python/library overhead and is hardware dependent; the first PyTorch fit includes startup overhead. The Asian optimal price and optimal decision regions remain unknown. Results describe this simulator and implemented training budgets, not market profitability.

## Files and reproducibility

The experiment suite completed in 5.85 minutes on the recorded environment. See the project-root `README.md` for installation, assumptions and a result-file map. Run `python run.py full` from the project root to regenerate into a new runs/ directory. Six numerical/behavior checks passed, plus original baseline reproduction and pathwise parity checks (see results_ml/integration_checks.json).

## Figures

![Manual optimizer convergence against the analytical objective.](bermudan-asian-option-ml/figures_ml/optimizer_convergence.png)

Manual optimizer convergence against the analytical objective.

![Neural training and validation curves for the predefined candidates.](bermudan-asian-option-ml/figures_ml/neural_training.png)

Neural training and validation curves for the predefined candidates.

![Fixed-target prediction performance versus training-data size.](bermudan-asian-option-ml/figures_ml/learning_curves.png)

Fixed-target prediction performance versus training-data size.

![Full-policy values over three training seeds.](bermudan-asian-option-ml/figures_ml/policy_values.png)

Full-policy values over three training seeds.

![Date-25 exercise decisions at sampled observed states; red = exercise, blue = wait. These are policy outputs, not optimal labels.](bermudan-asian-option-ml/figures_ml/decision_regions.png)

Date-25 exercise decisions at sampled observed states; red = exercise, blue = wait. These are policy outputs, not optimal labels.
