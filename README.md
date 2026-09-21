# Learning Exercise Policies for Bermudan Asian Put Options

This project studies how machine-learning models can be used to make early-exercise decisions for a **Bermudan Asian put option**.

The central question is:

> Given the option’s current state, is it better to exercise now or keep the option alive?

The project uses simulated stock-price paths to train models that estimate the value of waiting. Those estimates are then converted into exercise decisions and evaluated on new, unseen paths.

This is an option-pricing and optimal-stopping project. It is not a system for forecasting real stock prices or generating trading signals.

## Contract

The main contract is an arithmetic-average Asian put with a one-year maturity and 50 possible exercise dates.

A standard put option with strike \(K\) pays

$$
\max(K-S_t, 0),
$$

where \(S_t\) is the stock price when the option is exercised.

For an Asian put, the payoff depends instead on the average price observed so far:

$$
A_t = \frac{S_0 + S_1 + \cdots + S_t}{t+1}.
$$

Exercising at date \(t\) pays

$$
\max(K-A_t, 0).
$$

The base contract uses:

| Parameter           |  Value |
| ------------------- | -----: |
| Initial stock price |   $100 |
| Strike price        |   $100 |
| Risk-free rate      |     5% |
| Volatility          |    20% |
| Maturity            | 1 year |
| Exercise dates      |     50 |

The average includes the initial stock price. Exercise is allowed only on the 50 scheduled dates, which makes the contract Bermudan rather than fully American.

## Why early exercise is hard

At any exercise date, the immediate payoff is known. If the running average is $90 and the strike is $100, exercising pays $10.

The difficult part is deciding whether that $10 is better than waiting.

Waiting may be valuable because future stock prices could lower the running average and increase the eventual payoff. Waiting also has risk: the stock price may rise, the average may rise, and any future payoff must be discounted back to today.

The decision compares:

$$\text{immediate exercise payoff}$$ with $$\text{continuation value} = \text{estimated value of waiting}$$

The policy exercises whenever

$$\text{immediate payoff} >\text{predicted continuation value}.$$

This is an **optimal stopping** problem because exercising stops the contract permanently.

## Simulating stock-price paths

The project does not use historical stock data. It simulates possible stock-price paths under geometric Brownian motion, a standard model used in introductory option pricing.

At each small time step, the price changes according to a deterministic drift term and a random shock. The simulation uses the risk-free rate as the drift because the goal is option valuation under a risk-neutral pricing model.

A single path is one possible future:

| Date                | Stock price |
| ------------------- | ----------: |
| Today               |        $100 |
| First exercise date |         $97 |
| Later date          |         $91 |
| Later date          |         $95 |
| Expiration          |         $84 |

Thousands of paths are generated for training and separate paths are generated for validation and final evaluation.

Every payoff is discounted using

$$
e^{-rt},
$$

so a payoff received later is expressed in today’s dollars.

## Why the current stock price and average are both inputs

The running average determines the payoff if the option is exercised today. The current stock price helps determine what may happen to the running average later.

For example, these two states have the same immediate payoff:

| Current stock price | Running average | Immediate payoff |
| ------------------: | --------------: | ---------------: |
|                 $80 |             $95 |               $5 |
|                $110 |             $95 |               $5 |

But they should not necessarily lead to the same decision. A low current price may keep future averages low, while a high current price may pull the average upward.

The main feature experiment compares:

* an **average-only** model, using \(A_t\);
* a **full-state** model, using both \(S_t\) and \(A_t\).

This tests whether the extra state information improves the learned policy.

## Baseline: Longstaff–Schwartz regression

The main baseline is the Longstaff–Schwartz method, also called least-squares Monte Carlo.

It works backward through time.

1. At expiration, every simulated path receives its final payoff.
2. At the previous exercise date, the later discounted payoff becomes a regression target for the value of waiting.
3. A regression model estimates continuation value using the current state.
4. Paths where immediate exercise is better are marked as exercised at that date.
5. The algorithm moves backward to the next earlier date and repeats the process.

The baseline continuation model is quadratic regression.

For the Asian option, the full-state quadratic features are:

$$
[1,\ s,\ a,\ s^2,\ sa,\ a^2],
$$

where

$$
s = S_t/K,
\qquad
a = A_t/K.
$$

The interaction term \(sa\) allows the influence of stock price to depend on the running average.

After training, the policy is evaluated forward on new paths. The evaluation code uses only the state available at the current date and the saved models. It does not use future prices to make a decision.

## Classical pricing checks

The repository also contains simpler ordinary-put experiments used to validate the financial setup.

* `black_scholes.py` calculates the European option benchmark from the Black–Scholes formula.
* `monte_carlo.py` estimates a European payoff by simulation.
* `binomial.py` prices ordinary puts with a Cox–Ross–Rubinstein binomial tree.
* `longstaff_schwartz.py` trains the ordinary-put Longstaff–Schwartz baseline.

These checks are useful because the ordinary put has established numerical references. The Asian option is harder: there is no simple closed-form “correct” early-exercise price used as a final answer.

## Machine-learning models

The project compares several continuation-value models.

| Model                  | Role in the project                                  |
| ---------------------- | ---------------------------------------------------- |
| Quadratic regression   | Main Longstaff–Schwartz baseline                     |
| Cubic ridge regression | Tests a more flexible polynomial with regularization |
| Random forest          | Tests tree-based nonlinear regression                |
| Gradient boosting      | Tests sequential tree ensembles                      |
| Neural network         | Tests a learned nonlinear representation             |

### Ridge regression

Ridge regression adds a penalty for large coefficients:

$$
\text{loss}
=
\text{mean squared error}
+
\lambda \sum_{j \ne 0} \beta_j^2.
$$

The penalty can reduce overfitting when polynomial features are highly correlated or overly flexible.

### Random forest and gradient boosting

Trees split the input space into regions. For example, a tree may make different continuation estimates for low-average/low-price states and low-average/high-price states.

A random forest averages many trees. Gradient boosting fits trees sequentially, with each new tree attempting to improve errors made by the earlier ones.

### Neural network

The neural model is a small PyTorch network with:

* two inputs: current stock price and running average;
* two hidden layers;
* ReLU activation functions;
* one output: predicted continuation value.

It is trained with mini-batches and mean squared error. The model standardizes its input features using training data and uses a validation split for early stopping.

## Controlled prediction experiment

The project has a fixed-target supervised-learning experiment in addition to the full early-exercise experiment.

At the midpoint of the option’s life, each model receives the same inputs:

$$
(S_t, A_t).
$$

Each model also receives the same target: the discounted payoff from following a separately trained future exercise policy.

This makes model comparison cleaner because every model is predicting the same target. Performance is measured with mean squared error:

$$
\text{MSE}
=
\frac{1}{n}
\sum_{i=1}^{n}
(\hat y_i-y_i)^2.
$$

Lower MSE means the model predicts the common continuation target more accurately.

However, lower prediction error does not automatically produce a better exercise policy. Small errors near the exercise boundary can change an exercise decision, while larger errors far from the boundary may not matter.

## Full policy evaluation

Each selected model is also trained as a complete early-exercise policy.

For each model family:

1. Simulate training paths.
2. Fit continuation models backward through the 49 intermediate exercise dates.
3. Freeze the learned policy.
4. Evaluate it on 50,000 fresh test paths.
5. Record discounted payoff, exercise time, early-exercise frequency, and runtime.

The same test paths are used for all policy comparisons. This allows paired payoff differences to be calculated path by path.

For two policies \(A\) and \(B\), the paired difference on path \(i\) is

$$
D_i =
\text{payoff from } A
-
\text{payoff from } B.
$$

The project reports the mean of these differences and a paired 95% confidence interval.

## Optimization experiments

The repository also implements several optimization methods from scratch for a regularized regression objective:

* batch gradient descent;
* mini-batch stochastic gradient descent;
* momentum;
* Nesterov accelerated gradient.

The implementation includes:

* the loss function;
* its analytic gradient;
* a closed-form ridge solution for comparison;
* a numerical gradient check;
* convergence plots showing objective value against updates and time.

This part of the project is separate from the full option policies. Its purpose is to study how optimization methods behave when fitting the continuation-value regression problem.

## Data split

Different simulated paths are used for different purposes.

| Dataset              | Purpose                                                 |
| -------------------- | ------------------------------------------------------- |
| Training paths       | Fit model parameters                                    |
| Validation paths     | Select among predefined model configurations            |
| Teacher-policy paths | Build common targets for the controlled prediction task |
| Final test paths     | Evaluate the selected policies                          |

This separation matters because a model should not be evaluated on data that determined its hyperparameters or architecture.

The main policy results use three different training seeds, 42, 43, and 44, to show how much the learned policy changes when the training sample changes.

## Results

The table below reports mean policy value across three training seeds, evaluated on the same 50,000 final test paths.

| Model                  | Mean discounted policy value |
| ---------------------- | ---------------------------: |
| Quadratic regression   |                      $3.9561 |
| Cubic ridge regression |                      $3.9560 |
| Neural network         |                      $3.9487 |
| Gradient boosting      |                      $3.9258 |
| Random forest          |                      $3.9163 |

The quadratic and cubic ridge models are effectively tied. The neural network is close to the quadratic baseline, but the paired comparisons do not show a reliable improvement. The forest and boosted-tree policies are lower under the selected settings.

The main result is the feature ablation:

| Quadratic-policy input                | Estimated policy value |
| ------------------------------------- | ---------------------: |
| Running average only                  |                $3.3632 |
| Current stock price + running average |                $3.9531 |

Adding current stock price increased the policy value by about $0.59. The paired 95% confidence interval for that gain was approximately:

$$
[0.574,\ 0.606].
$$

The experiment therefore suggests that, for this problem, choosing the correct state information mattered more than replacing the quadratic model with a more complex one.

## Repository structure

| Location                      | Purpose                                                  |
| ----------------------------- | -------------------------------------------------------- |
| `src/simulation.py`           | Simulates stock-price paths                              |
| `src/asians_payoff.py`        | Computes running averages and Asian put payoffs          |
| `src/asian_lsm.py`            | Implements the quadratic Asian Longstaff–Schwartz policy |
| `src/continuation_dataset.py` | Builds common targets and evaluates generic policies     |
| `src/optimizers.py`           | Implements gradient-based optimization methods           |
| `src/ml_models.py`            | Defines ridge, forest, boosting, and neural models       |
| `experiments/compare_ml.py`   | Runs the full study                                      |
| `results_ml/`                 | Saved numerical outputs                                  |
| `figures_ml/`                 | Figures generated by the experiment                      |
| `archive/`                    | Earlier pricing benchmarks and exploratory scripts       |

## Reproducing the experiment

Install the dependencies in the `quant` environment:

```bash
conda activate quant
cd ~/Desktop/american-option-ml-clean
python -m pip install -r requirements.txt
```

Run the numerical checks:

```bash
python run.py check
```

Run the full experiment:

```bash
python run.py full
```

The full run trains the models, performs model selection on validation paths, evaluates the final policies, and writes CSV files and figures to a timestamped folder inside `runs/`.

Neural-network and boosting results may vary slightly across machines or package versions. The quadratic baseline, ridge comparison, feature-ablation result, and broad ranking should reproduce closely.

## Limitations

This is a simulation study. Its conclusions depend on:

* the geometric Brownian-motion price model;
* the contract parameters;
* the number of simulated paths;
* the tested hyperparameter grid;
* the selected model architectures and training budgets.

The project does not know the exact optimal early-exercise value of the Asian option. It estimates the values of the learned policies. The reported values are simulated discounted option payoffs, not real trading profits.
