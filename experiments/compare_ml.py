"""Run the prespecified experiment suite. CPU-only; outputs checkpointed by stage."""

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
import json, csv, platform, sys
from pathlib import Path
from time import perf_counter
from dataclasses import asdict, replace
import numpy as np
import sklearn, torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from continuation_dataset import *
from ml_models import make_model
from optimizers import *

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_ml"
FIG = ROOT / "figures_ml"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def save(name, data):
    (OUT / (name + ".json")).write_text(json.dumps(data, indent=2, allow_nan=False))


def table(name, rows):
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with (OUT / (name + ".csv")).open("w") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


def log(msg):
    print(msg, flush=True)


def mse(y, p):
    return float(np.mean((y - p) ** 2))


def checks():
    np.testing.assert_allclose(
        averages(np.array([[100, 80, 90], [100, 110, 90]])),
        [[100, 90, 90], [100, 105, 100]],
    )
    rng = np.random.default_rng(9)
    x = np.column_stack([np.ones(100), rng.normal(size=(100, 3))])
    y = rng.normal(size=100)
    w = rng.normal(size=4)
    eps = 1e-6
    numerical = []
    for i in range(4):
        e = np.eye(4)[i] * eps
        numerical.append(
            (objective(w + e, x, y, 0.01) - objective(w - e, x, y, 0.01)) / (2 * eps)
        )
    np.testing.assert_allclose(gradient(w, x, y, 0.01), numerical, atol=1e-7)
    fitted, _ = optimize(x, y, lr=0.3, lam=0.01, epochs=300)
    np.testing.assert_allclose(fitted, closed_form(x, y, 0.01), atol=1e-7)
    c = Contract()
    p = simulate(c, 1000, 5)
    empty = {"models": {}, "average_only": False}
    np.testing.assert_allclose(evaluate_policy(p, c, empty)[0], hold_payoffs(p, c))

    class Zero:
        def predict(self, x):
            return np.zeros(len(x))

    forced = {"models": {t: Zero() for t in range(1, c.steps)}, "average_only": False}
    payoff, times = evaluate_policy(p, c, forced)
    a = averages(p)
    expected = np.where(a[:, 1:] < c.strike, np.arange(1, c.steps + 1), c.steps).min(
        axis=1
    )
    np.testing.assert_array_equal(times, expected)
    np.testing.assert_allclose(
        payoff,
        np.maximum(c.strike - a[np.arange(len(p)), expected], 0)
        * np.exp(-c.rate * expected / c.steps),
    )
    # Changes after exercise cannot alter a path's decision or realized payoff.
    changed = p.copy()
    for i, t in enumerate(times):
        changed[i, t + 1 :] = 999
    np.testing.assert_allclose(evaluate_policy(changed, c, forced)[0], payoff)
    save(
        "checks",
        {
            "running_average": "passed",
            "analytic_gradient": "passed",
            "gd_vs_closed_form": "passed",
            "hold_only": "passed",
            "first_exercise": "passed",
            "post_exercise_invariance": "passed",
        },
    )
    log("Six numerical/behavior checks passed.")


def verify_integration():
    from asian_lsm import train_asian_put, evaluate_asian_put
    from longstaff_schwartz import train_lsm_put, evaluate_lsm_put

    c = Contract()
    tr = simulate(c, 20000, 42)
    va = simulate(c, 50000, 456)
    kwargs = dict(strike=c.strike, rate=c.rate, maturity=c.maturity)
    records = {}
    for average_only in [False, True]:
        original = train_asian_put(tr, **kwargs, average_only=average_only)
        p, t = evaluate_asian_put(va, original, **kwargs)
        integrated = train_policy(
            tr, c, {"kind": "quadratic"}, average_only=average_only
        )
        p2, t2 = evaluate_policy(va, c, integrated)
        np.testing.assert_allclose(p, p2, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(t, t2)
        expected = 3.331218 if average_only else 3.923369
        np.testing.assert_allclose(p.mean(), expected, atol=5e-7, rtol=0)
        records["average_only" if average_only else "stock_average"] = float(p.mean())
    vc = replace(c, kind="vanilla")
    original, _ = train_lsm_put(tr, **kwargs)
    value, _, _, p = evaluate_lsm_put(va, original, **kwargs, return_payoffs=True)
    integrated = train_policy(tr, vc, {"kind": "quadratic"})
    p2, _ = evaluate_policy(va, vc, integrated)
    np.testing.assert_allclose(p, p2, atol=1e-12, rtol=0)
    np.testing.assert_allclose(value, 6.097770, atol=5e-7, rtol=0)
    records.update(
        vanilla_quadratic=float(value),
        asian_hold=float(hold_payoffs(va, c).mean()),
        pathwise_parity="passed",
        original_baselines="passed",
    )
    save("integration_checks", records)
    log(
        "Original baselines and path-by-path integration checks passed: "
        + json.dumps(records)
    )


def run():
    start = perf_counter()
    checks()
    c = Contract()
    verify_integration()
    cfg = {
        "contract": asdict(c),
        "teacher_paths": 20000,
        "train_paths": 20000,
        "validation_paths": 50000,
        "test_paths": 50000,
        "training_seeds": [42, 43, 44],
        "teacher_seed": 1001,
        "validation_seed": 456,
        "fixed_test_seed": 19001,
        "policy_test_seed": 19002,
        "decision_step": 25,
        "notes": "Integrated with uploaded simulation.py, asian_lsm.py, asians_payoff.py and ridge_lsm.py. No antithetic sampling. All exercise at future observation dates only.",
    }
    save("config", cfg)
    save(
        "environment",
        {
            "python": sys.version,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    )
    train = simulate(c, 20000, 42)
    val = simulate(c, 50000, 456)
    teacher = train_policy(simulate(c, 20000, 1001), c, {"kind": "quadratic"})
    x, y, imm = fixed_targets(train, c, teacher)
    xv, yv, iv = fixed_targets(val, c, teacher)
    np.savez_compressed(
        OUT / "fixed_dataset.npz",
        X_train=x,
        y_train=y,
        exercise_train=imm,
        X_validation=xv,
        y_validation=yv,
        exercise_validation=iv,
    )
    log(
        f"Fixed-target dataset: {len(y)} training, {len(yv)} validation in-the-money observations."
    )
    # Optimization: same fixed quadratic design, intercept excluded from penalty.
    poly = PolynomialFeatures(2, include_bias=False)
    raw = poly.fit_transform(x)
    scaler = StandardScaler().fit(raw)
    z = np.column_stack([np.ones(len(x)), scaler.transform(raw)])
    zraw = np.column_stack([np.ones(len(x)), raw])
    lam = 0.001
    optimal = closed_form(z, y, lam)
    optimum = objective(optimal, z, y, lam)
    optrows = []
    histories = {}
    for method in ["gd", "sgd", "momentum", "nesterov"]:
        w, h = optimize(z, y, method=method, lr=0.03, lam=lam, epochs=400)
        label = method
        histories[label] = h
        optrows.append(
            {
                "method": method,
                "scaled": True,
                "lr": 0.03,
                "objective": objective(w, z, y, lam),
                "closed_form_objective": optimum,
                "objective_gap": objective(w, z, y, lam) - optimum,
                "seconds": h[-1]["seconds"],
                "updates": h[-1]["updates"],
            }
        )
    for scaling, design in [("raw", zraw), ("scaled", z)]:
        for lr in [0.003, 0.03, 0.3]:
            w, h = optimize(design, y, lr=lr, lam=lam, epochs=400)
            histories[f"{scaling}_lr{lr}"] = h
            ref = objective(closed_form(design, y, lam), design, y, lam)
            optrows.append(
                {
                    "method": "gd",
                    "scaled": scaling == "scaled",
                    "lr": lr,
                    "objective": objective(w, design, y, lam),
                    "closed_form_objective": ref,
                    "objective_gap": objective(w, design, y, lam) - ref,
                    "seconds": h[-1]["seconds"],
                    "updates": h[-1]["updates"],
                }
            )
    table("optimization", optrows)
    save("optimization_histories", histories)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ["gd", "sgd", "momentum", "nesterov"]:
        h = histories[name]
        gaps = np.maximum([r["objective"] - optimum for r in h], 1e-10)
        axes[0].semilogy([r["updates"] for r in h], gaps, label=name)
        axes[1].semilogy([r["seconds"] for r in h], gaps, label=name)
    for ax in axes:
        ax.set_ylabel("Objective minus closed-form optimum")
        ax.legend()
        ax.grid(alpha=0.2)
    axes[0].set_xlabel("Parameter updates")
    axes[1].set_xlabel("Seconds")
    fig.tight_layout()
    fig.savefig(FIG / "optimizer_convergence.png")
    plt.close(fig)
    candidates = [("quadratic", {"kind": "quadratic"})]
    candidates += [
        (f"ridge_d{d}_a{a}", {"kind": "ridge", "degree": d, "alpha": a})
        for d in [1, 2, 3]
        for a in [0.01, 1.0, 100.0]
    ]
    candidates += [
        (f"forest_d{d}", {"kind": "forest", "depth": d, "leaf": 40}) for d in [4, 8]
    ]
    candidates += [
        (f"boost_l{leaves}", {"kind": "boost", "leaves": leaves, "iterations": 60})
        for leaves in [7, 15]
    ]
    candidates += [
        (
            f"neural_w{w}_{opt}_wd{wd}",
            {
                "kind": "neural",
                "width": w,
                "optimizer": opt,
                "lr": 0.003 if opt == "adam" else 0.01,
                "weight_decay": wd,
                "epochs": 70,
            },
        )
        for w, opt, wd in [
            (16, "adam", 0.0001),
            (32, "adam", 0.0001),
            (32, "adam", 0.01),
            (32, "sgd", 0.0001),
        ]
    ]
    fits = {}
    scores = []
    neural_hist = {}
    for name, spec in candidates:
        t0 = perf_counter()
        model = make_model(spec, 123)
        if spec["kind"] == "neural":
            model.fit(x, y, validation=(xv, yv))
            neural_hist[name] = model.history
        else:
            model.fit(x, y)
        seconds = perf_counter() - t0
        t0 = perf_counter()
        pv = model.predict(xv)
        predsecs = perf_counter() - t0
        row = {
            "name": name,
            "family": spec["kind"],
            "train_mse": mse(y, model.predict(x)),
            "validation_mse": mse(yv, pv),
            "train_seconds": seconds,
            "prediction_seconds": predsecs,
            "best_epoch": getattr(model, "best_epoch", None),
        }
        scores.append(row)
        fits[name] = model
        log(
            f"Supervised {name}: validation MSE {row['validation_mse']:.4f}, {seconds:.2f}s"
        )
    table("model_selection", scores)
    save("neural_histories", neural_hist)
    selected = {}
    for kind in ["quadratic", "ridge", "forest", "boost", "neural"]:
        winner = min(
            [r for r in scores if r["family"] == kind],
            key=lambda r: r["validation_mse"],
        )
        selected[kind] = {
            "name": winner["name"],
            "spec": dict(candidates)[winner["name"]],
        }
    save(
        "frozen_models", selected
    )  # Written BEFORE any final test sample is generated.
    log("Model configurations frozen from validation MSE: " + json.dumps(selected))
    # Learning curves: same targets, nested samples; fixed validation, no retuning.
    curves = []
    for size in [1000, 3000, len(x)]:
        for family, entry in selected.items():
            model = make_model(entry["spec"], 123)
            model.fit(x[:size], y[:size])
            p = model.predict(xv)
            curves.append(
                {
                    "family": family,
                    "training_observations": size,
                    "train_mse": mse(y[:size], model.predict(x[:size])),
                    "validation_mse": mse(yv, p),
                }
            )
    table("learning_curves", curves)
    fig, ax = plt.subplots(figsize=(7, 4))
    for family in selected:
        rows = [r for r in curves if r["family"] == family]
        ax.plot(
            [r["training_observations"] for r in rows],
            [r["validation_mse"] for r in rows],
            "o-",
            label=family,
        )
    ax.set(
        xlabel="Training observations",
        ylabel="Validation MSE ($ squared)",
        title="Learning curves: fixed future-policy targets",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "learning_curves.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in neural_hist.items():
        axes[0].plot([r["epoch"] for r in h], [r["train_mse"] for r in h], label=name)
        axes[1].plot(
            [r["epoch"] for r in h], [r["validation_mse"] for r in h], label=name
        )
    axes[0].set_title("Neural training loss")
    axes[1].set_title("Neural validation loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE ($ squared)")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "neural_training.png")
    plt.close(fig)
    # Train policies before testing. Hyperparameters are chosen for prediction MSE,
    # not retuned to test payoff, allowing a direct test of the metric mismatch.
    policies = {}
    validation_rows = []
    for seed in [42, 43, 44]:
        tr = simulate(c, 20000, seed)
        for family, entry in selected.items():
            log(f"Training full policy: seed {seed}, {family} ...")
            policy = train_policy(tr, c, entry["spec"], seed=seed)
            policies[(seed, family)] = policy
            pv, tv = evaluate_policy(val, c, policy)
            validation_rows.append(
                {
                    "seed": seed,
                    "family": family,
                    "validation_value": float(pv.mean()),
                    "training_seconds": policy["seconds"],
                }
            )
            log(
                f"  {family}: validation value {pv.mean():.5f}; training {policy['seconds']:.1f}s"
            )
            table("policy_validation", validation_rows)
    # Feature ablation, same configured models. Only seed 42, prespecified.
    ablations = {}
    for family in ["quadratic", "neural"]:
        log("Training average-only ablation: " + family)
        ablations[family] = train_policy(
            train, c, selected[family]["spec"], seed=42, average_only=True
        )
    # Test sample is now opened, after every core configuration has been frozen.
    test = simulate(c, 50000, 19002)
    hold = hold_payoffs(test, c)
    rows = []
    payoffs = {}
    times = {}
    for (seed, family), policy in policies.items():
        t0 = perf_counter()
        p, t = evaluate_policy(test, c, policy)
        seconds = perf_counter() - t0
        payoffs[f"{seed}_{family}"] = p
        times[f"{seed}_{family}"] = t
        row = {
            "seed": seed,
            "family": family,
            **summary(p),
            "gain_over_hold": float(np.mean(p - hold)),
            "early_exercise": float(np.mean(t < c.steps)),
            "train_seconds": policy["seconds"],
            "evaluation_seconds": seconds,
        }
        rows.append(row)
    for row in rows:
        diff = (
            payoffs[f"{row['seed']}_{row['family']}"]
            - payoffs[f"{row['seed']}_quadratic"]
        )
        row.update({"paired_vs_quadratic_" + k: v for k, v in summary(diff).items()})
    table("policy_test", rows)
    save("hold_test", summary(hold))
    np.savez_compressed(OUT / "policy_test_payoffs.npz", hold=hold, **payoffs)
    np.savez_compressed(OUT / "policy_test_exercise_times.npz", **times)
    ablation_rows = []
    for family, policy in ablations.items():
        p, t = evaluate_policy(test, c, policy)
        full = payoffs[f"42_{family}"]
        gain = summary(full - p)
        ablation_rows.append(
            {
                "family": family,
                "average_only_value": float(p.mean()),
                "stock_average_value": float(full.mean()),
                "average_only_early_exercise": float(np.mean(t < c.steps)),
                **{"gain_" + k: v for k, v in gain.items()},
            }
        )
    table("feature_ablation", ablation_rows)
    xt, yt, it = fixed_targets(simulate(c, 50000, 19001), c, teacher)
    fixedrows = []
    for family, entry in selected.items():
        model = fits[entry["name"]]
        pred = model.predict(xt)
        fixedrows.append(
            {
                "family": family,
                "name": entry["name"],
                "test_mse": mse(yt, pred),
                "test_observations": len(yt),
                "validation_mse": next(
                    r["validation_mse"] for r in scores if r["name"] == entry["name"]
                ),
            }
        )
    table("fixed_test", fixedrows)
    # Decision plots use actual observed states, not extrapolation rectangles.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for ax, family in zip(axes, ["quadratic", "boost", "neural"]):
        model = policies[(42, family)]["models"][25]
        pick = np.random.default_rng(7).choice(
            len(xt), min(3500, len(xt)), replace=False
        )
        decisions = it[pick] > model.predict(xt[pick])
        ax.scatter(
            xt[pick, 0] * 100,
            xt[pick, 1] * 100,
            c=decisions,
            cmap="coolwarm",
            s=3,
            alpha=0.45,
            vmin=0,
            vmax=1,
        )
        ax.set_title(family)
        ax.set_xlabel("Current stock price ($)")
    axes[0].set_ylabel("Running average ($)")
    fig.suptitle("Six-month policy decisions: red = exercise; blue = wait", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "decision_regions.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, family in enumerate(selected):
        vals = [r["mean"] for r in rows if r["family"] == family]
        ax.scatter([j] * len(vals), vals, s=35)
    ax.axhline(hold.mean(), color="gray", linestyle="--", label="Hold to expiration")
    ax.set_xticks(range(len(selected)), list(selected))
    ax.set_ylabel("Test policy value ($)")
    ax.set_title("Three training seeds; same 50,000 test paths")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "policy_values.png")
    plt.close(fig)
    # Prespecified one-seed robustness, retrain at each new parameter setting.
    robustness = []
    for label, sc in [
        ("low_vol", replace(c, volatility=0.10)),
        ("high_vol", replace(c, volatility=0.35)),
        ("higher_strike", replace(c, strike=110.0)),
    ]:
        tr = simulate(sc, 20000, 42)
        te = simulate(sc, 50000, 19003)
        hp = hold_payoffs(te, sc)
        local = {}
        for family in ["quadratic", "ridge", "neural"]:
            log(f"Robustness {label}, {family} ...")
            policy = train_policy(tr, sc, selected[family]["spec"], seed=42)
            p, t = evaluate_policy(te, sc, policy)
            local[family] = p
            robustness.append(
                {
                    "scenario": label,
                    "family": family,
                    "volatility": sc.volatility,
                    "strike": sc.strike,
                    **summary(p),
                    "hold_value": float(hp.mean()),
                    "early_exercise": float(np.mean(t < sc.steps)),
                    "training_seconds": policy["seconds"],
                }
            )
        for row in robustness:
            if row["scenario"] == label:
                row.update(
                    {
                        "paired_vs_quadratic_" + k: v
                        for k, v in summary(
                            local[row["family"]] - local["quadratic"]
                        ).items()
                    }
                )
        table("robustness", robustness)
    # Simple ordinary-put benchmark, schedule matched to CRR tree.
    vc = replace(c, kind="vanilla")
    tr = simulate(vc, 20000, 42)
    te = simulate(vc, 50000, 19004)
    vanilla = []
    for family in ["quadratic", "forest", "neural"]:
        log("Vanilla reference " + family)
        policy = train_policy(tr, vc, selected[family]["spec"], seed=42)
        p, t = evaluate_policy(te, vc, policy)
        vanilla.append(
            {"family": family, **summary(p), "training_seconds": policy["seconds"]}
        )
    table("vanilla", vanilla)
    from black_scholes import black_scholes_price

    bs = black_scholes_price(
        stock_price=vc.spot,
        strike=vc.strike,
        rate=vc.rate,
        volatility=vc.volatility,
        maturity=vc.maturity,
        option_type="put",
    )
    save(
        "vanilla_reference",
        {
            "tree_500_steps": tree_price(vc, 500),
            "tree_1000_steps": tree_price(vc, 1000),
            "black_scholes_european": bs,
            "european_mc": summary(hold_payoffs(te, vc)),
        },
    )
    import joblib

    joblib.dump(
        {
            "contract": c,
            "selected": selected,
            "seed42_policies": {k: v for (s, k), v in policies.items() if s == 42},
            "teacher": teacher,
        },
        OUT / "trained_policies.joblib",
        compress=3,
    )
    save(
        "run_complete",
        {
            "seconds": perf_counter() - start,
            "status": "complete",
            "fixed_test_observations": len(yt),
        },
    )
    log(f"COMPLETE: {perf_counter()-start:.1f} seconds")


if __name__ == "__main__":
    run()
