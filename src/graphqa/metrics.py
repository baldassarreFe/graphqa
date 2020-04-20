from typing import List, Dict, Mapping, Any

import numpy as np
import pandas as pd
import sklearn.metrics
import scipy.stats
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from loguru import logger


def rmse(preds, true):
    return np.sqrt(sklearn.metrics.mean_squared_error(preds, true))


def r2_score(preds, true):
    return sklearn.metrics.r2_score(true, preds)


def pearson(preds, true):
    return scipy.stats.pearsonr(preds, true)[0]


def spearmanr(preds, true):
    return scipy.stats.spearmanr(preds, true)[0]


def kendalltau(preds, true):
    return scipy.stats.kendalltau(preds, true)[0]


def first_rank_loss(preds, true):
    return true.max() - true.iloc[preds.argmax()]


def zscore(preds, true):
    zscores = pd.Series(scipy.stats.zscore(true), index=true.index)
    return zscores.iloc[preds.argmax()]


def per_score_metrics(df_score: pd.DataFrame):
    df = df_score.droplevel(axis=1, level="score")
    df = df.dropna(axis="index", subset=["true"])
    if len(df) == 0:
        logger.warning(
            f'Empty score dataframe after removing NaN values from "true"\n{df_score}'
        )
        return pd.Series({"RMSE": 0, "R2": 1, "R": 1, "ρ": 1, "τ": 1, "z": 0, "FRL": 0})
    pred = df["pred"]
    true = df["true"]
    return pd.Series(
        {
            "RMSE": rmse(pred, true),
            "R2": r2_score(pred, true),
            "R": pearson(pred, true),
            "ρ": spearmanr(pred, true),
            "τ": kendalltau(pred, true),
            "z": zscore(pred, true),
            "FRL": first_rank_loss(pred, true),
        }
    )


def compute_global_metrics(df_target: pd.DataFrame):
    return (
        df_target.groupby(axis=1, level="score")
        .apply(per_score_metrics)
        .rename_axis(index="metric")
    )


def compute_local_metrics(df_decoy: pd.DataFrame):
    return (
        df_decoy.groupby(axis=1, level="score")
        .apply(per_score_metrics)
        .rename_axis(index="metric")
    )


def groupby_apply(df, by, fn, *, n_jobs=-1, verbose=0):
    if not isinstance(by, (tuple, list)):
        by = [by]

    def wrapper(g: tuple):
        group_key = g[0]
        group_df = g[1]
        return group_key, fn(group_df)

    with Parallel(n_jobs=n_jobs, verbose=verbose) as pool:
        result = pool(delayed(wrapper)(g) for g in df.groupby(by))
    result = pd.concat([r[1] for r in result], keys=[r[0] for r in result], names=by)
    return result


def scores_from_outputs(outputs: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    scores_global = (
        pd.DataFrame(
            {
                k: np.concatenate([o["scores_global"][k] for o in outputs])
                for k in outputs[0]["scores_global"]
                if isinstance(k, tuple)
            },
            index=pd.MultiIndex.from_arrays(
                [
                    sum((o["scores_global"][k] for o in outputs), [])
                    for k in outputs[0]["scores_global"]
                    if not isinstance(k, tuple)
                ],
                names=[
                    k for k in outputs[0]["scores_global"] if not isinstance(k, tuple)
                ],
            ),
        )
        .rename_axis(columns=["score", None])
        .sort_index()
    )

    scores_local = (
        pd.DataFrame(
            {
                k: np.concatenate([o["scores_local"][k] for o in outputs])
                for k in outputs[0]["scores_local"]
                if isinstance(k, tuple)
            },
            index=pd.MultiIndex.from_arrays(
                [
                    np.concatenate([o["scores_local"][k] for o in outputs])
                    for k in outputs[0]["scores_local"]
                    if not isinstance(k, tuple)
                ],
                names=[
                    k for k in outputs[0]["scores_local"] if not isinstance(k, tuple)
                ],
            ),
        )
        .rename_axis(columns=["score", None])
        .sort_index()
    )
    return {"global": scores_global, "local": scores_local}


def metrics_from_scores(scores: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    # Global metrics
    scores_global = scores["global"]
    metrics_global = compute_global_metrics(scores_global)
    metrics_global_per_target = (
        groupby_apply(scores_global, "target_id", compute_global_metrics, n_jobs=10)
        .groupby("metric")
        .mean()
    )
    # print("Global metrics (across all targets)", metrics_global, sep="\n")
    # print("Global metrics (averaged per target)", metrics_global_per_target, sep="\n")

    # Local metrics
    scores_local = scores["local"]
    metrics_local = compute_local_metrics(scores_local)
    metrics_local_per_decoy = (
        groupby_apply(
            scores_local, ["target_id", "decoy_id"], compute_local_metrics, n_jobs=10
        )
        .groupby("metric")
        .mean()
    )
    # print("Local metrics (across all decoys)", metrics_local, sep="\n")
    # print("Local metrics (averaged per decoy)", metrics_local_per_decoy, sep="\n")

    return {
        "global": metrics_global,
        "global_per_target": metrics_global_per_target,
        "local": metrics_local,
        "local_per_decoy": metrics_local_per_decoy,
    }


def figures_from_scores(scores: Mapping[str, pd.DataFrame]) -> Dict[str, plt.Figure]:
    scores_global = scores["global"]
    scores_local = scores["local"]

    figures = {}
    for key, title, true, preds in [
        (
            "global/gdtts",
            "GDT-TS",
            scores_global["gdtts"]["true"],
            scores_global["gdtts"]["pred"],
        ),
        (
            "global/cad",
            "CAD",
            scores_global["cad"]["true"],
            scores_global["cad"]["pred"],
        ),
        (
            "local/lddt",
            "LDDT",
            scores_local["lddt"]["true"],
            scores_local["lddt"]["pred"],
        ),
        ("local/cad", "CAD", scores_local["cad"]["true"], scores_local["cad"]["pred"]),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=100)
        histogram(ax, title, true, preds)
        figures[key] = fig

    return figures


def log_dict_from_metrics(
    metrics: Mapping[str, pd.DataFrame], prefix=None, sep="/"
) -> Dict[str, Any]:
    if prefix is None:
        prefix = ()
    if not isinstance(prefix, tuple):
        prefix = (prefix,)
    return {
        **{
            sep.join(prefix + ("global",) + k): v
            for k, v in metrics["global"].unstack().to_dict().items()
        },
        **{
            sep.join(prefix + ("global_per_target",) + k): v
            for k, v in metrics["global_per_target"].unstack().to_dict().items()
        },
        **{
            sep.join(prefix + ("local",) + k): v
            for k, v in metrics["local"].unstack().to_dict().items()
        },
        **{
            sep.join(prefix + ("local_per_decoy",) + k): v
            for k, v in metrics["local_per_decoy"].unstack().to_dict().items()
        },
    }


def histogram(ax, title, true, preds):
    bins = np.linspace(0, 1, 100 + 1)
    hist, _, _ = np.histogram2d(true, preds, bins=bins)
    ax.pcolormesh(bins, bins, hist.T, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "", "", "", "", "1"])
    ax.set_title(title)
    ax.set_xlabel(f"True", labelpad=-10)
    ax.set_ylabel("Predicted")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "", "", "", "", "1"])
    ax.set_ylabel("Predicted", labelpad=-8)
    ax.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=0.5, zorder=2)
