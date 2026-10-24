"""Gráficas del proyecto MiBici: artefactos de MLflow y figuras de reports/figures/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

PALETTE = ["#2a6f97", "#e07a5f", "#3d405b", "#81b29a", "#f2cc8f", "#9c6644", "#6d597a", "#b56576"]


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_daily_forecast(dates: pd.Series, y_true, y_pred, title: str, path: Path) -> Path:
    """P1: viajes totales del sistema por día, observados vs. predichos (suma de estaciones)."""
    df = pd.DataFrame({"date": pd.to_datetime(dates), "real": y_true, "predicho": y_pred})
    daily = df.groupby("date")[["real", "predicho"]].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily["real"], color=PALETTE[2], lw=1.6, label="Real")
    ax.plot(daily.index, daily["predicho"], color=PALETTE[1], lw=1.4, ls="--", label="Predicho")
    ax.set_title(title)
    ax.set_ylabel("Viajes iniciados por día")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_regression_diagnostics(y_true, y_pred, title: str, path: Path) -> Path:
    """P1: real contra predicho y residuales contra predicho, por estación y día."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    lim = float(np.percentile(np.r_[y_true, y_pred], 99.5))
    axes[0].scatter(y_pred, y_true, s=4, alpha=0.15, color=PALETTE[0], rasterized=True)
    axes[0].plot(
        [0, lim], [0, lim], color=PALETTE[1], ls="--", lw=1.2, label="Predicción perfecta"
    )
    axes[0].set(xlim=(0, lim), ylim=(0, lim), xlabel="Predicho", ylabel="Real")
    axes[0].set_title("Real contra predicho")
    axes[0].legend()
    axes[1].scatter(y_pred, residuals, s=4, alpha=0.15, color=PALETTE[2], rasterized=True)
    axes[1].axhline(0, color=PALETTE[1], ls="--", lw=1.2)
    axes[1].set(xlim=(0, lim), xlabel="Predicho", ylabel="Residual (real - predicho)")
    axes[1].set_title(f"Residuales (media {residuals.mean():.2f})")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    return _save(fig, path)


def plot_classification(y_true, y_score, y_pred, title: str, path: Path) -> Path:
    """P2: curva precisión-exhaustividad y matriz de confusión."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    PrecisionRecallDisplay.from_predictions(
        y_true, y_score, ax=axes[0], curve_kwargs={"color": PALETTE[0]}
    )
    axes[0].set_title("Curva precisión-exhaustividad")
    axes[0].grid(alpha=0.3)
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        ax=axes[1],
        cmap="Blues",
        display_labels=["Sin riesgo", "Riesgo"],
        colorbar=False,
    )
    axes[1].set_title("Matriz de confusión")
    fig.suptitle(title)
    return _save(fig, path)


def plot_cluster_map(df: pd.DataFrame, labels, title: str, path: Path) -> Path:
    """P3: ubicación de las estaciones coloreadas por grupo."""
    fig, ax = plt.subplots(figsize=(7, 7))
    labels = np.asarray(labels)
    for k in np.unique(labels):
        sel = labels == k
        ax.scatter(
            df.loc[sel, "longitude"],
            df.loc[sel, "latitude"],
            s=22,
            alpha=0.85,
            color=PALETTE[k % len(PALETTE)],
            label=f"Grupo {k} (n={sel.sum()})",
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_title(title)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_cluster_profiles(df: pd.DataFrame, labels, title: str, path: Path) -> Path:
    """P3: perfil horario medio de salidas de cada grupo (días hábiles y fin de semana)."""
    labels = np.asarray(labels)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, tag, name in [(axes[0], "wd", "Días hábiles"), (axes[1], "we", "Fin de semana")]:
        cols = [f"dep_{tag}_h{h:02d}" for h in range(24)]
        for k in np.unique(labels):
            ax.plot(
                range(24),
                df.loc[labels == k, cols].mean().values,
                lw=1.8,
                color=PALETTE[k % len(PALETTE)],
                label=f"Grupo {k}",
            )
        ax.set_title(name)
        ax.set_xlabel("Hora del día")
        ax.set_xticks(range(0, 24, 3))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Proporción de salidas")
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    return _save(fig, path)


def plot_k_selection(ks, silhouettes, chosen_k: int, path: Path) -> Path:
    """P3: silueta en función del número de grupos."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(ks, silhouettes, marker="o", color=PALETTE[0])
    ax.axvline(chosen_k, color=PALETTE[1], ls="--", label=f"k elegido = {chosen_k}")
    ax.set_xlabel("Número de grupos (k)")
    ax.set_ylabel("Silueta (train)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)
