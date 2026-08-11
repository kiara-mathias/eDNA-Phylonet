"""Confidence calibration metrics + post-hoc recalibrators.

Validates that reported per-rank confidence behaves like a probability:
bin predictions by confidence, compare bin-wise accuracy (reliability
diagram), and summarize miscalibration as Expected Calibration Error (ECE).

When ECE is too high, fit a per-rank map from raw confidence -> P(correct)
on the in-distribution val split via Platt scaling (logistic) or isotonic
regression, then re-score held-out predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_RANKS = ("species", "genus", "family", "order")
CalibratorMethod = Literal["isotonic", "platt"]


@dataclass(frozen=True)
class ReliabilityBinSummary:
    """One equal-width confidence bin's empirical calibration stats."""

    bin_index: int
    conf_low: float
    conf_high: float
    mean_confidence: float
    accuracy: float
    count: int


@dataclass(frozen=True)
class RankCalibrationReport:
    rank: str
    n: int
    ece: float
    bins: list[ReliabilityBinSummary]


class ConfidenceCalibrator:
    """Per-rank post-hoc map from raw confidence in [0, 1] to P(correct)."""

    def __init__(self, method: CalibratorMethod = "isotonic") -> None:
        if method not in ("isotonic", "platt"):
            raise ValueError(f"Unknown calibrator method: {method!r}")
        self.method = method
        self._model: Any | None = None
        self._constant: float | None = None

    def fit(self, confidence: np.ndarray, correct: np.ndarray) -> "ConfidenceCalibrator":
        conf = np.asarray(confidence, dtype=np.float64).ravel()
        y = np.asarray(correct, dtype=np.float64).ravel()
        if len(conf) != len(y):
            raise ValueError("confidence and correct must have the same length")
        if len(conf) == 0:
            raise ValueError("cannot fit calibrator on empty arrays")

        # Degenerate labels: map everything to the empirical accuracy.
        if np.unique(y).size < 2:
            self._constant = float(y.mean())
            self._model = None
            return self

        self._constant = None
        if self.method == "isotonic":
            model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            model.fit(conf, y)
            self._model = model
        else:
            # Platt scaling: P(correct | conf) via logistic regression on the
            # raw score. With a single feature this is the classic Platt map.
            model = LogisticRegression(solver="lbfgs", max_iter=1000)
            model.fit(conf.reshape(-1, 1), y.astype(int))
            self._model = model
        return self

    def predict(self, confidence: np.ndarray | list[float]) -> np.ndarray:
        conf = np.asarray(confidence, dtype=np.float64).ravel()
        if self._constant is not None:
            return np.full(conf.shape, self._constant, dtype=np.float64)
        if self._model is None:
            raise RuntimeError("ConfidenceCalibrator.fit() must be called before predict()")
        if self.method == "isotonic":
            return np.clip(self._model.predict(conf), 0.0, 1.0)
        proba = self._model.predict_proba(conf.reshape(-1, 1))[:, 1]
        return np.clip(proba, 0.0, 1.0)


def reliability_bins(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> list[ReliabilityBinSummary]:
    """Equal-width [0, 1] bins of confidence vs. observed accuracy."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    conf = np.asarray(confidence, dtype=np.float64).ravel()
    y = np.asarray(correct, dtype=np.float64).ravel()
    if len(conf) != len(y):
        raise ValueError("confidence and correct must have the same length")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBinSummary] = []
    for i in range(n_bins):
        low, high = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            mask = (conf >= low) & (conf <= high)
        else:
            mask = (conf >= low) & (conf < high)
        count = int(mask.sum())
        if count == 0:
            mean_conf = float("nan")
            acc = float("nan")
        else:
            mean_conf = float(conf[mask].mean())
            acc = float(y[mask].mean())
        bins.append(
            ReliabilityBinSummary(
                bin_index=i,
                conf_low=low,
                conf_high=high,
                mean_confidence=mean_conf,
                accuracy=acc,
                count=count,
            )
        )
    return bins


def expected_calibration_error(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE = sum_b (n_b / N) * |acc(b) - conf(b)| over equal-width bins."""
    conf = np.asarray(confidence, dtype=np.float64).ravel()
    y = np.asarray(correct, dtype=np.float64).ravel()
    n = len(conf)
    if n == 0:
        return float("nan")
    ece = 0.0
    for b in reliability_bins(conf, y, n_bins=n_bins):
        if b.count == 0:
            continue
        ece += (b.count / n) * abs(b.accuracy - b.mean_confidence)
    return float(ece)


def evaluate_rank_calibration(
    confidence: np.ndarray,
    correct: np.ndarray,
    rank: str,
    n_bins: int = 10,
) -> RankCalibrationReport:
    return RankCalibrationReport(
        rank=rank,
        n=int(len(confidence)),
        ece=expected_calibration_error(confidence, correct, n_bins=n_bins),
        bins=reliability_bins(confidence, correct, n_bins=n_bins),
    )


def fit_rank_calibrators(
    confidence_by_rank: dict[str, np.ndarray],
    correct_by_rank: dict[str, np.ndarray],
    method: CalibratorMethod = "isotonic",
    ranks: tuple[str, ...] = _RANKS,
) -> dict[str, ConfidenceCalibrator]:
    """Fit one calibrator per rank from (confidence, correctness) pairs."""
    calibrators: dict[str, ConfidenceCalibrator] = {}
    for rank in ranks:
        if rank not in confidence_by_rank or rank not in correct_by_rank:
            continue
        calibrator = ConfidenceCalibrator(method=method)
        calibrator.fit(confidence_by_rank[rank], correct_by_rank[rank])
        calibrators[rank] = calibrator
    return calibrators


def leave_one_holdout_recalibrate(
    confidence: np.ndarray,
    correct: np.ndarray,
    holdout_fraction: np.ndarray,
    method: CalibratorMethod = "isotonic",
) -> np.ndarray:
    """Recalibrate with leave-one-holdout-out fits (no eval-holdout leakage).

    For each distinct ``holdout_fraction``, fit a calibrator on the other
    holdouts' rows and apply it to that holdout. Needed when in-distribution
    val calibrators do not transfer to clade-exclusion test queries.
    """
    conf = np.asarray(confidence, dtype=np.float64).ravel()
    y = np.asarray(correct, dtype=np.float64).ravel()
    groups = np.asarray(holdout_fraction).ravel()
    if not (len(conf) == len(y) == len(groups)):
        raise ValueError("confidence, correct, and holdout_fraction must align")
    if len(conf) == 0:
        return conf.copy()

    out = np.empty_like(conf)
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        # Can't LOHO with a single group; fall back to in-sample fit (biased).
        calibrator = ConfidenceCalibrator(method=method)
        calibrator.fit(conf, y)
        return calibrator.predict(conf)

    for group in unique_groups:
        train_mask = groups != group
        eval_mask = groups == group
        calibrator = ConfidenceCalibrator(method=method)
        calibrator.fit(conf[train_mask], y[train_mask])
        out[eval_mask] = calibrator.predict(conf[eval_mask])
    return out


def apply_rank_calibrators(
    confidence_by_rank: dict[str, np.ndarray],
    calibrators: dict[str, ConfidenceCalibrator],
) -> dict[str, np.ndarray]:
    """Transform raw confidences with fitted per-rank calibrators."""
    out: dict[str, np.ndarray] = {}
    for rank, conf in confidence_by_rank.items():
        if rank in calibrators:
            out[rank] = calibrators[rank].predict(conf)
        else:
            out[rank] = np.asarray(conf, dtype=np.float64).copy()
    return out


def plot_reliability_diagram(
    reports: list[RankCalibrationReport],
    output_path: str | Any,
    title: str = "Reliability diagram",
) -> None:
    """One curve per rank: mean predicted confidence vs. observed accuracy."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1.2, label="perfect")

    colors = {
        "species": "#1b7a3d",
        "genus": "#2a6f97",
        "family": "#c98a12",
        "order": "#b3261e",
    }
    for report in reports:
        xs = [b.mean_confidence for b in report.bins if b.count > 0]
        ys = [b.accuracy for b in report.bins if b.count > 0]
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.0,
            color=colors.get(report.rank, "#333333"),
            label=f"{report.rank} (ECE={report.ece:.3f})",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Predicted confidence (bin mean)")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
