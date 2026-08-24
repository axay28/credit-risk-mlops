import numpy as np
import pytest

from src.metrics import brier_score, gini, roc_auc, threshold_table


def test_roc_auc_perfect_and_reversed_rankings() -> None:
    labels = np.array([0, 0, 1, 1])

    assert roc_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)


def test_roc_auc_uses_average_ranks_for_tied_scores() -> None:
    labels = np.array([0, 1, 0, 1])
    tied_scores = np.array([0.5, 0.5, 0.5, 0.5])

    assert roc_auc(labels, tied_scores) == pytest.approx(0.5)
    assert gini(labels, tied_scores) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        (np.array([0, 1]), np.array([0.2])),
        (np.array([[0, 1]]), np.array([0.2, 0.8])),
        (np.array([0, 2]), np.array([0.2, 0.8])),
        (np.array([0, 1]), np.array([0.2, np.nan])),
    ],
)
def test_roc_auc_rejects_invalid_inputs(labels: np.ndarray, scores: np.ndarray) -> None:
    with pytest.raises(ValueError):
        roc_auc(labels, scores)


def test_brier_score_and_threshold_table() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.05, 0.2, 0.4, 0.8])
    exposure = np.array([100.0, 200.0, 300.0, 400.0])

    assert brier_score(labels, scores) == pytest.approx(0.110625)
    table = threshold_table(labels, scores, exposure, thresholds=np.array([0.3]))
    assert table.loc[0, "approval_rate"] == pytest.approx(0.5)
    assert table.loc[0, "approved_bad_rate"] == pytest.approx(0.0)
    assert table.loc[0, "defaults_captured"] == pytest.approx(1.0)
