import pytest
import numpy as np
from floodlight.metrics.apen import approx_entropy


@pytest.mark.unit
def test_logistic_map(check_vals=[0, 0.23, 0.45], precision=0.1) -> None:
    """Check results for logistic map according to Pincus (1991)"""

    def logistic_map(N, R, x_0=np.random.uniform(0.0, 1.0)):
        x = np.zeros(N)
        x[0] = x_0
        for i in range(N - 1):
            x[i + 1] = R * x[i] * (1 - x[i])
        return x

    R = [3.5, 3.6, 3.8]
    N = [300, 1000, 3000]
    apens_logistic = np.zeros(len(R) * len(N))
    counter = 0
    for r in R:
        log_map = logistic_map(np.max(N), r, x_0=0.5)
        for n in N:
            apens_logistic[counter] = approx_entropy(log_map[0:n], 2, 0.025)
            counter += 1
    apens_logistic.shape = (3, 3)
    assert np.all(np.abs(np.mean(apens_logistic, axis=1) - check_vals) < precision)


@pytest.mark.unit
def test_series_5(precision=0.1, reps=100) -> None:
    """Check results according to Pincus et al. (1991)"""
    signal = np.tile([1, 2, 1, 3], reps)
    chk_01 = np.abs(approx_entropy(signal, m=1, r=0.5) - 0.5 * np.log(2.0)) < precision
    chk_02 = np.abs(approx_entropy(signal, m=2, r=0.5)) < precision
    assert chk_01 and chk_02


@pytest.mark.unit
def test_series_7(precision=0.01, N=400) -> None:
    """Check results according to Pincus et al. (1991)"""
    signal = np.random.choice([0.002, -0.001], N) + np.tile([1, 0], int(N / 2))
    assert np.abs(approx_entropy(signal, m=2, r=0.5)) < precision
