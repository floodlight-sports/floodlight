import pytest
import numpy as np

from floodlight.metrics.entropy import approx_entropy


# Test approx_entropy function
@pytest.mark.unit
def test_approx_entropy_logistic_map(check_vals=[0, 0.23, 0.45], precision=0.1) -> None:
    """Check results for a logistic map according to Pincus (1991).

    References
    ----------
    Pincus, S. M. (1991). Approximate entropy as a measure of system complexity.
    Proceedings of the National Academy of Sciences, 88(6), 2297-2301.

    Parameters
    ----------
    check_vals: Values from Pincus (1991)
    precision: allowed deviation from test values.

    """

    def logistic_map(N, R, x_0=np.random.uniform(0.0, 1.0)):
        """Generates a signal from a logistic map system.

        Pincus(1991) equation [12]
        x_{i+1} = R x_i (1 - x_i)

        """
        x = np.zeros(N)
        x[0] = x_0
        for i in range(N - 1):
            x[i + 1] = R * x[i] * (1 - x[i])
        return x

    # Using the values from Pincus (1991)
    R = [3.5, 3.6, 3.8]  # Parameters for logistic map
    N = [300, 1000, 3000]  # length of signal
    apens_logistic = np.zeros(len(R) * len(N))
    counter = 0
    # iterate through combinations of R and N
    for r in R:
        log_map = logistic_map(np.max(N), r, x_0=0.5)
        for n in N:
            apens_logistic[counter] = approx_entropy(log_map[0:n], 2, 0.025)
            counter += 1
    apens_logistic.shape = (3, 3)

    assert np.all(np.abs(np.mean(apens_logistic, axis=1) - check_vals) < precision)


@pytest.mark.unit
def test_approx_entropy_series_5(precision=0.1, reps=100) -> None:
    """Check results according to Pincus et al. (1991).

    Pincus, S. M., Gladstone, I. M., & Ehrenkranz, R. A. (1991). A regularity statistic
    for  medical data analysis. Journal of clinical monitoring, 7(4), 335-345.
    """
    # signal consists of the consecutive pattern [1,2,3]
    signal = np.tile([1, 2, 1, 3], reps)
    # should equal zero
    assert np.abs(approx_entropy(signal, m=1, r=0.5) - 0.5 * np.log(2.0)) < precision
    # should equal zero
    assert np.abs(approx_entropy(signal, m=2, r=0.5)) < precision


@pytest.mark.unit
def test_approx_entropy_series_7(precision=0.01, N=400) -> None:
    """Check results according to Pincus et al. (1991)."""
    # signal consists of repeated [1,0] added with either 0.002 or -0.001
    # with selection probability p = 1/2.
    signal = np.random.choice([0.002, -0.001], N) + np.tile([1, 0], int(N / 2))
    # should equal zero
    assert np.abs(approx_entropy(signal, m=2, r=0.5)) < precision
