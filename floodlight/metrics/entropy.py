import numpy as np
import numpy.typing as npt


def approx_entropy(sig: npt.NDArray, m: int = 2, r: float = 0.5) -> float:
    """Calculates the Approximate Entropy ApEn(m,r) of sig according to Pincus (1991).
    [1]_

    Parameters
    ----------
    sig: np.array
        A time-series as np.ndarray with a single dimension (sig.ndim == 1).
    m: int, optional
        Comparison length of runs. Typically, m in {2,3}. Defaults to 2.
    r: float, optional
        Filtering level. Defaults to 0.5.

    Returns
    -------
    ApEn: float
        The Approximate Entropy of sig.

    Notes
    -----
    Time-series must be taken at equally spaced time points. Lower bound according to
    Pincus, Gladstone, Ehrenkranz (1991) is 50 time points [2]_. The filtering level r
    should be at least three times larger in magnitude as the noise.

    Rule of thumb: 0.1-0.25 of data STD.

    References
    ----------
    .. [1] `Pincus, S. M. (1991). Approximate entropy as a measure of system complexity.
            Proceedings of the National Academy of Sciences, 88(6), 2297-2301.
            <https://www.pnas.org/doi/10.1073/pnas.88.6.2297>`_
    .. [2] `Pincus, S. M., Gladstone, I. M., & Ehrenkranz, R. A. (1991). A regularity
            statistic for medical data analysis. Journal of clinical monitoring, 7(4),
            335-345.
            <https://link.springer.com/article/10.1007/BF01619355>`_
    """

    # sanity checks
    if type(sig) is not np.ndarray:
        raise TypeError(f"sig should be Numpy.ndarray, got {type(sig)}.")
    if sig.ndim != 1:
        raise TypeError(f"sig should have only a single dimension, got {sig.ndim}")
    if np.any(np.isnan(sig)):
        raise ValueError("Signal cannot contain Numpy.NaNs.")

    N = len(sig)

    def phi_m(m_):
        """Small helper function which calculates the sample entropy.

        Parameters
        ----------
        m: comparison length

        Returns
        -------
        Phi: sample entropy
        """
        no_parts = N - m_ + 1
        x_i_s = np.zeros((no_parts, m_))
        # determine reference patterns for chosen segment lengths
        for i in range(no_parts):
            x_i_s[i, :] = sig[i : (i + m_)]
        # placeholder for to determine pattern regularity
        c_i_m_r_s = np.zeros(no_parts)
        # iterate through all comparisons
        for i in range(no_parts):
            # determine the maximum distance between current reference pattern
            # and the remaining patterns
            d_i_j = np.max(np.abs(x_i_s - x_i_s[i, :]), axis=1)
            # Sum maximum distances across reference patterns
            c_i_m_r_s[i] = np.sum(d_i_j <= r)
        # calculate entropy
        return np.sum(np.log(c_i_m_r_s)) / no_parts - np.log(no_parts)

    # calculates the approximate entropy as the difference
    # between the entropies with two different consecutive segment
    # lengths.
    ap = phi_m(m) - phi_m(m + 1)
    # clamp minimum ap value to zero.
    if ap < np.finfo("float64").eps:
        ap = 0.0
    return ap
