import numpy as np
import numpy.typing as npt


def approx_entropy(sig: npt.NDArray, m: int = 2, r: float = 0.5) -> float:
    """Calculates the Approximate Entropy ApEn(m,r) of sig according to Pincus (1991).

    Notes
    -----
        Time-series must be taken at equally spaced time points.
        Lower bound according to Pincus, Gladstone, Ehrenkranz (1991) is 50 time points.
        The filtering level r should be at least three times larger in magnitude
        as the noise. Rule of thumb: 0.1-0.25 of data STD.

    References:
        Pincus, S. M. (1991). Approximate entropy as a measure of system complexity.
            Proceedings of the National Academy of Sciences, 88(6), 2297-2301.
        Pincus, S. M., Gladstone, I. M., & Ehrenkranz, R. A. (1991). A regularity
            statistic for medical data analysis. Journal of clinical monitoring, 7(4),
            335-345.

    Parameters
    ----------
    sig: time-series as numpy array
    m: comparison length of runs (integer), typical m in {2,3}
    r: filtering level (real number)

    Returns
    -------
    ApEn: float - the Approximate Entropy of sig
    """
    N = len(sig)

    def phi_m(m_):
        no_parts = N - m_ + 1
        x_i_s = np.zeros((no_parts, m_))
        for i in range(no_parts):
            x_i_s[i, :] = sig[i : (i + m_)]
        c_i_m_r_s = np.zeros(no_parts)
        for i in range(no_parts):
            d_i_j = np.max(np.abs(x_i_s - x_i_s[i, :]), axis=1)
            c_i_m_r_s[i] = np.sum(d_i_j <= r)
        return np.sum(np.log(c_i_m_r_s)) / no_parts - np.log(no_parts)

    ap = phi_m(m) - phi_m(m + 1)
    if ap < np.finfo("float64").eps:
        ap = 0.0
    return ap
