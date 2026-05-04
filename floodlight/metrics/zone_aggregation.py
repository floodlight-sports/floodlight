import numpy as np
import pandas as pd

from floodlight import PlayerProperty, TeamProperty
from floodlight.utils.types import Numeric


def aggregate_property_by_zones(
    property_to_aggregate: PlayerProperty | TeamProperty,
    binning_property: PlayerProperty | TeamProperty,
    zones: list[tuple[Numeric, Numeric]],
    zone_names: list[str] | None = None,
    aggregation: str = "sum",
) -> pd.DataFrame:
    """Aggregates a property over threshold-based zones of another property.

    This function bins frames based on the value of ``binning_property`` and
    aggregates values from ``property_to_aggregate`` within each zone. Common use
    cases include calculating distance covered per velocity zone or time spent in
    different intensity zones [1]_.

    Parameters
    ----------
    property_to_aggregate: PlayerProperty or TeamProperty
        Property values to aggregate. For PlayerProperty, shape is (T, N) where T is
        the number of frames and N is the number of players. For TeamProperty, shape
        is (T,) where T is the number of frames.
    binning_property: PlayerProperty or TeamProperty
        Property values used to determine zone membership. Must have the same shape
        as ``property_to_aggregate``.
    zones: list[tuple[Numeric, Numeric]]
        List of (min, max) threshold tuples defining each zone. Zones use half-open
        intervals [min, max) where the minimum is inclusive and maximum is exclusive.
        For example, [(0, 2), (2, 4)] creates two zones: [0, 2) and [2, 4).
    zone_names: list[str], optional
        Names for each zone. If None, zones are named as "*min* to *max*". Must
        have the same length as ``zones`` if provided.
    aggregation: str, optional
        Aggregation function to apply within each zone. Options:

        - "sum": Sum of property values in zone
        - "count": Number of frames with valid data in zone
        - "mean": Average property value in zone
        - "min": Minimum property value in zone
        - "max": Maximum property value in zone

        Default is 'sum'.

    Returns
    -------
    zone_aggregates: pd.DataFrame
        DataFrame with aggregated values. For PlayerProperty inputs, rows correspond
        to players and columns to zones. For TeamProperty inputs, a single-row
        DataFrame is returned. Empty zones (no frames matching) return NaN for
        mean/min/max and 0 for sum/count.

    Notes
    -----
    Valid property combinations:

    - PlayerProperty by PlayerProperty: Both must have shape (T, N) with matching T and
      N
    - TeamProperty by TeamProperty: Both must have shape (T,) with matching T
    - PlayerProperty by TeamProperty: Aggregation property has shape (T, N),
      binning property has shape (T,). The binning values are broadcast across all
      players.

    Invalid combination:

    - TeamProperty by PlayerProperty: Cannot bin a single team value using
      player-specific thresholds.

    Frames where either property has NaN values are excluded from all aggregations.
    The boundary handling uses half-open intervals [min, max) to avoid ambiguity at
    zone boundaries.

    Examples
    --------
    Calculate distance covered in velocity zones for each player:

    >>> import numpy as np
    >>> from floodlight import PlayerProperty
    >>> from floodlight.metrics.zone_aggregation import aggregate_property_by_zones

    >>> # Create sample data: 4 frames, 2 players
    >>> distances = PlayerProperty(
    ...     property=np.array([[10, 5], [10, 5], [10, 5], [10, 5]], dtype=float),
    ...     name="distance"
    ... )
    >>> velocities = PlayerProperty(
    ...     property=np.array([[1, 6], [3, 8], [1, 6], [3, 8]], dtype=float),
    ...     name="velocity"
    ... )
    >>> # Define velocity zones (m/s)
    >>> zones = [(0, 2), (2, 4), (5, 9)]
    >>> zone_names = ["Low", "Medium", "High"]
    >>> result = aggregate_property_by_zones(
    ...     distances, velocities, zones, zone_names, aggregation='sum'
    ... )
    >>> result
       Low  Medium  High
    0  20.0    20.0   0.0
    1   0.0     0.0  20.0

    Calculate time spent (frame count) in metabolic power zones:

    >>> power = PlayerProperty(
    ...     property=np.array([[5, 15], [8, 25], [12, 30], [6, 18]], dtype=float),
    ...     name="power"
    ... )
    >>> zones = [(0, 10), (10, 20), (20, 35)]
    >>> result = aggregate_property_by_zones(
    ...     power, power, zones, aggregation='count'
    ... )
    >>> result
       0 to 10  10 to 20  20 to 35
    0      3.0       1.0       0.0
    1      0.0       2.0       2.0

    References
    ----------
    .. [1] `Miguel, M., Oliveira, R., Loureiro, N. Garcia-Rubio, J. & Ibáñez, S.
           (2021). Load Measures in Training/Match Monitoring in Soccer: A Systematic
           Review. International Journal of Environmental Research and Public Health,
           18(5), 2721.
           <https://www.mdpi.com/1660-4601/18/5/2721>`_
    """

    n_zones = len(zones)

    # Validate zone_names parameter
    if zone_names is not None and len(zone_names) != n_zones:
        raise ValueError(
            f"zone_names length ({len(zone_names)}) must match zones length ({n_zones})"
        )

    # Generate default zone names if not provided
    if zone_names is None:
        zone_names = [f"{min_val} to {max_val}" for min_val, max_val in zones]

    # Validate aggregation parameter
    agg_funcs = {
        "sum": lambda arr: np.nansum(arr, axis=0),
        "count": lambda arr: np.sum(~arr.mask & ~np.isnan(arr.data), axis=0),
        "mean": lambda arr: np.nanmean(arr, axis=0),
        "min": lambda arr: np.nanmin(arr, axis=0),
        "max": lambda arr: np.nanmax(arr, axis=0),
    }

    if aggregation not in agg_funcs:
        raise ValueError(
            f"aggregation must be one of {list(agg_funcs.keys())}, got '{aggregation}'"
        )

    agg_func = agg_funcs[aggregation]

    # Get property arrays
    prop_to_agg = property_to_aggregate.property
    binning_prop = binning_property.property

    # Validate property combination: TeamProperty by PlayerProperty is invalid
    if prop_to_agg.ndim == 1 and binning_prop.ndim == 2:
        raise ValueError(
            "Cannot aggregate TeamProperty by PlayerProperty. "
            "Valid combinations: PlayerProperty by PlayerProperty, "
            "TeamProperty by TeamProperty, or PlayerProperty by TeamProperty."
        )

    # Validate matching time dimension
    if prop_to_agg.shape[0] != binning_prop.shape[0]:
        raise ValueError(
            f"Time dimensions must match: property_to_aggregate has "
            f"{prop_to_agg.shape[0]} frames but binning_property has "
            f"{binning_prop.shape[0]} frames"
        )

    # Handle TeamProperty: reshape (T,) to (T, 1) for uniform processing
    if prop_to_agg.ndim == 1:
        prop_to_agg = prop_to_agg.reshape(-1, 1)
        n_entities = 1
    elif prop_to_agg.ndim == 2:
        n_entities = prop_to_agg.shape[1]
    else:
        raise ValueError(
            "property_to_aggregate must be 1D (TeamProperty) or 2D (PlayerProperty)"
        )

    if binning_prop.ndim == 1:
        binning_prop = binning_prop.reshape(-1, 1)
    elif binning_prop.ndim == 2:
        # For PlayerProperty by PlayerProperty, validate matching N dimension
        if prop_to_agg.shape[1] != binning_prop.shape[1]:
            raise ValueError(
                f"Player dimensions must match: property_to_aggregate has "
                f"{prop_to_agg.shape[1]} players but binning_property has "
                f"{binning_prop.shape[1]} players"
            )

    # Initialize output array
    aggregated_values = np.full((n_entities, n_zones), np.nan)

    # Loop over zones and aggregate
    for i, (min_val, max_val) in enumerate(zones):
        # Create mask for frames in this zone (half-open interval [min, max))
        in_zone_mask = np.bitwise_and(binning_prop >= min_val, binning_prop < max_val)

        # Broadcast mask if needed (PlayerProperty by TeamProperty case)
        if in_zone_mask.shape != prop_to_agg.shape:
            in_zone_mask = np.broadcast_to(in_zone_mask, prop_to_agg.shape)

        # Mask the property to aggregate
        masked_property = np.ma.masked_array(prop_to_agg, ~in_zone_mask)

        # Apply aggregation function
        aggregated_values[:, i] = agg_func(masked_property).data

    # Create DataFrame with zone names as columns
    zone_aggregates = pd.DataFrame(data=aggregated_values, columns=zone_names)

    return zone_aggregates
