from floodlight import Teamsheet, Code, XY, Pitch
from floodlight.core.events import Events

from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np

from datetime import timedelta


def _get_metadata(metadata: "Metadata") -> Tuple[Dict, Dict, Dict, Pitch]:
    """Reads metadata from kloppy Metadata object and extracts information about match
    metainfo, periods, playing directions, and the pitch.

    Parameters
    ----------
    metadata: Metadata
        Kloppy Metadata object containing match information.

    Returns
    -------
    metadata_dict: dict
        Dictionary with meta information such as framerate.
    periods: Dict[str, Tuple[int, int]]
        Dictionary with start and endframes of all segments, e.g.,
        ``periods[segment] = (startframe, endframe)``.
    directions: Dict[str, Dict[str, str]]
        Dictionary with playing direction information of all segments and teams,
        e.g., ``directions[segment][team] = 'lr'``
    pitch: Pitch
        Pitch object with actual pitch length and width.
    """
    from kloppy.domain import AttackingDirection

    home_team, away_team = metadata.teams

    metadata_dict = {}
    periods = {}
    directions = {}

    metadata_dict["framerate"] = metadata.frame_rate
    metadata_dict["length"] = (
        metadata.pitch_dimensions.x_dim if metadata.pitch_dimensions else None
    )
    metadata_dict["width"] = (
        metadata.pitch_dimensions.y_dim if metadata.pitch_dimensions else None
    )
    metadata_dict["home_tID"] = home_team.team_id
    metadata_dict["away_tID"] = away_team.team_id

    for period in metadata.periods:
        segment = f"HT{period.id}"

        if isinstance(period.start_timestamp, timedelta) and isinstance(
            period.end_timestamp, timedelta
        ):
            if metadata.frame_rate:
                start_frame = int(
                    period.start_timestamp.total_seconds() * metadata.frame_rate
                )
                end_frame = int(
                    period.end_timestamp.total_seconds() * metadata.frame_rate
                )
                periods[segment] = (start_frame, end_frame + 1)
            else:
                start_frame = int(period.start_timestamp.total_seconds())
                end_frame = int(period.end_timestamp.total_seconds())
                periods[segment] = (start_frame, end_frame + 1)
        else:
            periods[segment] = (0, 0)  # Placeholder

    # Process playing directions
    for period in metadata.periods:
        segment = f"HT{period.id}"
        directions[segment] = {}

        # Get attacking direction for home team for this period
        home_attacking_direction = AttackingDirection.from_orientation(
            orientation=metadata.orientation, period=period
        )

        # Set directions based on attacking direction
        if home_attacking_direction == AttackingDirection.LTR:
            directions[segment]["Home"] = "lr"
            directions[segment]["Away"] = "rl"
        elif home_attacking_direction == AttackingDirection.RTL:
            directions[segment]["Home"] = "rl"
            directions[segment]["Away"] = "lr"
        else:
            # Default or NOT_SET case
            directions[segment]["Home"] = "lr"
            directions[segment]["Away"] = "rl"

    pitch = Pitch.from_template(
        template_name="secondspectrum",  # or appropriate template
        length=metadata.coordinate_system.pitch_length,
        width=metadata.coordinate_system.pitch_width,
        sport="football",
    )

    return metadata_dict, periods, directions, pitch


def _get_teamsheets(metadata: "MetaData") -> Dict[str, Teamsheet]:
    """
    Convert kloppy Metadata object to teamsheets structure.

    Args:
        metadata: Kloppy Metadata object containing teams data

    Returns:
        dict: Teamsheets dictionary with "Home" and "Away" keys
    """
    # Extract teams
    home_team, away_team = metadata.teams

    # Initialize teamsheets structure
    teamsheets = {team: None for team in ["Home", "Away"]}

    # Map teams to their ground designation
    team_mapping = {"Home": home_team, "Away": away_team}

    # Process each team
    for team_ground in ["Home", "Away"]:
        team_obj = team_mapping[team_ground]

        # Initialize teamsheet structure
        teamsheet = {
            column: [] for column in ["precedence", "player", "jID", "pID", "position"]
        }

        # Process each player in the team
        for player in team_obj.players:
            if player.name is not None:
                name = player.full_name
            elif player.first_name is not None and player.last_name is not None:
                name = f"{player.first_name} {player.last_name}"
            elif player.last_name is not None:
                name = player.last_name
            else:
                name = f"{player.team.ground}_{player.jersey_no}"

            position = (
                player.starting_position.code
                if player.starting_position is not None
                else "UNK"
            )
            jID = player.jersey_no
            pID = player.player_id
            precedence = _get_position_precedence(player.starting_position)

            # Add to teamsheet
            teamsheet["player"].append(name)
            teamsheet["position"].append(position)
            teamsheet["jID"].append(jID)
            teamsheet["pID"].append(pID)
            teamsheet["precedence"].append(precedence)

        # Convert to DataFrame and sort
        teamsheet_df = pd.DataFrame(teamsheet)
        teamsheet_df.sort_values("precedence", inplace=True)
        teamsheet_df.drop(["precedence"], axis=1, inplace=True)
        teamsheet_df.reset_index(drop=True, inplace=True)

        # Convert to Teamsheet object (assuming you have this class defined)
        teamsheet_obj = Teamsheet(teamsheet_df)
        teamsheets[team_ground] = teamsheet_obj

    return teamsheets


def _get_position_precedence(position):
    """
    Get precedence value for position sorting based on PositionType hierarchy.
    The precedence is simply the order in which positions are defined in the enum.

    Args:
        position: PositionType enum value

    Returns:
        int: Precedence value for sorting (lower = higher precedence)
    """
    from kloppy.domain import PositionType

    if position is None or position == PositionType.Unknown:
        return 999

    return list(PositionType).index(position)


def get_position_data(
    tracking_dataset: "TrackingDataset",
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Convert kloppy tracking dataset to floodlight objects structure.

    Parameters
    ----------
    tracking_dataset : kloppy tracking dataset
        Kloppy tracking dataset with to_df() method

    Returns
    -------
    xy_objects : dict
        Dictionary with XY objects for each segment and team
    possession_objects : dict
        Dictionary with possession Code objects for each segment
    ballstatus_objects : dict
        Dictionary with ball status Code objects for each segment
    """

    try:
        from kloppy.domain import TrackingDataset
    except ImportError:
        raise ImportError(
            "Seems like you don't have kloppy installed. Please"
            " install it using: pip install kloppy"
        )

    if not isinstance(tracking_dataset, TrackingDataset):
        raise TypeError(
            "'tracking_dataset' should be of type kloppy.domain.TrackingDataset"
        )

    # Get metadata information
    metadata_dict, periods, directions, pitch = _get_metadata(tracking_dataset.metadata)
    teamsheets = _get_teamsheets(tracking_dataset.metadata)

    teamsheet_home = teamsheets["Home"]
    teamsheet_away = teamsheets["Away"]
    if "xID" not in teamsheet_home.teamsheet.columns:
        teamsheet_home.add_xIDs()
    if "xID" not in teamsheet_away.teamsheet.columns:
        teamsheet_away.add_xIDs()

    segments = list(periods.keys())
    fps = metadata_dict["framerate"]

    df = tracking_dataset.to_df(engine="pandas")

    home_team, away_team = tracking_dataset.metadata.teams
    home_team_id = home_team.team_id
    away_team_id = away_team.team_id

    xy_objects = {}
    possession_objects = {}
    ballstatus_objects = {}

    for i, period in enumerate(tracking_dataset.metadata.periods):
        segment = segments[i]

        segment_df = df[df["period_id"] == period.id].copy()
        xy_objects[segment] = {}

        possession_code = _create_possession_code(
            segment_df, home_team_id, away_team_id
        )
        possession_objects[segment] = Code(
            code=possession_code,
            name="possession",
            definitions={"H": "Home", "A": "Away"},
            framerate=fps,
        )

        ballstatus_code = _create_ballstatus_code(segment_df)
        ballstatus_objects[segment] = Code(
            code=ballstatus_code,
            name="ballstatus",
            definitions={"D": "Dead", "A": "Alive"},
            framerate=fps,
        )

        for team_name in ["Home", "Away"]:
            team_obj = home_team if team_name == "Home" else away_team
            teamsheet = teamsheet_home if team_name == "Home" else teamsheet_away

            xy_data = _extract_team_xy_data(segment_df, team_obj, teamsheet)

            xy_objects[segment][team_name] = XY(
                xy=xy_data,
                framerate=fps,
                direction=directions[segment][team_name],
            )

        ball_xy_data = np.column_stack(
            [segment_df["ball_x"].values, segment_df["ball_y"].values]
        )
        xy_objects[segment]["Ball"] = XY(xy=ball_xy_data, framerate=fps)

    data_objects = (
        xy_objects,
        possession_objects,
        ballstatus_objects,
        teamsheets,
        pitch,
    )

    return data_objects


def _create_possession_code(
    df: pd.DataFrame, home_team_id: str, away_team_id: str
) -> np.ndarray:
    """
    Create possession code array from dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Segment dataframe with ball_owning_team_id column
    home_team_id : str
        Home team identifier
    away_team_id : str
        Away team identifier

    Returns
    -------
    possession_code : np.ndarray
        Array with 'H' for home possession, 'A' for away possession
    """
    possession_code = np.full(len(df), np.nan, dtype=object)  # Default to home

    away_mask = df["ball_owning_team_id"] == away_team_id
    possession_code[away_mask] = "A"

    home_mask = df["ball_owning_team_id"] == home_team_id
    possession_code[home_mask] = "H"

    return possession_code


def _create_ballstatus_code(df: pd.DataFrame) -> np.ndarray:
    """
    Create ball status code array from dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Segment dataframe with ball_state column

    Returns
    -------
    ballstatus_code : np.ndarray
        Array with 'A' for alive, 'D' for dead
    """
    from kloppy.domain import BallState

    ballstatus_code = np.full(len(df), "A", dtype=object)

    dead_mask = df["ball_state"].astype(str) == BallState.DEAD.value
    ballstatus_code[dead_mask] = "D"

    return ballstatus_code


def _create_ballstatus_code(df: pd.DataFrame) -> np.ndarray:
    """
    Create ball status code array from dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Segment dataframe with ball_state column

    Returns
    -------
    ballstatus_code : np.ndarray
        Array with 'A' for alive, 'D' for dead
    """
    ballstatus_code = np.full(len(df), "A", dtype=object)  # Default to alive

    if "ball_state" in df.columns:
        # Map BallState enum values to A/D
        dead_mask = df["ball_state"].astype(str) == "dead"
        ballstatus_code[dead_mask] = "D"

    return ballstatus_code


def _extract_team_xy_data(
    df: pd.DataFrame, team: "Team", teamsheet: "Teamsheet"
) -> np.ndarray:
    """
    Extract XY coordinate data for a team from dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Segment dataframe
    team : Team
        Kloppy Team object
    teamsheet : Teamsheet
        Teamsheet object with xID mappings

    Returns
    -------
    xy_data : np.ndarray
        Array with shape (n_frames, 2*n_players) containing x,y coordinates
    """
    # Ensure teamsheet has xIDs
    if "xID" not in teamsheet.teamsheet.columns:
        teamsheet.add_xIDs()

    # Get mapping from jersey number to array index (xID)
    links_jID_to_xID = teamsheet.get_links("jID", "xID")

    # Get the maximum xID to determine array size
    max_xID = max(links_jID_to_xID.values()) if links_jID_to_xID else 0
    n_players = max_xID + 1  # xID is 0-indexed

    # Initialize xy_data array with NaN values
    xy_data = np.full((len(df), 2 * n_players), np.nan)

    # Fill data for each player
    for player in team.players:
        player_id = player.player_id
        jersey_no = player.jersey_no

        # Check if this player has data in the dataframe
        x_col = f"{player_id}_x"
        y_col = f"{player_id}_y"

        if (
            x_col in df.columns
            and y_col in df.columns
            and jersey_no in links_jID_to_xID
        ):
            xID = links_jID_to_xID[jersey_no]

            x_idx = xID * 2
            y_idx = xID * 2 + 1

            xy_data[:, x_idx] = df[x_col].values
            xy_data[:, y_idx] = df[y_col].values

    return xy_data


def get_event_data(
    event_dataset: "EventDataset",
) -> Tuple[Dict[str, Dict[str, Events]], Pitch]:
    """
    Convert kloppy event dataset to segmented team dataframes structure.

    Parameters
    ----------
    event_dataset : kloppy event dataset
        Kloppy event dataset with to_df() method

    Returns
    -------
    team_dfs : dict
        Dictionary with team dataframes for each segment
        Structure: team_dfs[segment][team_id] = DataFrame
    """
    try:
        from kloppy.domain import EventDataset
    except ImportError:
        raise ImportError(
            "Seems like you don't have kloppy installed. Please"
            " install it using: pip install kloppy"
        )

    if not isinstance(event_dataset, EventDataset):
        raise TypeError(
            "'tracking_dataset' should be of type kloppy.domain.TrackingDataset"
        )

    # Get metadata information
    metadata_dict, periods, directions, pitch = _get_metadata(event_dataset.metadata)
    segments = list(periods.keys())

    # Convert event dataset to DataFrame
    events_df = event_dataset.to_df(
        "*", lambda event: {"qualifier": event.raw_event}, engine="pandas"
    )
    events_df = _process_events(events_df)
    # Get team information
    home_team, away_team = event_dataset.metadata.teams

    # Initialize team_dfs structure
    team_dfs = {segment: {} for segment in segments}

    # Process each segment
    for i, period in enumerate(event_dataset.metadata.periods):
        segment = segments[i]

        segment_events = events_df[events_df["period_id"] == period.id].copy()

        # Split events by team
        for team in [home_team, away_team]:
            # Get events for this team
            team_events = segment_events[segment_events["tID"] == team.team_id].copy()

            ground = team.ground.value.capitalize()
            # Ensure columns are in standard order
            team_dfs[segment][ground] = team_events[
                [
                    "eID",
                    "gameclock",
                    "tID",
                    "pID",
                    "outcome",
                    "timestamp",
                    "minute",
                    "second",
                    "qualifier",
                ]
            ]

            # Sort by gameclock and reset index
            team_dfs[segment][ground] = team_dfs[segment][ground].sort_values(
                "gameclock"
            )
            team_dfs[segment][ground] = team_dfs[segment][ground].reset_index(drop=True)

    events_objects = {}
    for segment in segments:
        events_objects[segment] = {}
        for team in ["Home", "Away"]:
            events_objects[segment][team] = Events(
                events=pd.DataFrame(data=team_dfs[segment][team]),
                direction=directions[segment][team],
            )
    return events_objects


def _process_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process all events using vectorized pandas operations.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events dataframe from kloppy
    is_successful : list
        List of successful result types

    Returns
    -------
    processed_df : pd.DataFrame
        Processed events dataframe
    """
    from kloppy.domain import (
        CarryResult,
        DuelResult,
        EventType,
        InterceptionResult,
        PassResult,
        ShotResult,
        TakeOnResult,
    )

    IS_SUCCESSFUL = [
        ShotResult.GOAL.value,
        ShotResult.OWN_GOAL.value,
        PassResult.COMPLETE.value,
        TakeOnResult.COMPLETE.value,
        CarryResult.COMPLETE.value,
        DuelResult.WON.value,
        InterceptionResult.SUCCESS.value,
    ]

    events_df["gameclock"] = events_df["timestamp"].dt.total_seconds()
    events_df["minute"] = (events_df["gameclock"] // 60).astype(int)
    events_df["second"] = (events_df["gameclock"] % 60).astype(int)

    result_strings = events_df["result"].astype(str)
    events_df["outcome"] = result_strings.isin(IS_SUCCESSFUL).astype(float)

    events_df.rename(
        columns={"event_type": "eID", "team_id": "tID", "player_id": "pID"},
        inplace=True,
    )

    return events_df
