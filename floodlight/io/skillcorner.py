import json
from typing import Dict, Tuple, Union, List

import numpy as np
import pandas as pd

from floodlight import Teamsheet, Code, XY, Pitch


def _get_meta_data(
    match_data: Dict[str, Union[List, str, Dict, int]]
) -> Tuple[int, int, List[int], int]:
    """
    Extract metadata from match data including team IDs, referee IDs, and ball ID.

    Parameters
    ----------
    match_data : dict
        A dictionary containing match data with keys 'home_team', 'away_team',
        'referees', and 'ball'. It is returned when importing the match_data.json
        provided by Skillcorner for each match.

    Returns
    -------
    home_team_id : int
        The ID of the home team.
    away_team_id : int
        The ID of the away team.
    referee_ids : list of int
        A list of trackable object IDs for the referees.
    ball_id : int
        The trackable object ID for the ball.

    """
    home_team_id = match_data["home_team"]["id"]
    away_team_id = match_data["away_team"]["id"]

    referee_ids = [x["trackable_object"] for x in match_data["referees"]]
    ball_id = match_data["ball"]["trackable_object"]

    return home_team_id, away_team_id, referee_ids, ball_id


def _get_teamsheets(
    match_data: Dict[str, Union[List, str, Dict, int]]
) -> Dict[str, Teamsheet]:
    """
    Extract teamsheets from match data and return them as `Teamsheet` objects.

    Parameters
    ----------
    match_data : dict
        A dictionary containing match data with keys 'home_team', 'away_team',
        'players', and 'ball'. It is returned when importing the match_data.json
        provided by Skillcorner for each match.

    Returns
    -------
    teamsheets: dict
        A dictionary containing teamsheets for 'Home' and 'Away'.
        Each teamsheet is an instance of the `Teamsheet` class, which includes
        additional player information.
    """
    home_team_id, away_team_id, _, _ = _get_meta_data(match_data)

    players = {
        "Home": [x for x in match_data["players"] if x["team_id"] == home_team_id],
        "Away": [x for x in match_data["players"] if x["team_id"] == away_team_id],
    }

    teamsheets = {}
    for team in players:
        team_df = pd.DataFrame(players[team])
        # add essential column
        team_df["player"] = team_df["trackable_object"]
        teamsheets.update({team: Teamsheet(team_df)})
        teamsheets[team].add_xIDs()

    return teamsheets


def _get_pitch_from_match_data(
    match_data: Dict[str, Union[List, str, Dict, int]]
) -> Pitch:
    """
    Create a Pitch object based on the dimensions provided in the match data.

    Parameters
    ----------
    match_data : dict
        A dictionary containing match data with keys 'pitch_length' and 'pitch_width'.
        It is returned by the `_get_meta_data()`-function that extracts relevant
        information from the match_data.json content.

    Returns
    -------
    pitch: Pitch
         Pitch object with actual pitch length and width.
    """

    pitch_length = match_data["pitch_length"]
    pitch_width = match_data["pitch_width"]
    pitch = Pitch(
        xlim=(-pitch_length / 2, pitch_length / 2),
        ylim=(-pitch_width / 2, pitch_width / 2),
        unit="m",
        boundaries="flexible",
        sport="football",
    )

    return pitch


def read_position_data_json(
    file_path_structured_data: str,
    file_path_match_data: str,
    teamsheet_home: Teamsheet = None,
    teamsheet_away: Teamsheet = None,
) -> tuple[
    Dict[str, Dict[str, XY]],
    Dict[str, Dict[str, Code]],
    Dict[str, Dict[str, Code]],
    Dict[str, Teamsheet],
    Pitch,
]:
    """
    Parse and process position data and match data from the `SkillCorner Open Dataset
    <https://github.com/SkillCorner/opendata>`_ from disk.

    This dataset was published by `SkillCorner <https://www.skillcorner.com/>`_
    in a joint initiative with `Friends Of Tracking
    <https://www.youtube.com/channel/UCUBFJYcag8j2rm_9HkrrA7w>`_. It contains 9 matches
    of broadcast tracking data in JSON format. The data for each match is structured
    in two separate files. The match data file contains meta information about the
    match and its competitors, like team and player's identifiers. The structured data
    file contains the position data for each frame.

    Parameters
    ----------
    file_path_structured_data : str
        The file path to the positions JSON file.
    file_path_match_data : str
        The file path to the match data JSON file.
    teamsheet_home : Teamsheet, optional
        The teamsheet for the home team (default is None).
    teamsheet_away : Teamsheet, optional
        The teamsheet for the away team (default is None).

    Returns
    -------
    xy_objects : dict
        A dictionary containing XY objects for each half and each team.
    possession_objects_team : dict
        A dictionary containing team possession codes for each half.
    possession_objects_player : dict
        A dictionary containing player possession codes for each half.
    teamsheets : dict
        A dictionary containing the teamsheets for 'Home', 'Away', and 'Ball'.
    pitch : Pitch
        The Pitch object representing the match pitch.
    """

    # hard coded framerate, as it is specified in the documentation of the dataset
    framerate = 10
    # mapping for Code-object definitions attribute
    home_away_link = {"home team": 1, "away team": 2}

    with open(file_path_match_data) as f:
        match_data = json.load(f)
    with open(file_path_structured_data) as f:
        positions = json.load(f)

    pitch = _get_pitch_from_match_data(match_data)
    home_team_id, away_team_id, referee_ids, ball_id = _get_meta_data(match_data)
    teamsheets = _get_teamsheets(match_data)

    if teamsheet_home is not None:
        teamsheets["Home"] = teamsheet_home
    if teamsheet_away is not None:
        teamsheets["Away"] = teamsheet_away

    pID_links = {}
    for team in teamsheets:
        pID_links.update({team: teamsheets[team].get_links("player", "xID")})

    frames = {
        "firstHalf": [x for x in positions if x["period"] == 1],
        "secondHalf": [x for x in positions if x["period"] == 2],
    }

    # create empty return objects
    xy_objects = {}
    possession_objects_team = {}
    possession_objects_player = {}
    # pre-allocate nd-arrays for xy- and possession-objects
    for half in frames:
        xy_objects.update({half: {}})
        possession_objects_team.update({half: np.full((len(frames[half]),), np.nan)})
        possession_objects_player.update({half: np.full((len(frames[half]),), np.nan)})
        for team in pID_links:
            xy_objects[half].update(
                {team: np.full((len(frames[half]), len(teamsheets[team]) * 2), np.nan)}
            )
        xy_objects[half].update({"Ball": np.full((len(frames[half]), 2), np.nan)})

    # loop through half times
    for half in frames:
        # loop through frames
        for t, frame in enumerate(frames[half]):
            team_possession = frame["possession"].get("group")
            player_possession = frame["possession"].get("trackable_object")
            # if ball possession is available, assign team- and player-possession
            if team_possession is not None:
                possession_objects_team[half][t] = home_away_link[team_possession]
            if player_possession in pID_links["Home"]:
                possession_objects_player[half][t] = pID_links["Home"][
                    player_possession
                ]
            elif player_possession in pID_links["Away"]:
                possession_objects_player[half][t] = pID_links["Away"][
                    player_possession
                ]

            # loop through players
            for tracked_object in frame["data"]:
                # trackable_object as identifier is missing for about 5% of the frames
                pID = tracked_object.get("trackable_object")
                # write x/y-coordinates into respective teams and players columns
                if pID in pID_links["Home"]:
                    x_column = pID_links["Home"][tracked_object["trackable_object"]] * 2
                    y_column = x_column + 1
                    xy_objects[half]["Home"][t, x_column] = tracked_object["x"]
                    xy_objects[half]["Home"][t, y_column] = tracked_object["y"]
                elif pID in pID_links["Away"]:
                    x_column = pID_links["Away"][tracked_object["trackable_object"]] * 2
                    y_column = x_column + 1
                    xy_objects[half]["Away"][t, x_column] = tracked_object["x"]
                    xy_objects[half]["Away"][t, y_column] = tracked_object["y"]
                elif pID is ball_id:
                    xy_objects[half]["Ball"][t, 0] = tracked_object["x"]
                    xy_objects[half]["Ball"][t, 1] = tracked_object["y"]
                # do not track referee
                elif pID in referee_ids:
                    pass
                # if the trackable_object ID is not available, it's impossible to assign
                # the coordinates to a team or player.
                elif pID is None:
                    pass

    # write nd-arrays into XY- and Code-Objects
    for half in xy_objects:
        possession_objects_team[half] = Code(
            possession_objects_team[half], "team_possession", {1: "Home", 2: "Away"}
        )
        possession_objects_player[half] = Code(
            possession_objects_player[half], "player_possession"
        )
        for team in xy_objects[half]:
            xy_objects[half][team] = XY(xy_objects[half][team], framerate=framerate)

    return (
        xy_objects,
        possession_objects_team,
        possession_objects_player,
        teamsheets,
        pitch,
    )
