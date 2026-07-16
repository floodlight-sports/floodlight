import json

import numpy as np

from floodlight.io.skillcorner import read_position_data_json


def test_read_position_data_associates_ball_id_by_value(tmp_path):
    ball_id = 98765432109876543210987654321
    home_id = 71000000000000000000000000001
    away_id = 72000000000000000000000000002
    match = {
        "home_team": {"id": 101},
        "away_team": {"id": 202},
        "referees": [],
        "ball": {"trackable_object": ball_id},
        "players": [
            {"team_id": 101, "trackable_object": home_id},
            {"team_id": 202, "trackable_object": away_id},
        ],
        "pitch_length": 105.0,
        "pitch_width": 68.0,
    }
    positions = [
        {
            "period": 1,
            "possession": {"group": "home team", "trackable_object": home_id},
            "data": [
                {"trackable_object": home_id, "x": 1.25, "y": 2.5},
                {"trackable_object": away_id, "x": -4.75, "y": 5.5},
                {"trackable_object": ball_id, "x": 12.5, "y": -3.25},
            ],
        }
    ]
    match_path = tmp_path / "match.json"
    positions_path = tmp_path / "positions.json"
    # Separate files require identifier linkage across independent JSON decodes.
    match_path.write_text(json.dumps(match), encoding="utf-8")
    positions_path.write_text(json.dumps(positions), encoding="utf-8")

    xy, _, _, _, _ = read_position_data_json(str(positions_path), str(match_path))

    ball = xy["firstHalf"]["Ball"]
    np.testing.assert_array_equal(ball.xy, [[12.5, -3.25]])
    assert np.isfinite(ball.xy).all()
    np.testing.assert_array_equal(xy["firstHalf"]["Home"].xy, [[1.25, 2.5]])
    np.testing.assert_array_equal(xy["firstHalf"]["Away"].xy, [[-4.75, 5.5]])
    assert ball.xy.shape == (1, 2)
    assert ball.framerate == 10
