import pandas as pd
from floodlight.core.events import Events


def parse_csv_to_event_object(csv_path):
    """
    Parse a CSV file into separate event objects based on the 'Timeline'(game) column. Each unique timeline (game) will
    result in one Event object.
    Parameters
    ----------
    csv_path : str
        Path to the input CSV file. The CSV should contain columns such as 'Row', 'Start time', 'Duration', and 'Timeline'.
    Returns
    -------
    list of Events
        A list containing `Events` objects, one for each unique game (grouped by 'Timeline').
    Notes
    -----
    The function performs the following steps:
    - Reads the CSV file.
    - Maps column names to internal attributes.
    - Calculates the 'endtime' as the sum of 'starttime' and 'duration'.
    - Groups events by 'timeline' to treat each unique timeline as a separate game.
    Example
    -------
    csv_path = "path_to_csv"
    events_objects = parse_csv_to_event_object(csv_path)
    """

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Create a mapping of column names to their intended event object attributes
    column_mapping = {
        "Row": "eID",  # Map the 'Row' column to 'eID'
        "Start time": "starttime",  # Map 'Start time' to 'starttime'
        "Duration": "duration",
        "Timeline": "timeline",
    }

    # Rename columns based on the mapping
    df = df.rename(columns=column_mapping)

    # Calculate the endtime
    df["endtime"] = df["starttime"] + df["duration"]

    # Assign the 'gameclock' column to be the same as 'starttime' (because mandatory for floodlight)
    df["gameclock"] = df["starttime"]

    # Group by the 'timeline' (which represents a match/game)
    grouped = df.groupby("timeline")

    events_objects = []

    # Process each group and create an Events object for each game
    for _, group in grouped:
        # Create an Events object for the current group (game)
        events = Events(group)
        events_objects.append(events)

    return events_objects